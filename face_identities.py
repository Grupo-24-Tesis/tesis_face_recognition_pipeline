import argparse
import ctypes
import glob
import json
import os
import sysconfig
import time

STARTED = time.monotonic()
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
for lib in glob.glob(os.path.join(sysconfig.get_paths()["purelib"],
                                  "nvidia", "cusolver", "lib", "libcusolver.so.*")):
    ctypes.CDLL(lib)

import cv2
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace

from face_recognition import (OUTPUT_MARGIN_BOTTOM, OUTPUT_MARGIN_SIDE, OUTPUT_MARGIN_TOP,
                              RESULTS_NAME, crop_face, rank_key, sharpness)
from face_embedding import EMBED_CHUNK, embed_faces, unit_embedding
from face_sampling import (ADAPTIVE, FIXED, MODE_LABELS, advance, hold_frames, sampling_steps,
                           start_state)
from identity_linking import env_flag, link_identities

FACES_DIR = os.path.join("database", "faces_identidades")
RETINA_THRESHOLD = 0.9
RETINA_ALLOW_UPSCALING = env_flag("FACE_DETECT_UPSCALE", True)
EMBEDDING_MODEL = "GhostFaceNet"
EMBED_BATCH = env_flag("FACE_EMBED_BATCH", True)
ALIGNED_SIZE = 112
ARCFACE_TEMPLATE = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                             [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
IDENTITIES_NAME = "identities.json"
BOX_COLOR = (0, 255, 0)


def open_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")
    return cap


def read_frames(cap, first, next_target):
    frame_idx, target = -1, first
    while target is not None:
        if not cap.grab():
            return
        frame_idx += 1
        if frame_idx < target:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            return
        yield frame_idx, frame
        target = next_target(frame_idx)


def aligned_face(frame, landmarks):
    eyes = sorted([landmarks["right_eye"], landmarks["left_eye"]], key=lambda p: p[0])
    mouth = sorted([landmarks["mouth_right"], landmarks["mouth_left"]], key=lambda p: p[0])
    src = np.array([eyes[0], eyes[1], landmarks["nose"], mouth[0], mouth[1]], dtype=np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(src, ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if matrix is None:
        return None
    return cv2.warpAffine(frame, matrix, (ALIGNED_SIZE, ALIGNED_SIZE), borderValue=0)


def embedding(aligned):
    rep = DeepFace.represent(aligned, model_name=EMBEDDING_MODEL,
                             detector_backend="skip", align=False)
    return unit_embedding(rep[0]["embedding"])


def add_embeddings(faces):
    if not faces:
        return
    aligned_faces = [face.pop("aligned") for face in faces]
    if EMBED_BATCH:
        model = DeepFace.build_model(EMBEDDING_MODEL, task="facial_recognition")
        values = embed_faces(model.forward, aligned_faces, model.input_shape)
    else:
        values = [embedding(aligned) for aligned in aligned_faces]
    for face, emb in zip(faces, values):
        face["embedding"] = emb


def detect_faces(frame_idx, frame, sample, interval_s):
    resp = RetinaFace.detect_faces(frame, threshold=RETINA_THRESHOLD,
                                   allow_upscaling=RETINA_ALLOW_UPSCALING)
    if not isinstance(resp, dict):
        return []
    faces = []
    for face in resp.values():
        x1, y1, x2, y2 = [int(v) for v in face["facial_area"]]
        box_crop = frame[max(0, y1):y2, max(0, x1):x2]
        if box_crop.size == 0:
            continue
        aligned = aligned_face(frame, face["landmarks"])
        if aligned is None:
            continue
        faces.append({
            "frame": frame_idx,
            "sample": sample,
            "interval_s": interval_s,
            "box": [x1, y1, x2, y2],
            "score": float(face["score"]),
            "px": min(x2 - x1, y2 - y1),
            "sharpness": sharpness(box_crop),
            "aligned": aligned,
        })
    return faces


def ranking(face):
    x1, y1, x2, y2 = face["box"]
    return rank_key({"face_px": (x2 - x1) * (y2 - y1),
                     "sharpness": face["sharpness"], "score": face["score"]})


def analyse_video(cap, fps):
    steps = sampling_steps(fps, ADAPTIVE)
    hold = hold_frames(fps)
    state = start_state(ADAPTIVE)
    detections, pending, sampled = [], [], []
    for frame_idx, frame in read_frames(cap, 0, lambda idx: idx + steps[state["mode"]]):
        mode = state["mode"]
        faces = detect_faces(frame_idx, frame, len(sampled), steps[mode] / fps)
        sampled.append((frame_idx, mode))
        detections.extend(faces)
        pending.extend(faces)
        if not EMBED_BATCH or len(pending) >= EMBED_CHUNK:
            add_embeddings(pending)
            pending = []
        state = advance(state, [face["box"] for face in faces], frame_idx, hold)
    add_embeddings(pending)
    return detections, sampled, steps


def render_outputs(video_path, sampled, base_step, fps, identities, faces_dir, annotated_path):
    best = {}
    labels = {}
    for identity_id, faces in enumerate(identities, start=1):
        winner = max(faces, key=ranking)
        best.setdefault(winner["frame"], []).append((identity_id, winner))
        for face in faces:
            labels.setdefault(face["frame"], []).append((identity_id, face["box"]))

    if not best and not annotated_path:
        return {}, False

    saved = {}
    last_needed = max(best) if best else -1
    repeats = {a: max(1, round((b - a) / base_step)) for a, b in zip(sampled, sampled[1:])}
    targets = iter(sampled[1:])
    writer = None
    cap = open_video(video_path)
    try:
        for frame_idx, frame in read_frames(cap, sampled[0] if sampled else None,
                                            lambda _: next(targets, None)):
            for identity_id, winner in best.get(frame_idx, []):
                crop = crop_face(frame, winner["box"], OUTPUT_MARGIN_SIDE,
                                 OUTPUT_MARGIN_TOP, OUTPUT_MARGIN_BOTTOM)
                if crop is None or crop.size == 0:
                    continue
                output_dir = os.path.join(faces_dir, str(identity_id))
                os.makedirs(output_dir, exist_ok=True)
                face_name = f"face_of_{identity_id}.jpg"
                cv2.imwrite(os.path.join(output_dir, face_name), crop)
                h, w = frame.shape[:2]
                x1, y1, x2, y2 = winner["box"]
                saved[identity_id] = (winner, os.path.join(str(identity_id), face_name),
                                      crop.shape[:2],
                                      [max(0, x1), max(0, y1), min(w, x2), min(h, y2)])

            if annotated_path:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(annotated_path, cv2.VideoWriter.fourcc(*"mp4v"),
                                             fps / base_step, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"No se pudo abrir el escritor de video para {annotated_path}")
                for identity_id, (x1, y1, x2, y2) in labels.get(frame_idx, []):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                    cv2.putText(frame, f"ID {identity_id}", (x1, max(12, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
                for _ in range(repeats.get(frame_idx, 1)):
                    writer.write(frame)
            elif frame_idx >= last_needed:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    return saved, writer is not None


def face_result(identity_id, faces, saved, fps):
    winner, face_file, (fh, fw), face_box = saved
    x1, y1, x2, y2 = face_box
    print(f"[{identity_id}] mejor rostro en el fotograma {winner['frame']} | "
          f"score {winner['score']:.4f} | rostro {fw}x{fh} | nitidez {winner['sharpness']:.1f} "
          f"({len(faces)} rostros de la identidad)")
    return {
        "track_id": str(identity_id),
        "face_file": face_file,
        "candidate": None,
        "score": winner["score"],
        "face_box": [x1, y1, x2, y2],
        "face_px": (x2 - x1) * (y2 - y1),
        "sharpness": winner["sharpness"],
        "candidates_checked": len(faces),
        "frame_idx": winner["frame"],
        "time_s": winner["frame"] / fps,
    }


def identity_record(identity_id, faces, fps):
    return {
        "id": identity_id,
        "faces": [{"frame": f["frame"], "time_s": f["frame"] / fps, "box": f["box"],
                   "score": f["score"], "px": f["px"], "interval_s": f["interval_s"]}
                  for f in faces],
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Identidades por rostro: detecta, alinea, "
                                                 "agrupa los rostros del video y guarda el "
                                                 "mejor de cada persona")
    parser.add_argument("--video", required=True, help="video de entrada")
    parser.add_argument("--faces-dir", default=FACES_DIR, help="carpeta de salida")
    parser.add_argument("--annotated-video", default=None,
                        help="si se indica, escribe ahí un video con las identidades")
    return parser.parse_args()


def sampling_summary(frames_per_mode, steps):
    if FIXED in steps:
        return f"uno cada {steps[FIXED]}"
    return ", ".join(f"{MODE_LABELS[mode]}: {count}" for mode, count in frames_per_mode.items())


def main():
    args = parse_args()
    cap = open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    try:
        detections, sampled, steps = analyse_video(cap, fps)
    finally:
        cap.release()

    identities = link_identities(detections, by_time=ADAPTIVE)
    os.makedirs(args.faces_dir, exist_ok=True)
    saved, annotated = render_outputs(args.video, [idx for idx, _ in sampled],
                                      min(steps.values()), fps, identities, args.faces_dir,
                                      args.annotated_video)

    results = []
    for identity_id, faces in enumerate(identities, start=1):
        if identity_id in saved:
            results.append(face_result(identity_id, faces, saved[identity_id], fps))

    frames_per_mode = {mode: sum(1 for _, m in sampled if m == mode) for mode in steps}
    wall_time_s = time.monotonic() - STARTED
    with open(os.path.join(args.faces_dir, RESULTS_NAME), "w") as f:
        json.dump({"tracks_checked": len(identities), "faces": results}, f, indent=2)
    with open(os.path.join(args.faces_dir, IDENTITIES_NAME), "w") as f:
        json.dump({
            "video": os.path.basename(args.video),
            "fps": fps,
            "step": steps.get(FIXED),
            "adaptive": ADAPTIVE,
            "steps": steps,
            "analysed_frames": len(sampled),
            "frames_per_mode": frames_per_mode,
            "wall_time_s": wall_time_s,
            "identities": [identity_record(k, faces, fps)
                           for k, faces in enumerate(identities, start=1)],
        }, f, indent=2)

    print(f"\nSe analizaron {len(sampled)} fotogramas "
          f"({sampling_summary(frames_per_mode, steps)}), se detectaron {len(detections)} rostros y se conservaron {len(identities)} "
          f"identidades; se guardó el mejor rostro de {len(results)} en {args.faces_dir}")
    duration = f" para {total_frames / fps:.1f} s de video" if total_frames > 0 else ""
    print(f"Tiempo total: {wall_time_s:.1f} s{duration}")
    if annotated:
        print(f"Video anotado escrito en {args.annotated_video}")


if __name__ == "__main__":
    main()
