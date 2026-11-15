import argparse
import ctypes
import glob
import json
import os
import sysconfig

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
from identity_linking import env_positive_float, link_identities

FACES_DIR = os.path.join("database", "faces_identidades")
FACE_INTERVAL_S = env_positive_float("FACE_INTERVAL_S", 0.1)
RETINA_THRESHOLD = 0.9
RETINA_ALLOW_UPSCALING = True
EMBEDDING_MODEL = "GhostFaceNet"
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


def analysed_frames(cap, step):
    frame_idx = -1
    while True:
        if not cap.grab():
            return
        frame_idx += 1
        if frame_idx % step != 0:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            return
        yield frame_idx, frame


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
    emb = np.array(rep[0]["embedding"], dtype=np.float32)
    return emb / (np.linalg.norm(emb) + 1e-9)


def detect_faces(frame_idx, frame):
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
            "box": [x1, y1, x2, y2],
            "score": float(face["score"]),
            "px": min(x2 - x1, y2 - y1),
            "sharpness": sharpness(box_crop),
            "embedding": embedding(aligned),
        })
    return faces


def ranking(face):
    x1, y1, x2, y2 = face["box"]
    return rank_key({"face_px": (x2 - x1) * (y2 - y1),
                     "sharpness": face["sharpness"], "score": face["score"]})


def render_outputs(video_path, step, fps, identities, faces_dir, annotated_path):
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
    writer = None
    cap = open_video(video_path)
    try:
        for frame_idx, frame in analysed_frames(cap, step):
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
                                             fps / step, (w, h))
                    if not writer.isOpened():
                        raise RuntimeError(f"No se pudo abrir el escritor de video para {annotated_path}")
                for identity_id, (x1, y1, x2, y2) in labels.get(frame_idx, []):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
                    cv2.putText(frame, f"ID {identity_id}", (x1, max(12, y1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLOR, 1, cv2.LINE_AA)
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
                   "score": f["score"], "px": f["px"]} for f in faces],
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


def main():
    args = parse_args()
    cap = open_video(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, round(fps * FACE_INTERVAL_S))
    detections = []
    analysed = 0
    try:
        for frame_idx, frame in analysed_frames(cap, step):
            analysed += 1
            detections.extend(detect_faces(frame_idx, frame))
    finally:
        cap.release()

    identities = link_identities(detections, step)
    os.makedirs(args.faces_dir, exist_ok=True)
    saved, annotated = render_outputs(args.video, step, fps, identities, args.faces_dir,
                           args.annotated_video)

    results = []
    for identity_id, faces in enumerate(identities, start=1):
        if identity_id in saved:
            results.append(face_result(identity_id, faces, saved[identity_id], fps))

    with open(os.path.join(args.faces_dir, RESULTS_NAME), "w") as f:
        json.dump({"tracks_checked": len(identities), "faces": results}, f, indent=2)
    with open(os.path.join(args.faces_dir, IDENTITIES_NAME), "w") as f:
        json.dump({
            "video": os.path.basename(args.video),
            "fps": fps,
            "step": step,
            "analysed_frames": analysed,
            "identities": [identity_record(k, faces, fps)
                           for k, faces in enumerate(identities, start=1)],
        }, f, indent=2)

    print(f"\nSe analizaron {analysed} fotogramas (uno cada {step}), se detectaron "
          f"{len(detections)} rostros y se conservaron {len(identities)} identidades; "
          f"se guardó el mejor rostro de {len(results)} en {args.faces_dir}")
    if annotated:
        print(f"Video anotado escrito en {args.annotated_video}")


if __name__ == "__main__":
    main()
