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
from retinaface import RetinaFace

CANDIDATES_DIR = os.path.join("database", "candidates_cfc_reid")
FACES_DIR = os.path.join("database", "faces_cfc_reid")
FACE_CONFIDENCE_THRESHOLD = 0.92
FACE_MARGIN = 0.25
OUTPUT_MARGIN_SIDE = 0.8
OUTPUT_MARGIN_TOP = 0.6
OUTPUT_MARGIN_BOTTOM = 1.8
DETECT_PADDING = 0.8
RETINA_ALLOW_UPSCALING = True
MIN_FACE_PX = 12
RANKING = "quality"
MANIFEST_NAME = "manifest.json"
RESULTS_NAME = "results.json"


def sharpness(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_face(pixel_array, facial_area, side, top, bottom):
    x1, y1, x2, y2 = facial_area
    bw, bh = x2 - x1, y2 - y1
    mx = int(bw * side)

    h, w = pixel_array.shape[:2]
    x1, y1 = max(0, x1 - mx), max(0, y1 - int(bh * top))
    x2, y2 = min(w, x2 + mx), min(h, y2 + int(bh * bottom))

    if x2 <= x1 or y2 <= y1:
        return None
    return pixel_array[y1:y2, x1:x2]


def best_face(pixel_array):
    h, w = pixel_array.shape[:2]
    pad = int(DETECT_PADDING * max(h, w))
    padded = cv2.copyMakeBorder(
        pixel_array, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    resp = RetinaFace.detect_faces(
        padded,
        threshold=FACE_CONFIDENCE_THRESHOLD,
        allow_upscaling=RETINA_ALLOW_UPSCALING,
    )
    if not isinstance(resp, dict) or not resp:
        return 0.0, None

    best_score = 0.0
    best_area = None
    for face in resp.values():
        score = float(face.get("score", 0.0))
        if score > best_score:
            best_score = score
            best_area = face.get("facial_area")

    if best_area is None:
        return best_score, None
    x1, y1, x2, y2 = (int(v) - pad for v in best_area)
    return best_score, (max(0, x1), max(0, y1), min(w, x2), min(h, y2))


def evaluate_candidate(crop):
    score, area = best_face(crop)
    if score < FACE_CONFIDENCE_THRESHOLD or area is None:
        return None

    x1, y1, x2, y2 = area
    if min(x2 - x1, y2 - y1) < MIN_FACE_PX:
        return None

    face = crop_face(crop, area, FACE_MARGIN, FACE_MARGIN, FACE_MARGIN)
    if face is None or face.size == 0:
        return None

    return {
        "area": area,
        "score": score,
        "face_px": (x2 - x1) * (y2 - y1),
        "sharpness": sharpness(face),
        "face": crop_face(crop, area, OUTPUT_MARGIN_SIDE,
                          OUTPUT_MARGIN_TOP, OUTPUT_MARGIN_BOTTOM),
    }


def rank_key(cand):
    if RANKING == "score":
        return (cand["score"], cand["face_px"])
    return (cand["face_px"] * (cand["sharpness"] ** 0.5), cand["score"])


def process_track(track_id, folder, faces_dir):
    best = None
    best_name = None
    checked = 0

    for fname in sorted(os.listdir(folder)):
        crop = cv2.imread(os.path.join(folder, fname))
        if crop is None:
            continue
        checked += 1

        cand = evaluate_candidate(crop)
        if cand is None:
            continue

        if best is None or rank_key(cand) > rank_key(best):
            best = cand
            best_name = fname

    if best is None:
        print(f"[{track_id}] ningún rostro supera {FACE_CONFIDENCE_THRESHOLD} "
              f"en {checked} candidatos")
        return None

    output_dir = os.path.join(faces_dir, str(track_id))
    os.makedirs(output_dir, exist_ok=True)
    face_name = f"face_of_{track_id}.jpg"
    cv2.imwrite(os.path.join(output_dir, face_name), best["face"])

    fh, fw = best["face"].shape[:2]
    print(f"[{track_id}] mejor = {best_name} | score {best['score']:.4f} | "
          f"rostro {fw}x{fh} | nitidez {best['sharpness']:.1f} "
          f"({checked} candidatos revisados)")
    return {
        "track_id": track_id,
        "face_file": os.path.join(str(track_id), face_name),
        "candidate": best_name,
        "score": best["score"],
        "face_box": list(best["area"]),
        "face_px": best["face_px"],
        "sharpness": best["sharpness"],
        "candidates_checked": checked,
    }


def load_manifest(candidates_dir):
    path = os.path.join(candidates_dir, MANIFEST_NAME)
    if not os.path.isfile(path):
        return None
    written = os.path.getmtime(path)
    for track_id in os.listdir(candidates_dir):
        folder = os.path.join(candidates_dir, track_id)
        if not os.path.isdir(folder):
            continue
        for fname in os.listdir(folder):
            if os.path.getmtime(os.path.join(folder, fname)) > written:
                print(f"Se ignora {MANIFEST_NAME} desactualizado: hay recortes más nuevos; "
                      f"los tiempos del video quedarán vacíos")
                return None
    with open(path) as f:
        return json.load(f)


def add_video_time(face, manifest):
    face["frame_idx"] = None
    face["time_s"] = None
    if manifest is None:
        return
    frames = manifest.get("tracks", {}).get(face["track_id"], {})
    frame_idx = frames.get(face["candidate"])
    if frame_idx is None:
        return
    face["frame_idx"] = frame_idx
    face["time_s"] = frame_idx / manifest["fps"]


def parse_args():
    parser = argparse.ArgumentParser(description="Etapa 2: mejor rostro por track")
    parser.add_argument("--candidates-dir", default=CANDIDATES_DIR)
    parser.add_argument("--faces-dir", default=FACES_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    if not os.path.isdir(args.candidates_dir):
        print(f"No hay candidatos en {args.candidates_dir}; primero ejecute tracking.py")
        return

    manifest = load_manifest(args.candidates_dir)
    faces = []
    total = 0
    for track_id in sorted(os.listdir(args.candidates_dir)):
        folder = os.path.join(args.candidates_dir, track_id)
        if not os.path.isdir(folder):
            continue
        total += 1
        face = process_track(track_id, folder, args.faces_dir)
        if face is not None:
            add_video_time(face, manifest)
            faces.append(face)

    os.makedirs(args.faces_dir, exist_ok=True)
    with open(os.path.join(args.faces_dir, RESULTS_NAME), "w") as f:
        json.dump({"tracks_checked": total, "faces": faces}, f, indent=2)

    print(f"\nSe guardó un rostro para {len(faces)} de {total} personas seguidas "
          f"en {args.faces_dir}")


if __name__ == "__main__":
    main()