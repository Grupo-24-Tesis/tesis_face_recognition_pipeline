import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2
from retinaface import RetinaFace

CANDIDATES_DIR = os.path.join("database", "candidates_cfc_reid")
FACES_DIR = os.path.join("database", "faces_cfc_reid")
FACE_CONFIDENCE_THRESHOLD = 0.92
FACE_MARGIN = 0.25
RANKING = "quality"


def sharpness(image) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def crop_face(pixel_array, facial_area):
    x1, y1, x2, y2 = facial_area
    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * FACE_MARGIN), int(bh * FACE_MARGIN)

    h, w = pixel_array.shape[:2]
    x1, y1 = max(0, x1 - mx), max(0, y1 - my)
    x2, y2 = min(w, x2 + mx), min(h, y2 + my)

    if x2 <= x1 or y2 <= y1:
        return None
    return pixel_array[y1:y2, x1:x2]


def best_face(pixel_array):
    resp = RetinaFace.detect_faces(
        pixel_array,
        threshold=FACE_CONFIDENCE_THRESHOLD,
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

    return best_score, best_area


def evaluate_candidate(crop):
    score, area = best_face(crop)
    if score < FACE_CONFIDENCE_THRESHOLD or area is None:
        return None

    face = crop_face(crop, area)
    if face is None or face.size == 0:
        return None

    x1, y1, x2, y2 = area
    return {
        "score": score,
        "face_px": (x2 - x1) * (y2 - y1),
        "sharpness": sharpness(face),
        "face": face,
    }


def rank_key(cand):
    if RANKING == "score":
        return (cand["score"], cand["face_px"])
    return (cand["face_px"] * (cand["sharpness"] ** 0.5), cand["score"])


def process_track(track_id, folder) -> bool:
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
        print(f"[{track_id}] no face above {FACE_CONFIDENCE_THRESHOLD} "
              f"in {checked} candidates")
        return False

    output_dir = os.path.join(FACES_DIR, str(track_id))
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, f"face_of_{track_id}.jpg"), best["face"])

    fh, fw = best["face"].shape[:2]
    print(f"[{track_id}] best = {best_name} | score {best['score']:.4f} | "
          f"face {fw}x{fh} | sharpness {best['sharpness']:.1f} "
          f"({checked} candidates checked)")
    return True


def main():
    if not os.path.isdir(CANDIDATES_DIR):
        print(f"No candidates in {CANDIDATES_DIR} — run tracking.py first")
        return

    found = 0
    total = 0
    for track_id in sorted(os.listdir(CANDIDATES_DIR)):
        folder = os.path.join(CANDIDATES_DIR, track_id)
        if not os.path.isdir(folder):
            continue
        total += 1
        if process_track(track_id, folder):
            found += 1

    print(f"\nSaved a face for {found} of {total} tracked people in {FACES_DIR}")


if __name__ == "__main__":
    main()