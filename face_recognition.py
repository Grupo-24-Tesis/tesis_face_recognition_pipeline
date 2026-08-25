import os

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2
from retinaface import RetinaFace

CANDIDATES_DIR = os.path.join("database", "candidates_deepocsort")
FACES_DIR = os.path.join("database", "faces_deepocsort")

# RetinaFace scores are floats in [0, 1], so 94% -> 0.94.
FACE_CONFIDENCE_THRESHOLD = 0.94


def best_face(pixel_array):
    """Return (score, facial_area) of the highest-scoring face, or (0.0, None)."""
    resp = RetinaFace.detect_faces(
        pixel_array,
        threshold=FACE_CONFIDENCE_THRESHOLD,
    )

    # detect_faces returns a dict {face_1: {...}} when faces are found,
    # and a non-dict (or empty) value otherwise, so guard for that.
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


def isFace(pixel_array, track_id) -> bool:
    score, _area = best_face(pixel_array)

    # Defensive: detect_faces already filters by threshold, but don't
    # rely on that in case the backend version ignores the argument.
    if score < FACE_CONFIDENCE_THRESHOLD:
        if score > 0.0:
            print(f"  rejected face for {track_id}: score {score:.4f} < "
                  f"{FACE_CONFIDENCE_THRESHOLD}")
        return False

    output_dir = os.path.join(FACES_DIR, str(track_id))
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"face_of_{track_id}.jpg")
    cv2.imwrite(save_path, pixel_array)
    print(f"Face confidence score: {score:.4f}")
    return True


def main():
    if not os.path.isdir(CANDIDATES_DIR):
        print(f"No candidates in {CANDIDATES_DIR} — run tracking.py first")
        return

    for track_id in sorted(os.listdir(CANDIDATES_DIR)):
        folder = os.path.join(CANDIDATES_DIR, track_id)
        if not os.path.isdir(folder):
            continue
        # Same behaviour as before: first crop that clears the confidence
        # threshold wins, then we stop checking this person.
        for fname in sorted(os.listdir(folder)):
            crop = cv2.imread(os.path.join(folder, fname))
            if crop is None:
                continue
            if isFace(crop, track_id):
                print(f"Face saved for track ID {track_id}")
                break
        else:
            print(f"No usable face for track ID {track_id}")


if __name__ == "__main__":
    main()