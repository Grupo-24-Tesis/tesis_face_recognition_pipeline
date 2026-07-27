import os
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import cv2
from retinaface import RetinaFace

CANDIDATES_DIR = os.path.join("database", "candidates")


# We could add more validations to trust that
# the array of pixels has a face in it, such
# as adding a confidence threshold we need to
# meet to be able to tell that the face is usable.
def isFace(pixel_array, track_id) -> bool:
    resp = RetinaFace.detect_faces(pixel_array)

    # detect_faces returns a dict {face_1: {...}} when faces are found,
    # and a non-dict (or empty) value otherwise, so guard for that.
    if isinstance(resp, dict) and len(resp.keys()) > 0:
        output_dir = os.path.join("database", "faces", str(track_id))
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f"face_of_{track_id}.jpg")
        cv2.imwrite(save_path, pixel_array)
        return True
    return False


def main():
    if not os.path.isdir(CANDIDATES_DIR):
        print(f"No candidates in {CANDIDATES_DIR} — run tracking.py first")
        return

    for track_id in sorted(os.listdir(CANDIDATES_DIR)):
        folder = os.path.join(CANDIDATES_DIR, track_id)
        if not os.path.isdir(folder):
            continue
        # Same behaviour as before: first crop where RetinaFace finds a
        # face wins, then we stop checking this person.
        for fname in sorted(os.listdir(folder)):
            crop = cv2.imread(os.path.join(folder, fname))
            if crop is None:
                continue
            if isFace(crop, track_id):
                print(f"Face saved for track ID {track_id}")
                break


if __name__ == "__main__":
    main()
