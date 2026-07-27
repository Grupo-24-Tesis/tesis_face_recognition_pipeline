import os
import cv2
from ultralytics import YOLO  # type: ignore

model = YOLO("yolo26m.pt")
VIDEO_PATH = "HD SDI CCTV 1080P Security Camera Front Door Entrance Demo [-IvBKBx0UBo].mkv"
TRACKER = "custom_tracktrack.yaml"

CANDIDATES_DIR = os.path.join("database", "candidates")

# The face check now lives in a separate process, so we can't stop
# cropping the moment a face is found. Instead we save a few sampled
# crops per track ID and let RetinaFace pick from them in stage 2.
MAX_CANDIDATES = 10
SAMPLE_EVERY = 15  # frames between saved crops of the same person

candidate_counts = {}   # track_id -> crops saved so far
last_saved_frame = {}   # track_id -> frame index of last saved crop


def people_tracking():
    results = model.track(
        source=VIDEO_PATH,
        # show=True,
        tracker=TRACKER,
        stream=True,
        classes=[0],  # only track the "person" class
    )

    for frame_idx, r in enumerate(results):
        frame = r.orig_img  # full frame as a BGR ndarray (H, W, 3)

        # Skip frames where nothing is tracked yet (ids can be None
        # on the first frames before the tracker assigns them)
        if r.boxes is None or r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy().astype(int)  # type: ignore # (N, 4) -> x1, y1, x2, y2
        track_ids = r.boxes.id.cpu().numpy().astype(int)  # type: ignore # (N,)

        h, w = frame.shape[:2]

        for (x1, y1, x2, y2), track_id in zip(boxes, track_ids):
            # Enough candidates for this person already
            if candidate_counts.get(track_id, 0) >= MAX_CANDIDATES:
                continue
            # Sample over time so candidates cover different poses
            if frame_idx - last_saved_frame.get(track_id, -SAMPLE_EVERY) < SAMPLE_EVERY:
                continue

            # Clamp coordinates so the crop never goes out of bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box, nothing to crop

            person_crop = frame[y1:y2, x1:x2]  # ndarray of just this person

            out_dir = os.path.join(CANDIDATES_DIR, str(track_id))
            os.makedirs(out_dir, exist_ok=True)
            n = candidate_counts.get(track_id, 0)
            cv2.imwrite(os.path.join(out_dir, f"candidate_{n:02d}.jpg"), person_crop)
            candidate_counts[track_id] = n + 1
            last_saved_frame[track_id] = frame_idx


def main():
    people_tracking()
    total = sum(candidate_counts.values())
    print(f"Saved {total} candidate crops for {len(candidate_counts)} people in {CANDIDATES_DIR}")


if __name__ == "__main__":
    main()
