import os
import cv2
from ultralytics import YOLO  # type: ignore

model = YOLO("yolo26m.pt")

VIDEO_PATH = "/data/FaceRecognition/Camera_ch5_20260824100831_20260824102210.mp4"
TRACKER = "custom_tracktrack.yaml"
CANDIDATES_DIR = os.path.join("database", "candidates_cfc_reid")
# Run detection + tracking only on 1 out of every DETECT_EVERY frames.
DETECT_EVERY = 30
MAX_CANDIDATES = 10
SAMPLE_EVERY_STEPS = 1

candidate_counts = {}   # track_id -> crops saved so far
last_saved_step = {}    # track_id -> processing step of last saved crop


def people_tracking():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    frame_idx = -1   # index in the original video
    step = -1        # index of frames we actually ran the model on

    try:
        while True:
            # grab() advances the decoder without fully decoding the frame,
            # so skipped frames cost almost nothing.
            if not cap.grab():
                break
            frame_idx += 1

            if frame_idx % DETECT_EVERY != 0:
                continue

            ok, frame = cap.retrieve()
            if not ok:
                break
            step += 1

            results = model.track(
                source=frame,
                tracker=TRACKER,
                persist=True,     # keep tracker state across calls
                save=True,
                classes=[0],      # only track the "person" class
                verbose=False
            )
            r = results[0]

            # ids can be None before the tracker assigns them
            if r.boxes is None or r.boxes.id is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy().astype(int)      # type: ignore
            track_ids = r.boxes.id.cpu().numpy().astype(int)    # type: ignore

            h, w = frame.shape[:2]

            for (x1, y1, x2, y2), track_id in zip(boxes, track_ids):
                track_id = int(track_id)

                # Enough candidates for this person already
                if candidate_counts.get(track_id, 0) >= MAX_CANDIDATES:
                    continue
                # Sample over time so candidates cover different poses
                prev = last_saved_step.get(track_id, -SAMPLE_EVERY_STEPS)
                if step - prev < SAMPLE_EVERY_STEPS:
                    continue

                # Clamp coordinates so the crop never goes out of bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box, nothing to crop

                person_crop = frame[y1:y2, x1:x2]

                out_dir = os.path.join(CANDIDATES_DIR, str(track_id))
                os.makedirs(out_dir, exist_ok=True)
                n = candidate_counts.get(track_id, 0)
                cv2.imwrite(
                    os.path.join(out_dir, f"candidate_{n:02d}.jpg"),
                    person_crop,
                )
                candidate_counts[track_id] = n + 1
                last_saved_step[track_id] = step
    finally:
        cap.release()


def main():
    people_tracking()
    total = sum(candidate_counts.values())
    print(
        f"Saved {total} candidate crops for {len(candidate_counts)} people "
        f"in {CANDIDATES_DIR}"
    )


if __name__ == "__main__":
    main()