import os

import cv2
from ultralytics import YOLO  # type: ignore

model = YOLO("yolo26m.pt")

VIDEO_PATH = "/data/FaceRecognition/Camera_ch5_20260824100831_20260824102210.mp4"
TRACKER = "custom_tracktrack.yaml"
CANDIDATES_DIR = os.path.join("database", "candidates_cfc_reid")
DETECT_EVERY = 30
MAX_CANDIDATES = 10
SAMPLE_EVERY_STEPS = 1

SAVE_VIDEO = True
OUTPUT_VIDEO = "tracked.mp4"
FOURCC = "mp4v"

candidate_counts = {}
last_saved_step = {}


def people_tracking():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    writer = None
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = max(src_fps / DETECT_EVERY, 1.0)

    frame_idx = -1
    step = -1

    try:
        while True:
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
                persist=True,
                classes=[0],
                verbose=False,
            )
            r = results[0]

            annotated = r.plot() if SAVE_VIDEO else None

            if annotated is not None:
                if writer is None:
                    ah, aw = annotated.shape[:2]
                    writer = cv2.VideoWriter(
                        OUTPUT_VIDEO,
                        cv2.VideoWriter.fourcc(*FOURCC),
                        out_fps,
                        (aw, ah),
                    )
                    if not writer.isOpened():
                        raise RuntimeError(
                            f"Could not open video writer for {OUTPUT_VIDEO} "
                            f"({FOURCC}, {aw}x{ah} @ {out_fps:.2f} fps)"
                        )
                    print(f"Writing {OUTPUT_VIDEO}: {aw}x{ah} @ {out_fps:.2f} fps")
                writer.write(annotated)

            if r.boxes is None or r.boxes.id is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy().astype(int)      # type: ignore
            track_ids = r.boxes.id.cpu().numpy().astype(int)    # type: ignore

            h, w = frame.shape[:2]

            for (x1, y1, x2, y2), track_id in zip(boxes, track_ids):
                track_id = int(track_id)

                if candidate_counts.get(track_id, 0) >= MAX_CANDIDATES:
                    continue

                prev = last_saved_step.get(track_id, -SAMPLE_EVERY_STEPS)
                if step - prev < SAMPLE_EVERY_STEPS:
                    continue

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

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
        if writer is not None:
            writer.release()


def main():
    people_tracking()
    total = sum(candidate_counts.values())
    print(
        f"Saved {total} candidate crops for {len(candidate_counts)} people "
        f"in {CANDIDATES_DIR}"
    )
    if SAVE_VIDEO:
        print(f"Annotated video written to {OUTPUT_VIDEO}")


if __name__ == "__main__":
    main()