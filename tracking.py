import argparse
import json
import os
import cv2
from ultralytics import YOLO  # type: ignore

# Defaults for manual runs. worker.py overrides them per job via CLI args.
VIDEO_PATH = "/data/FaceRecognition/Camera_ch5_20260824100831_20260824102210.mp4"
TRACKER = "custom_tracktrack.yaml"
CANDIDATES_DIR = os.path.join("database", "candidates_cfc_reid")
# Run detection + tracking only on 1 out of every DETECT_EVERY frames.
DETECT_EVERY = 30
MAX_CANDIDATES = 10
SAMPLE_EVERY_STEPS = 1
# Written next to the track folders so stage 2 can tell when each crop was taken.
MANIFEST_NAME = "manifest.json"

candidate_counts = {}   # track_id -> crops saved so far
last_saved_step = {}    # track_id -> processing step of last saved crop
candidate_frames = {}   # track_id -> {crop file name: frame index in the video}


def people_tracking(model, video_path, candidates_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
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

                out_dir = os.path.join(candidates_dir, str(track_id))
                os.makedirs(out_dir, exist_ok=True)
                n = candidate_counts.get(track_id, 0)
                fname = f"candidate_{n:02d}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), person_crop)
                candidate_counts[track_id] = n + 1
                last_saved_step[track_id] = step
                candidate_frames.setdefault(track_id, {})[fname] = frame_idx
    finally:
        cap.release()

    return fps


def write_manifest(candidates_dir, video_path, fps):
    os.makedirs(candidates_dir, exist_ok=True)
    manifest = {
        "video": os.path.basename(video_path),
        "fps": fps,
        "detect_every": DETECT_EVERY,
        "tracks": {
            str(track_id): frames
            for track_id, frames in sorted(candidate_frames.items())
        },
    }
    with open(os.path.join(candidates_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Stage 1: people tracking")
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--candidates-dir", default=CANDIDATES_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    # A manifest from an earlier run must not describe this run's crops.
    stale = os.path.join(args.candidates_dir, MANIFEST_NAME)
    if os.path.exists(stale):
        os.remove(stale)
    model = YOLO("yolo26m.pt")
    fps = people_tracking(model, args.video, args.candidates_dir)
    write_manifest(args.candidates_dir, args.video, fps)
    total = sum(candidate_counts.values())
    print(
        f"Saved {total} candidate crops for {len(candidate_counts)} people "
        f"in {args.candidates_dir}"
    )


if __name__ == "__main__":
    main()
