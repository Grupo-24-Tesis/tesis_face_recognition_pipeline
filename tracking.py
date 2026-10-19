import argparse
import json
import os
import cv2
from ultralytics import YOLO  # type: ignore

# Defaults for manual runs. worker.py overrides them per job via CLI args.
VIDEO_PATH = "/data/FaceRecognition/Camera_ch5_20260824100831_20260824102210.mp4"
TRACKER = "custom_tracktrack.yaml"
CANDIDATES_DIR = os.path.join("database", "candidates_cfc_reid")
# Run detection + tracking every DETECT_INTERVAL_S seconds of video, whatever
# the frame rate (a fixed frame count meant 2 s between looks on 15 fps CCTV).
DETECT_INTERVAL_S = 0.5
# Crops kept per person. When a track reaches the cap, every other crop is
# dropped and its sampling interval doubles, so the crops always span the
# whole time the person is on screen instead of only their first seconds.
MAX_CANDIDATES = 10
# Consecutive analysed frames whose colour histograms correlate below this are
# a scene cut (edited or phone videos). The tracker is reset there so people
# from different scenes are never merged into one track. Fixed CCTV cameras
# stay far above it.
SCENE_CUT_CORRELATION = 0.5
# Written next to the track folders so stage 2 can tell when each crop was taken.
MANIFEST_NAME = "manifest.json"

candidate_counts = {}   # track_id -> crops currently kept
last_saved_step = {}    # track_id -> processing step of last saved crop
sample_every = {}       # track_id -> steps between saved crops (doubles on thinning)
next_index = {}         # track_id -> number for the next crop file name
candidate_frames = {}   # track_id -> {crop file name: frame index in the video}


def frame_histogram(frame):
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist)


def people_tracking(model, video_path, candidates_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detect_every = max(1, round(fps * DETECT_INTERVAL_S))
    frame_idx = -1   # index in the original video
    step = -1        # index of frames we actually ran the model on
    prev_hist = None
    id_offset = 0    # keeps track IDs unique after a reset restarts them at 1
    max_id = 0
    cuts = 0

    try:
        while True:
            # grab() advances the decoder without fully decoding the frame,
            # so skipped frames cost almost nothing.
            if not cap.grab():
                break
            frame_idx += 1

            if frame_idx % detect_every != 0:
                continue

            ok, frame = cap.retrieve()
            if not ok:
                break
            step += 1

            hist = frame_histogram(frame)
            if (prev_hist is not None and
                    cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                    < SCENE_CUT_CORRELATION):
                trackers = getattr(model.predictor, "trackers", None)
                if trackers:
                    trackers[0].reset()
                    id_offset = max_id
                    cuts += 1
            prev_hist = hist

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
                track_id = int(track_id) + id_offset
                max_id = max(max_id, track_id)

                # Sample over time so candidates cover different poses
                every = sample_every.get(track_id, 1)
                prev = last_saved_step.get(track_id, -every)
                if step - prev < every:
                    continue

                # Clamp coordinates so the crop never goes out of bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue  # degenerate box, nothing to crop

                person_crop = frame[y1:y2, x1:x2]

                out_dir = os.path.join(candidates_dir, str(track_id))
                os.makedirs(out_dir, exist_ok=True)
                n = next_index.get(track_id, 0)
                fname = f"candidate_{n:03d}.jpg"
                cv2.imwrite(os.path.join(out_dir, fname), person_crop)
                next_index[track_id] = n + 1
                last_saved_step[track_id] = step
                frames = candidate_frames.setdefault(track_id, {})
                frames[fname] = frame_idx
                candidate_counts[track_id] = len(frames)

                if len(frames) >= MAX_CANDIDATES:
                    thin_candidates(track_id, out_dir)
    finally:
        cap.release()

    if cuts:
        print(f"Reset the tracker at {cuts} scene cuts")
    return fps, detect_every


def thin_candidates(track_id, out_dir):
    """Drop every other crop of a full track and halve its sampling rate."""
    frames = candidate_frames[track_id]
    for i, fname in enumerate(sorted(frames)):
        if i % 2 == 1:
            os.remove(os.path.join(out_dir, fname))
            del frames[fname]
    candidate_counts[track_id] = len(frames)
    sample_every[track_id] = sample_every.get(track_id, 1) * 2


def write_manifest(candidates_dir, video_path, fps, detect_every):
    os.makedirs(candidates_dir, exist_ok=True)
    manifest = {
        "video": os.path.basename(video_path),
        "fps": fps,
        "detect_every": detect_every,
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
    fps, detect_every = people_tracking(model, args.video, args.candidates_dir)
    write_manifest(args.candidates_dir, args.video, fps, detect_every)
    total = sum(candidate_counts.values())
    print(
        f"Saved {total} candidate crops for {len(candidate_counts)} people "
        f"in {args.candidates_dir}"
    )


if __name__ == "__main__":
    main()
