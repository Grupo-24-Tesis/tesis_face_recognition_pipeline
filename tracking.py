import argparse
import json
import os
import cv2
from ultralytics import YOLO

TRACKER = "custom_tracktrack.yaml"
CANDIDATES_DIR = os.path.join("database", "candidates_cfc_reid")
DETECT_INTERVAL_S = 0.5
MAX_CANDIDATES = 40
SCENE_CUT_CORRELATION = 0.5
YOLO_WEIGHTS = "yolo26m.pt"
YOLO_CONF = 0.1
YOLO_IOU = 0.7
YOLO_IMGSZ = 640
MANIFEST_NAME = "manifest.json"

candidate_counts = {}
last_saved_step = {}
sample_every = {}
next_index = {}
candidate_frames = {}
candidate_boxes = {}


def frame_histogram(frame):
    small = cv2.resize(frame, (160, 90))
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(hist, hist)


def people_tracking(model, video_path, candidates_dir, annotated_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detect_every = max(1, round(fps * DETECT_INTERVAL_S))
    frame_idx = -1
    step = -1
    prev_hist = None
    id_offset = 0
    max_id = 0
    cuts = 0
    writer = None

    try:
        while True:
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
                persist=True,
                classes=[0],
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                imgsz=YOLO_IMGSZ,
                verbose=False
            )
            r = results[0]

            if annotated_path:
                plotted = r.plot()
                if writer is None:
                    ah, aw = plotted.shape[:2]
                    writer = cv2.VideoWriter(annotated_path, cv2.VideoWriter.fourcc(*"mp4v"),
                                             max(fps / detect_every, 1.0), (aw, ah))
                    if not writer.isOpened():
                        raise RuntimeError(f"No se pudo abrir el escritor de video para {annotated_path}")
                writer.write(plotted)

            if r.boxes is None or r.boxes.id is None:
                continue

            boxes = r.boxes.xyxy.cpu().numpy().astype(int)
            track_ids = r.boxes.id.cpu().numpy().astype(int)

            h, w = frame.shape[:2]

            for (x1, y1, x2, y2), track_id in zip(boxes, track_ids):
                track_id = int(track_id) + id_offset
                max_id = max(max_id, track_id)

                every = sample_every.get(track_id, 1)
                prev = last_saved_step.get(track_id, -every)
                if step - prev < every:
                    continue

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                if x2 <= x1 or y2 <= y1:
                    continue

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
                candidate_boxes.setdefault(track_id, {})[fname] = [
                    int(x1), int(y1), int(x2), int(y2)]
                candidate_counts[track_id] = len(frames)

                if len(frames) >= MAX_CANDIDATES:
                    thin_candidates(track_id, out_dir)
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    if cuts:
        print(f"Se reinició el tracker en {cuts} cortes de escena")
    return fps, detect_every


def thin_candidates(track_id, out_dir):
    frames = candidate_frames[track_id]
    for i, fname in enumerate(sorted(frames)):
        if i % 2 == 1:
            os.remove(os.path.join(out_dir, fname))
            del frames[fname]
            candidate_boxes.get(track_id, {}).pop(fname, None)
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
        "boxes": {
            str(track_id): boxes
            for track_id, boxes in sorted(candidate_boxes.items())
        },
    }
    with open(os.path.join(candidates_dir, MANIFEST_NAME), "w") as f:
        json.dump(manifest, f, indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Etapa 1: seguimiento de personas")
    parser.add_argument("--video", required=True)
    parser.add_argument("--candidates-dir", default=CANDIDATES_DIR)
    parser.add_argument("--annotated-video", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    stale = os.path.join(args.candidates_dir, MANIFEST_NAME)
    if os.path.exists(stale):
        os.remove(stale)
    model = YOLO(YOLO_WEIGHTS)
    fps, detect_every = people_tracking(model, args.video, args.candidates_dir,
                                        args.annotated_video)
    write_manifest(args.candidates_dir, args.video, fps, detect_every)
    total = sum(candidate_counts.values())
    print(
        f"Se guardaron {total} recortes candidatos de {len(candidate_counts)} personas "
        f"en {args.candidates_dir}"
    )
    if args.annotated_video:
        print(f"Video anotado escrito en {args.annotated_video}")


if __name__ == "__main__":
    main()
