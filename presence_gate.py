import argparse
import json
import os
import time

from identity_linking import env_positive_float, env_positive_int
from presence_intervals import active_intervals, format_time, switch_log

YOLO_WEIGHTS = "yolo26m.pt"
PERSON_CLASS = 0
CHECKS_PER_S = env_positive_float("GATE_CHECKS_PER_S", 4.0)
REWIND_S = env_positive_float("GATE_REWIND_S", 1.0)
EXIT_S = env_positive_float("GATE_EXIT_S", 2.0)
CONF = env_positive_float("GATE_CONF", 0.25)
IMGSZ = env_positive_int("GATE_IMGSZ", 640)


def settings():
    return {"checks_per_s": CHECKS_PER_S, "rewind_s": REWIND_S, "exit_s": EXIT_S,
            "conf": CONF, "imgsz": IMGSZ}


def check_step(fps):
    return max(1, round(fps / CHECKS_PER_S))


def read_checks(cap, step, person_present):
    checks = []
    frame_idx = -1
    while cap.grab():
        frame_idx += 1
        if frame_idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        checks.append([frame_idx, bool(person_present(frame))])
    return checks, frame_idx + 1


def build_document(video, fps, declared_frames, step, checks, frames_read, wall_s):
    total_frames = declared_frames if declared_frames > 0 else frames_read
    intervals = active_intervals(checks, fps, total_frames, REWIND_S, EXIT_S)
    active_frames = sum(end - start + 1 for start, end in intervals)
    return {
        "video": os.path.basename(video),
        "fps": fps,
        "total_frames": total_frames,
        "check_step": step,
        "settings": settings(),
        "checks": checks,
        "intervals": [list(interval) for interval in intervals],
        "switches": switch_log(intervals, checks, fps, total_frames),
        "active_frames": active_frames,
        "active_fraction": active_frames / total_frames if total_frames > 0 else 0.0,
        "wall_s": wall_s,
    }


def describe_switches(switches):
    for item in switches:
        print(f"{item['evento']} desde {item['tiempo']} "
              f"(persona a las {item['disparo_tiempo']})")


def load_detector():
    from ultralytics import YOLO
    model = YOLO(YOLO_WEIGHTS)

    def person_present(frame):
        result = model.predict(source=frame, classes=[PERSON_CLASS], conf=CONF,
                               imgsz=IMGSZ, verbose=False)[0]
        return len(result.boxes) > 0
    return person_present


def parse_args():
    parser = argparse.ArgumentParser(description="Compuerta de presencia: revisa el video con YOLO "
                                                 "unas veces por segundo y escribe los tramos "
                                                 "donde hay alguien en cámara")
    parser.add_argument("--video", required=True, help="video de entrada")
    parser.add_argument("--out", required=True, help="archivo presence.json de salida")
    return parser.parse_args()


def main():
    started = time.monotonic()
    args = parse_args()
    import cv2
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"No se pudo abrir el video: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    declared_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = check_step(fps)
    person_present = load_detector()
    try:
        checks, frames_read = read_checks(cap, step, person_present)
    finally:
        cap.release()
    document = build_document(args.video, fps, declared_frames, step, checks, frames_read,
                              time.monotonic() - started)
    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(document, f)
    describe_switches(document["switches"])
    print(f"{len(checks)} revisiones, activo el {document['active_fraction']:.1%} del video "
          f"({format_time(document['active_frames'] / fps)} de "
          f"{format_time(document['total_frames'] / fps)}), {document['wall_s']:.1f} s")


if __name__ == "__main__":
    main()
