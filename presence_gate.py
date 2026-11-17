import argparse
import json
import os
import sys
import time

from identity_linking import env_non_negative_float, env_positive_float, env_positive_int
from presence_intervals import active_intervals, format_time, switch_log

YOLO_WEIGHTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "yolo26m.pt")
PERSON_CLASS = 0
CHECKS_PER_S = env_positive_float("GATE_CHECKS_PER_S", 4.0)
REWIND_S = env_non_negative_float("GATE_REWIND_S", 1.0)
EXIT_S = env_non_negative_float("GATE_EXIT_S", 2.0)


def gate_conf():
    value = env_positive_float("GATE_CONF", 0.25)
    if value > 1:
        raise SystemExit(f"GATE_CONF debe estar entre 0 y 1, no {value!r}")
    return value


CONF = gate_conf()
IMGSZ = env_positive_int("GATE_IMGSZ", 640)


def settings():
    return {"checks_per_s": CHECKS_PER_S, "rewind_s": REWIND_S, "exit_s": EXIT_S,
            "conf": CONF, "imgsz": IMGSZ}


def check_step(fps):
    return max(1, round(fps / CHECKS_PER_S))


def read_checks(cap, step, person_confidence):
    checks = []
    confidences = []
    frame_idx = -1
    while cap.grab():
        frame_idx += 1
        if frame_idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            print(f"No se pudo leer el fotograma {frame_idx}: se cuenta como presencia y se "
                  f"sigue", file=sys.stderr)
            checks.append([frame_idx, True])
            confidences.append([frame_idx, 1.0])
            continue
        confidence = float(person_confidence(frame))
        checks.append([frame_idx, confidence > 0])
        confidences.append([frame_idx, confidence])
    return checks, confidences, frame_idx + 1


def build_document(video, fps, step, checks, frames_read, wall_s, confidences=None):
    if confidences is None:
        confidences = [[frame, 1.0 if found else 0.0] for frame, found in checks]
    total_frames = frames_read
    intervals = active_intervals(checks, fps, total_frames, REWIND_S, EXIT_S)
    active_frames = sum(end - start + 1 for start, end in intervals)
    return {
        "video": os.path.basename(video),
        "fps": fps,
        "total_frames": total_frames,
        "check_step": step,
        "settings": settings(),
        "checks": checks,
        "confidences": confidences,
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

    def person_confidence(frame):
        result = model.predict(source=frame, classes=[PERSON_CLASS], conf=CONF,
                               imgsz=IMGSZ, verbose=False)[0]
        if len(result.boxes) == 0:
            return 0.0
        return float(result.boxes.conf.max())
    return person_confidence


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
    step = check_step(fps)
    person_confidence = load_detector()
    try:
        checks, confidences, frames_read = read_checks(cap, step, person_confidence)
    finally:
        cap.release()
    document = build_document(args.video, fps, step, checks, frames_read,
                              time.monotonic() - started, confidences)
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
