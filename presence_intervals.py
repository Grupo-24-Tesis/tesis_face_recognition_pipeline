import json
from bisect import bisect_left, bisect_right

ADAPTIVE_WITH_GATING = ("La compuerta de presencia no se puede usar con FACE_ADAPTIVE=1: "
                        "use FACE_ADAPTIVE=0 o --no-gating")


class PresenceFileError(ValueError):
    pass


def format_time(seconds):
    tenths = int(seconds * 10 + 1e-6)
    whole, tenth = divmod(tenths, 10)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}.{tenth}"


def active_intervals(checks, fps, total_frames, rewind_s, exit_s):
    rewind = round(rewind_s * fps)
    after = round(exit_s * fps)
    last_frame = total_frames - 1 if total_frames > 0 else None
    intervals = []
    for frame, found in checks:
        if not found:
            continue
        start = max(0, frame - rewind)
        end = frame + after
        if last_frame is not None:
            end = min(end, last_frame)
            if start > last_frame:
                continue
        if intervals and intervals[-1][1] + 1 >= start:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
    return intervals


def switch_log(intervals, checks, fps, total_frames=0):
    found = [frame for frame, detected in checks if detected]
    events = []
    for start, end in intervals:
        inside = [frame for frame in found if start <= frame <= end]
        first, last = (inside[0], inside[-1]) if inside else (start, end)
        events.append(event("activo", start, first, fps))
        if total_frames <= 0 or end + 1 < total_frames:
            events.append(event("inactivo", end + 1, last, fps))
    return events


def event(kind, frame, trigger, fps):
    return {"evento": kind, "frame": frame, "tiempo": format_time(frame / fps),
            "disparo_frame": trigger, "disparo_tiempo": format_time(trigger / fps)}


def frame_in_intervals(frame_idx, intervals):
    position = bisect_right(intervals, frame_idx, key=lambda interval: interval[0]) - 1
    return position >= 0 and intervals[position][0] <= frame_idx <= intervals[position][1]


def next_allowed(frame_idx, step, intervals):
    candidate = -(-max(0, frame_idx) // step) * step
    position = bisect_left(intervals, candidate, key=lambda interval: interval[1])
    while position < len(intervals):
        start, end = intervals[position]
        if candidate < start:
            candidate = -(-start // step) * step
        if candidate <= end:
            return candidate
        position += 1
    return None


def gating_refusal(gating, adaptive):
    return ADAPTIVE_WITH_GATING if gating and adaptive else None


def fps_mismatch(path, presence_fps, fps):
    if abs(presence_fps - fps) > 1e-3:
        return (f"El archivo de presencia {path} es de un video con {presence_fps} "
                f"fotogramas por segundo y este tiene {fps}")
    return None


def load_presence(path, fps):
    try:
        with open(path) as f:
            presence = json.load(f)
        intervals = [(int(start), int(end)) for start, end in presence["intervals"]]
        active_fraction = float(presence["active_fraction"])
        total_frames = int(presence["total_frames"])
        presence_fps = float(presence["fps"])
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PresenceFileError(f"No se pudo leer el archivo de presencia {path}: {exc}")
    mismatch = fps_mismatch(path, presence_fps, fps)
    if mismatch:
        raise PresenceFileError(mismatch)
    return {"intervals": intervals, "active_fraction": active_fraction,
            "total_frames": total_frames}
