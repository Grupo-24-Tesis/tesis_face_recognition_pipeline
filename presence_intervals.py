from bisect import bisect_left


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
    position = bisect_left(intervals, (frame_idx + 1,)) - 1
    return position >= 0 and intervals[position][0] <= frame_idx <= intervals[position][1]


def next_allowed(frame_idx, step, intervals):
    candidate = -(-max(0, frame_idx) // step) * step
    for start, end in intervals:
        if end < candidate:
            continue
        if candidate < start:
            candidate = -(-start // step) * step
        if candidate <= end:
            return candidate
    return None
