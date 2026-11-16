from identity_linking import LINK_IOU, env_flag, env_positive_float, env_positive_int, iou

FIXED = "fixed"
DENSE = "dense"
SPARSE = "sparse"
IDLE = "idle"
MODE_LABELS = {FIXED: "fijo", DENSE: "denso", SPARSE: "disperso", IDLE: "inactivo"}

ADAPTIVE = env_flag("FACE_ADAPTIVE", True)
INTERVAL_S = env_positive_float("FACE_INTERVAL_S", 0.1)
INTERVAL_DENSE_S = env_positive_float("FACE_INTERVAL_DENSE_S", 0.1)
INTERVAL_SPARSE_S = env_positive_float("FACE_INTERVAL_SPARSE_S", 0.3)
INTERVAL_IDLE_S = env_positive_float("FACE_INTERVAL_IDLE_S", 1.0)
CROWD_FACES = env_positive_int("FACE_CROWD_FACES", 3)
IDLE_FRAMES = env_positive_int("FACE_IDLE_FRAMES", 3)
DENSE_HOLD_S = env_positive_float("FACE_DENSE_HOLD_S", 1.0)


def frame_step(fps, interval_s):
    return max(1, round(fps * interval_s))


def sampling_steps(fps, adaptive):
    if not adaptive:
        return {FIXED: frame_step(fps, INTERVAL_S)}
    return {DENSE: frame_step(fps, INTERVAL_DENSE_S),
            SPARSE: frame_step(fps, INTERVAL_SPARSE_S),
            IDLE: frame_step(fps, INTERVAL_IDLE_S)}


def hold_frames(fps):
    return round(fps * DENSE_HOLD_S)


def start_state(adaptive):
    return {"mode": DENSE if adaptive else FIXED, "boxes": [], "empty": 0, "hold_until": -1}


def continues(box, previous_boxes):
    return any(iou(box, previous) >= LINK_IOU for previous in previous_boxes)


def advance(state, boxes, frame_idx, hold):
    if state["mode"] == FIXED:
        return state
    empty = 0 if boxes else state["empty"] + 1
    hold_until = state["hold_until"]
    arrival = any(not continues(box, state["boxes"]) for box in boxes)
    if len(boxes) >= CROWD_FACES or arrival:
        mode = DENSE
        hold_until = frame_idx + hold
    elif frame_idx < hold_until:
        mode = DENSE
    elif boxes:
        mode = SPARSE
    elif empty >= IDLE_FRAMES:
        mode = IDLE
    else:
        mode = state["mode"]
    return {"mode": mode, "boxes": [list(box) for box in boxes], "empty": empty,
            "hold_until": hold_until}
