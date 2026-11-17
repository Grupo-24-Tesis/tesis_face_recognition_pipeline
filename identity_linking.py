import math
import os

import numpy as np


def env_positive_int(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        number = int(value)
    except ValueError:
        number = 0
    if number < 1:
        raise SystemExit(f"{name} debe ser un entero positivo, no {value!r}")
    return number


def env_positive_float(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        number = float(value)
    except ValueError:
        number = 0.0
    if not math.isfinite(number) or number <= 0:
        raise SystemExit(f"{name} debe ser un número positivo, no {value!r}")
    return number


def env_non_negative_float(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        number = float(value)
    except ValueError:
        number = -1.0
    if not math.isfinite(number) or number < 0:
        raise SystemExit(f"{name} debe ser un número no negativo, no {value!r}")
    return number


def env_flag(name, default):
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    if value not in ("0", "1"):
        raise SystemExit(f"{name} debe ser 0 o 1, no {value!r}")
    return value == "1"


MIN_SCORE = 0.99
LINK_IOU = 0.3
LINK_SIMILARITY = 0.2
MAX_SKIP = 2
MERGE_THRESHOLD = 0.35
TWIN_THRESHOLD = 0.6
MIN_FACES = env_positive_int("FACE_MIN_FACES", 8)
MIN_PRESENCE_S = env_positive_float("FACE_MIN_PRESENCE_S", 0.8)
MIN_FACES_ABS = env_positive_int("FACE_MIN_FACES_ABS", 3)
PRESENCE_TOLERANCE_S = 1e-6


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union else 0.0


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


def build_tracklets(detections):
    by_frame = {}
    for d in detections:
        if d["score"] >= MIN_SCORE:
            by_frame.setdefault(d["frame"], []).append(d)
    tracklets, open_tracklets = [], []
    for frame in sorted(by_frame):
        current = by_frame[frame]
        sample = current[0]["sample"]
        live = [t for t in open_tracklets if sample - t[-1]["sample"] <= 1 + MAX_SKIP]
        pairs = []
        for ti, t in enumerate(live):
            for fi, f in enumerate(current):
                overlap = iou(t[-1]["box"], f["box"])
                similarity = float(np.dot(t[-1]["embedding"], f["embedding"]))
                if overlap >= LINK_IOU and similarity >= LINK_SIMILARITY:
                    pairs.append((overlap + similarity, ti, fi))
        pairs.sort(reverse=True)
        used_t, used_f = set(), set()
        for _, ti, fi in pairs:
            if ti in used_t or fi in used_f:
                continue
            live[ti].append(current[fi])
            used_t.add(ti)
            used_f.add(fi)
        for fi, f in enumerate(current):
            if fi not in used_f:
                t = [f]
                tracklets.append(t)
                live.append(t)
        open_tracklets = live
    return tracklets


def merge_tracklets(tracklets):
    if not tracklets:
        return []
    means = np.array([unit(np.sum([f["embedding"] for f in t], 0)) for t in tracklets])
    weights = np.array([len(t) for t in tracklets], dtype=np.float64)
    similarity = means @ means.T
    members = [[k] for k in range(len(tracklets))]
    frames = [{f["frame"] for f in t} for t in tracklets]
    alive = list(range(len(tracklets)))

    def link(a, b):
        block = similarity[np.ix_(members[a], members[b])]
        w = np.outer(weights[members[a]], weights[members[b]])
        return float((block * w).sum() / w.sum())

    sims = {}
    for x in range(len(alive)):
        for y in range(x + 1, len(alive)):
            a, b = alive[x], alive[y]
            if not (frames[a] & frames[b]):
                sims[(a, b)] = link(a, b)
    while sims:
        (a, b), s = max(sims.items(), key=lambda kv: kv[1])
        if s < MERGE_THRESHOLD:
            break
        members[a] += members[b]
        frames[a] |= frames[b]
        alive.remove(b)
        sims = {k: v for k, v in sims.items() if a not in k and b not in k}
        for c in alive:
            if c == a:
                continue
            key = (min(a, c), max(a, c))
            if not (frames[a] & frames[c]):
                sims[key] = link(key[0], key[1])
    groups = []
    for a in alive:
        faces = [f for k in members[a] for f in tracklets[k]]
        groups.append({"faces": faces, "frames": frames[a],
                       "sum": np.sum([f["embedding"] for f in faces], 0)})
    return groups


def quality(group):
    return (float(np.median([f["sharpness"] for f in group["faces"]]))
            * float(np.median([f["px"] for f in group["faces"]])))


def drop_mirror_twins(groups):
    if TWIN_THRESHOLD >= 1 or len(groups) < 2:
        return groups
    means = np.array([unit(g["sum"]) for g in groups])
    similarity = means @ means.T
    drop = set()
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if similarity[i, j] >= TWIN_THRESHOLD and groups[i]["frames"] & groups[j]["frames"]:
                drop.add(i if quality(groups[i]) < quality(groups[j]) else j)
    return [g for k, g in enumerate(groups) if k not in drop]


def presence_s(faces):
    return math.fsum(f["interval_s"] for f in faces)


def present_enough(faces, by_time):
    if not by_time:
        return len(faces) >= MIN_FACES
    return (len(faces) >= MIN_FACES_ABS
            and presence_s(faces) >= MIN_PRESENCE_S - PRESENCE_TOLERANCE_S)


def link_identities(detections, by_time=False):
    groups = merge_tracklets(build_tracklets(detections))
    kept = [g for g in groups if present_enough(g["faces"], by_time)]
    kept = drop_mirror_twins(kept)
    kept.sort(key=lambda g: min(f["frame"] for f in g["faces"]))
    return [sorted(g["faces"], key=lambda f: f["frame"]) for g in kept]
