import math
import os


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
