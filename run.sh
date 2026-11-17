#!/usr/bin/env bash
set -e

GATING=1
POSITIONAL=()
for ARG in "$@"; do
  if [ "$ARG" = "--no-gating" ]; then
    GATING=0
  else
    POSITIONAL+=("$ARG")
  fi
done

if [ ${#POSITIONAL[@]} -lt 1 ]; then
  echo "uso: ./run.sh [--no-gating] VIDEO [CARPETA_DE_SALIDA]" >&2
  exit 1
fi

MODE="${PIPELINE_MODE:-faces}"
if [ "$MODE" != "faces" ] && [ "$MODE" != "tracks" ]; then
  echo "PIPELINE_MODE desconocido '$MODE': use faces o tracks" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")" && pwd)"
VIDEO="$(realpath "${POSITIONAL[0]}")"
NAME="$(basename "${VIDEO%.*}")"
OUT="$(realpath -m "${POSITIONAL[1]:-$REPO/database/run/$NAME}")"

cd "$REPO"

if [ "$MODE" = "faces" ]; then
  rm -rf "$OUT/faces"
  mkdir -p "$OUT"
  FACE_ARGS=(--video "$VIDEO" --faces-dir "$OUT/faces" --annotated-video "$OUT/identities.mp4")
  if [ "$GATING" = "1" ]; then
    venvs/track/bin/python presence_gate.py --video "$VIDEO" --out "$OUT/presence.json"
    FACE_ARGS+=(--presence "$OUT/presence.json")
  fi
  venvs/faces/bin/python face_identities.py "${FACE_ARGS[@]}"
  exit 0
fi

rm -rf "$OUT/candidates" "$OUT/faces"
mkdir -p "$OUT"

venvs/track/bin/python tracking.py --video "$VIDEO" --candidates-dir "$OUT/candidates" \
  --annotated-video "$OUT/tracked.mp4"
venvs/faces/bin/python face_recognition.py --candidates-dir "$OUT/candidates" \
  --faces-dir "$OUT/faces"
