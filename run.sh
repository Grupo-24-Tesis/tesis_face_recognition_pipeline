#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "uso: ./run.sh VIDEO [CARPETA_DE_SALIDA]" >&2
  exit 1
fi

MODE="${PIPELINE_MODE:-faces}"
if [ "$MODE" != "faces" ] && [ "$MODE" != "tracks" ]; then
  echo "PIPELINE_MODE desconocido '$MODE': use faces o tracks" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")" && pwd)"
VIDEO="$(realpath "$1")"
NAME="$(basename "${VIDEO%.*}")"
OUT="$(realpath -m "${2:-$REPO/database/run/$NAME}")"

cd "$REPO"

if [ "$MODE" = "faces" ]; then
  rm -rf "$OUT/faces"
  mkdir -p "$OUT"
  venvs/faces/bin/python face_identities.py --video "$VIDEO" --faces-dir "$OUT/faces" \
    --annotated-video "$OUT/identities.mp4"
  exit 0
fi

rm -rf "$OUT/candidates" "$OUT/faces"
mkdir -p "$OUT"

venvs/track/bin/python tracking.py --video "$VIDEO" --candidates-dir "$OUT/candidates" \
  --annotated-video "$OUT/tracked.mp4"
venvs/faces/bin/python face_recognition.py --candidates-dir "$OUT/candidates" \
  --faces-dir "$OUT/faces"
