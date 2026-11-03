#!/usr/bin/env bash
set -e

if [ $# -lt 1 ]; then
  echo "uso: ./run.sh VIDEO [CARPETA_DE_SALIDA]" >&2
  exit 1
fi

REPO="$(cd "$(dirname "$0")" && pwd)"
VIDEO="$(realpath "$1")"
NAME="$(basename "${VIDEO%.*}")"
OUT="$(realpath -m "${2:-$REPO/database/run/$NAME}")"

cd "$REPO"
rm -rf "$OUT/candidates" "$OUT/faces"
mkdir -p "$OUT"

venvs/track/bin/python tracking.py --video "$VIDEO" --candidates-dir "$OUT/candidates" \
  --annotated-video "$OUT/tracked.mp4"
venvs/faces/bin/python face_recognition.py --candidates-dir "$OUT/candidates" \
  --faces-dir "$OUT/faces"
