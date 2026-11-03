#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# rm -rf database/candidates   # descomentar para empezar de cero en cada ejecución

venvs/track/bin/python tracking_results.py
venvs/faces/bin/python face_recognition.py
