#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# rm -rf database/candidates   # uncomment for a clean slate each run

venvs/track/bin/python tracking_results.py
venvs/faces/bin/python face_recognition.py
