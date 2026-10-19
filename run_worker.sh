#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Load AWS / backend settings from .env if present (variables: top of worker.py).
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

exec venvs/worker/bin/python worker.py "$@"
