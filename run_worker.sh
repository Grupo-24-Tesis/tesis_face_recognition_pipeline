#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# Carga la configuración de AWS y del backend desde .env si existe (las variables están al inicio de worker.py).
if [ -f .env ]; then
  set -a; . ./.env; set +a
fi

exec venvs/worker/bin/python worker.py "$@"
