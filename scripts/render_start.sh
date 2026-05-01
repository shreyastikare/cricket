#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
DB_PATH="${DB_PATH:-/var/data/ipl.db}"
PORT="${PORT:-8050}"

cd "$ROOT_DIR"

shutdown() {
  local status=$?
  trap - EXIT INT TERM
  if [[ -n "${WORKER_PID:-}" ]]; then
    kill "$WORKER_PID" 2>/dev/null || true
  fi
  if [[ -n "${WEB_PID:-}" ]]; then
    kill "$WEB_PID" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  exit "$status"
}

trap shutdown EXIT INT TERM

DB_PATH="$DB_PATH" bash scripts/render_live_worker.sh &
WORKER_PID=$!

DB_PATH="$DB_PATH" gunicorn app.app:server \
  --bind "0.0.0.0:${PORT}" \
  --workers 1 \
  --threads 4 \
  --timeout 120 &
WEB_PID=$!

wait -n "$WORKER_PID" "$WEB_PID"
