#!/usr/bin/env bash
#
# FastVLM API startup script
# Loads the FastVLM model once and serves HTTP requests on port 8000.
# Logs are timestamped and streamed to stdout.
#

# --- Configuration ---
APP="fastvlm_api:app"
HOST="0.0.0.0"
PORT="8000"
LOG_LEVEL="info"

# Resolve absolute venv path relative to this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="${SCRIPT_DIR}/../.venv310_fastvlm"

# --- Helpers ---
timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

echo "[$(timestamp)] 🔧 Activating virtual environment..."
if [ ! -f "${VENV_PATH}/bin/activate" ]; then
    echo "[$(timestamp)] ❌ Virtual environment not found at ${VENV_PATH}"
    exit 1
fi
# shellcheck disable=SC1090
source "${VENV_PATH}/bin/activate"

echo "[$(timestamp)] 🚀 Starting FastVLM API on http://${HOST}:${PORT}"
echo "[$(timestamp)] Logs will stream below; press Ctrl+C to stop."

# --- Run the service ---
exec uvicorn "${APP}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --log-level "${LOG_LEVEL}" \
    --reload \
    2>&1 | while IFS= read -r line; do
        echo "[$(timestamp)] $line"
    done

