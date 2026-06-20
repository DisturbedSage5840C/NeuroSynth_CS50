#!/usr/bin/env zsh
# NeuroSynth v4 — local bring-up.
# Flow: install deps -> generate realistic synthetic data -> train full stack
#       (enforcing the AUC >= 0.92 gate) -> launch the FastAPI backend.
#
#   ./run_local.sh             # train (if needed) then serve on :8888
#   ./run_local.sh --no-serve  # train only
#   ./run_local.sh --retrain   # force regenerate data + retrain
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

START_SERVER=1
FORCE_RETRAIN=0
for arg in "$@"; do
  case "$arg" in
    --no-serve) START_SERVER=0 ;;
    --retrain)  FORCE_RETRAIN=1 ;;
  esac
done

if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv/bin/python"
  PIP="$ROOT_DIR/.venv/bin/pip"
elif [[ -x "$ROOT_DIR/.venv312/bin/python" ]]; then
  PYTHON="$ROOT_DIR/.venv312/bin/python"
  PIP="$ROOT_DIR/.venv312/bin/pip"
else
  PYTHON="python3"
  PIP="pip3"
fi

export PYTHONPATH="$ROOT_DIR:$ROOT_DIR/src"
# Single-threaded native pools + duplicate-OpenMP allowance: avoids the macOS
# segfault when LightGBM and torch are trained back-to-back in one process.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE

DATA="$ROOT_DIR/data/realistic_v4.parquet"
MANIFEST="$ROOT_DIR/models/model_manifest.json"

echo "[1/4] Installing backend dependencies"
"$PIP" install -q -r backend/requirements.txt

echo "[2/4] Generating realistic synthetic dataset"
if [[ "$FORCE_RETRAIN" -eq 1 || ! -f "$DATA" ]]; then
  "$PYTHON" scripts/data/build_realistic_synthetic.py \
    --n 15000 --noise 0.5 --gain 2.5 --seed 42 --out "$DATA"
else
  echo "      $DATA exists — skipping (use --retrain to regenerate)"
fi

echo "[3/4] Training full model stack (AUC >= 0.92 gate)"
if [[ "$FORCE_RETRAIN" -eq 1 || ! -f "$MANIFEST" ]]; then
  "$PYTHON" train.py --validate
else
  echo "      $MANIFEST exists — skipping (use --retrain to retrain)"
fi

if [[ "$START_SERVER" -eq 1 ]]; then
  echo "[4/4] Starting FastAPI backend on http://localhost:8888"
  exec "$PYTHON" -m uvicorn backend.api:app --host 0.0.0.0 --port 8888
else
  echo "[4/4] Done. Start the API with:"
  echo "      PYTHONPATH=$ROOT_DIR:$ROOT_DIR/src $PYTHON -m uvicorn backend.api:app --host 0.0.0.0 --port 8888"
fi
