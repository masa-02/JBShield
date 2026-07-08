#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-qwen3-8b}"
RUN_ID="${2:-jbshield-qwen-phase2-qwen3-8b}"
LOG="${3:-logs/JBShield-D_qwen_phase2-qwen3-8b.log}"

echo "===== log error summary ====="
if [[ -f "$LOG" ]]; then
  grep -nE "===== configs/runtime/|Traceback|OutOfMemory|CUDA out of memory|Skip |success|failed|Phase2" "$LOG" | tail -80 || true
else
  echo "missing log: $LOG"
fi

echo
echo "===== audit/result dir ====="
RUN_DIR="result/${MODEL}/runs/${RUN_ID}"
if [[ -d "$RUN_DIR" ]]; then
  find "$RUN_DIR" -maxdepth 2 -type f -printf "%p\t%s bytes\n" | sort
else
  echo "missing: $RUN_DIR"
fi

echo
echo "===== summary.json ====="
if [[ -f "$RUN_DIR/summary.json" ]]; then
  uv run python - <<PY
import json
from pathlib import Path
p = Path("$RUN_DIR/summary.json")
data = json.loads(p.read_text())
for k in ["status", "model", "run_id", "error_type", "error", "phase2"]:
    if k in data:
        print(f"{k}: {data[k]}")
PY
else
  echo "missing summary.json"
fi

echo
echo "===== phase2 outputs ====="
PHASE2_DIR="outputs/phase2/${RUN_ID}"
if [[ -d "$PHASE2_DIR" ]]; then
  find "$PHASE2_DIR" -maxdepth 3 -type f -printf "%p\t%s bytes\n" | sort
  uv run python - <<PY
from pathlib import Path
p = Path("$PHASE2_DIR")
try:
    import pandas as pd
    for name in ["prompts.parquet", "token_spans.parquet", "jbshield_scores.parquet", "model_metadata.parquet"]:
        f = p / name
        if f.exists():
            df = pd.read_parquet(f)
            print(f"{name}: rows={len(df)} cols={list(df.columns)}")
    from safetensors.torch import load_file
    for name in ["hidden_last.safetensors", "hidden_spans.safetensors", "calibration_artifacts/concept_vectors.safetensors"]:
        f = p / name
        if f.exists():
            tensors = load_file(str(f))
            print(f"{name}: " + ", ".join(f"{k}{tuple(v.shape)}" for k, v in tensors.items()))
except Exception as e:
    print("inspect failed:", type(e).__name__, e)
PY
else
  echo "missing: $PHASE2_DIR"
fi

echo
echo "===== hidden representation cache ====="
for CACHE_ID in "Qwen__Qwen3-8B" "$MODEL"; do
  CACHE_DIR="representations/${CACHE_ID}"
  echo "--- $CACHE_DIR ---"
  if [[ -d "$CACHE_DIR" ]]; then
    find "$CACHE_DIR" -maxdepth 2 -type f -printf "%p\t%s bytes\n" | sort
    uv run python - <<PY
from pathlib import Path
import pandas as pd
from safetensors.torch import load_file
root = Path("$CACHE_DIR")
for meta in sorted(root.glob("*/metadata.parquet")):
    df = pd.read_parquet(meta)
    print(f"{meta.parent.name}: metadata rows={len(df)}")
    last = meta.parent / "hidden_last.safetensors"
    if last.exists():
        t = load_file(str(last))["hidden_last"]
        print(f"  hidden_last shape={tuple(t.shape)}")
    tail = meta.parent / "hidden_tail.safetensors"
    if tail.exists():
        t = load_file(str(tail))["hidden_tail"]
        print(f"  hidden_tail shape={tuple(t.shape)}")
PY
  else
    echo "missing"
  fi
done