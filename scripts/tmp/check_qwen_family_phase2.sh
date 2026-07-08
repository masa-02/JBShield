#!/usr/bin/env bash
set -euo pipefail

MANIFEST="${1:-configs/runtime/manifests/_qwen.txt}"
RUN_PREFIX="${2:-jbshield-qwen-phase2}"
OUT_CSV="${3:-scripts/tmp/qwen_family_phase2_status.csv}"

export MANIFEST RUN_PREFIX OUT_CSV

uv run python - <<'PY'
import json
import os
from pathlib import Path

import pandas as pd
import yaml
from safetensors import safe_open

manifest = Path(os.environ["MANIFEST"])
run_prefix = os.environ["RUN_PREFIX"]
out_csv = Path(os.environ["OUT_CSV"])

required = [
    "prompts.parquet",
    "model_metadata.parquet",
    "token_spans.parquet",
    "behavior_labels.parquet",
    "generation_outputs.parquet",
    "hidden_last.safetensors",
    "hidden_spans.safetensors",
    "jbshield_scores.parquet",
    "calibration_artifacts/thresholds.parquet",
    "calibration_artifacts/concept_stats.parquet",
    "calibration_artifacts/concept_vectors.safetensors",
]

def read_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {"_read_error": True}

def parquet_rows(path):
    if not path.exists():
        return None
    try:
        return len(pd.read_parquet(path))
    except Exception:
        return "ERR"

def tensor_shapes(path):
    if not path.exists():
        return {}
    try:
        out = {}
        with safe_open(str(path), framework="pt", device="cpu") as f:
            for key in f.keys():
                out[key] = tuple(f.get_slice(key).get_shape())
        return out
    except Exception as e:
        return {"_error": type(e).__name__}

def read_manifest_models(path):
    models = {}
    if not path.exists():
        return models
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        cfg = Path(line)
        if not cfg.exists():
            models[cfg.stem] = {"config": line, "model_path": None, "quantization": None}
            continue
        data = yaml.safe_load(cfg.read_text()) or {}
        model = data.get("model", {}) or {}
        loading = data.get("model_loading", {}) or {}
        name = model.get("name") or cfg.stem
        models[name] = {
            "config": line,
            "model_path": model.get("path"),
            "quantization": loading.get("quantization"),
        }
    return models

models = read_manifest_models(manifest)

# manifestにないが既に出力があるQwen runも拾う
for d in sorted(Path("outputs/phase2").glob(f"{run_prefix}-qwen*")):
    name = d.name[len(run_prefix) + 1:]
    models.setdefault(name, {"config": None, "model_path": None, "quantization": None})

rows = []
for name, meta in sorted(models.items()):
    run_id = f"{run_prefix}-{name}"
    phase_dir = Path("outputs/phase2") / run_id
    result_dir = Path("result") / name / "runs" / run_id
    summary = read_json(result_dir / "summary.json")

    missing = [rel for rel in required if not (phase_dir / rel).exists()]

    prompts = parquet_rows(phase_dir / "prompts.parquet")
    spans = parquet_rows(phase_dir / "token_spans.parquet")
    scores = parquet_rows(phase_dir / "jbshield_scores.parquet")
    labels = parquet_rows(phase_dir / "behavior_labels.parquet")

    hidden_last = tensor_shapes(phase_dir / "hidden_last.safetensors")
    hidden_spans = tensor_shapes(phase_dir / "hidden_spans.safetensors")
    concept_vectors = tensor_shapes(phase_dir / "calibration_artifacts" / "concept_vectors.safetensors")

    hidden_last_shape = next(iter(hidden_last.values()), None) if hidden_last else None
    hidden_spans_shape = next(iter(hidden_spans.values()), None) if hidden_spans else None
    concept_vector_count = len([k for k in concept_vectors if not k.startswith("_")])

    has_required = phase_dir.exists() and not missing
    shape_ok = (
        isinstance(prompts, int)
        and isinstance(spans, int)
        and isinstance(scores, int)
        and hidden_last_shape is not None
        and hidden_spans_shape is not None
        and hidden_last_shape[0] == prompts
        and hidden_spans_shape[0] == prompts
        and spans == prompts * 2
    )
    full_default_rows = prompts == 7400 and spans == 14800 and scores == 12500 and concept_vector_count == 10

    if has_required and shape_ok and full_default_rows:
        verdict = "complete_full"
    elif has_required and shape_ok:
        verdict = "complete_partial_or_subset"
    elif phase_dir.exists():
        verdict = "incomplete_phase2"
    else:
        verdict = "missing_phase2"

    rows.append({
        "model": name,
        "config": meta.get("config"),
        "model_path": meta.get("model_path"),
        "quantization": meta.get("quantization"),
        "run_id": run_id,
        "summary_status": summary.get("status"),
        "summary_error_type": summary.get("error_type"),
        "phase2_exists": phase_dir.exists(),
        "missing_files": ",".join(missing),
        "prompts_rows": prompts,
        "token_spans_rows": spans,
        "behavior_labels_rows": labels,
        "jbshield_scores_rows": scores,
        "hidden_last_shape": str(hidden_last_shape),
        "hidden_spans_shape": str(hidden_spans_shape),
        "concept_vector_count": concept_vector_count,
        "verdict": verdict,
    })

df = pd.DataFrame(rows)
out_csv.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out_csv, index=False)

cols = [
    "model", "quantization", "summary_status", "summary_error_type",
    "phase2_exists", "prompts_rows", "token_spans_rows",
    "jbshield_scores_rows", "hidden_last_shape", "concept_vector_count", "verdict",
]
print(df[cols].to_string(index=False))
print()
print(f"wrote: {out_csv}")
PY