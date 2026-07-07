import argparse
import json
import math
import struct
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import load_file


PHASE2_REQUIRED_FILES = [
    "prompts.parquet",
    "model_metadata.parquet",
    "token_spans.parquet",
    "behavior_labels.parquet",
    "hidden_last.safetensors",
    "hidden_spans.safetensors",
    "jbshield_scores.parquet",
    "calibration_artifacts/thresholds.parquet",
    "calibration_artifacts/concept_stats.parquet",
    "calibration_artifacts/concept_vectors.safetensors",
]


@dataclass
class AnalysisState:
    output_dir: Path
    warnings: list[str] = field(default_factory=list)
    runs: list[dict] = field(default_factory=list)
    artifact_manifest: list[dict] = field(default_factory=list)
    score_metrics: list[dict] = field(default_factory=list)
    score_distributions: list[dict] = field(default_factory=list)
    sample_scores: list[dict] = field(default_factory=list)
    score_error_analysis: list[dict] = field(default_factory=list)
    thresholds: list[dict] = field(default_factory=list)
    concept_stats: list[dict] = field(default_factory=list)
    critical_layer_summary: list[dict] = field(default_factory=list)
    concept_vector_cosine: list[dict] = field(default_factory=list)
    span_quality: list[dict] = field(default_factory=list)
    hidden_layer_metrics: list[dict] = field(default_factory=list)
    hidden_source_summary: list[dict] = field(default_factory=list)
    phase2_behavior_labels: list[dict] = field(default_factory=list)
    behavior_coverage: list[dict] = field(default_factory=list)
    behavior_score_alignment: list[dict] = field(default_factory=list)
    behavior_detection_metrics: list[dict] = field(default_factory=list)
    result_summaries: list[dict] = field(default_factory=list)


def _read_json(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(payload), f, ensure_ascii=False, indent=2)
        f.write("\n")


def _jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _read_parquet(path, state):
    if not path.exists():
        state.warnings.append(f"missing parquet: {path}")
        return pd.DataFrame()
    return pd.read_parquet(path)


def _write_table(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _state_frames(state):
    return {
        "runs": pd.DataFrame(state.runs),
        "artifact_manifest": pd.DataFrame(state.artifact_manifest),
        "score_metrics": pd.DataFrame(state.score_metrics),
        "score_distributions": pd.DataFrame(state.score_distributions),
        "sample_scores": pd.DataFrame(state.sample_scores),
        "score_error_analysis": pd.DataFrame(state.score_error_analysis),
        "thresholds": pd.DataFrame(state.thresholds),
        "concept_stats": pd.DataFrame(state.concept_stats),
        "critical_layer_summary": pd.DataFrame(state.critical_layer_summary),
        "concept_vector_cosine": pd.DataFrame(state.concept_vector_cosine),
        "span_quality": pd.DataFrame(state.span_quality),
        "hidden_layer_metrics": pd.DataFrame(state.hidden_layer_metrics),
        "hidden_source_summary": pd.DataFrame(state.hidden_source_summary),
        "behavior_coverage": pd.DataFrame(state.behavior_coverage),
        "behavior_score_alignment": pd.DataFrame(state.behavior_score_alignment),
        "behavior_detection_metrics": pd.DataFrame(state.behavior_detection_metrics),
        "result_summaries": pd.DataFrame(state.result_summaries),
    }


def _write_frames(output_dir, frames, export_csv=False):
    for name, frame in frames.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False)
        if export_csv:
            csv_dir = output_dir / "tables"
            csv_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(csv_dir / f"{name}.csv", index=False)


def _file_size(path):
    return path.stat().st_size if path.exists() else 0


def _mb(size_bytes):
    return round(size_bytes / (1024 * 1024), 3)


def _as_float(value):
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _column_or_default(df, column, default):
    if column in df:
        return df[column].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def _json_loads_maybe(value):
    if isinstance(value, dict):
        return value
    if value is None:
        return {}
    try:
        if pd.isna(value):
            return {}
    except TypeError:
        pass
    if isinstance(value, str) and value.strip():
        try:
            payload = json.loads(value)
            return payload if isinstance(payload, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _error_type(label, prediction):
    try:
        label = int(label)
        prediction = int(prediction)
    except (TypeError, ValueError):
        return "unknown"
    if label == 1 and prediction == 1:
        return "tp"
    if label == 0 and prediction == 1:
        return "fp"
    if label == 1 and prediction == 0:
        return "fn"
    if label == 0 and prediction == 0:
        return "tn"
    return "unknown"


def _safe_min(values):
    values = [value for value in values if value is not None and not pd.isna(value)]
    return min(values) if values else None


def _classification_metrics(scores):
    if scores.empty or "label" not in scores or "prediction" not in scores:
        return {
            "num_samples": 0,
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }
    labels = scores["label"].fillna(0).astype(int)
    preds = scores["prediction"].fillna(0).astype(int)
    tp = int(((labels == 1) & (preds == 1)).sum())
    fp = int(((labels == 0) & (preds == 1)).sum())
    fn = int(((labels == 1) & (preds == 0)).sum())
    tn = int(((labels == 0) & (preds == 0)).sum())
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    accuracy = (tp + tn) / total if total else 0.0
    return {
        "num_samples": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def _cosine(a, b):
    denom = torch.linalg.norm(a) * torch.linalg.norm(b)
    if float(denom) == 0.0:
        return None
    return float(torch.dot(a, b) / denom)


def _vector_kind(name):
    if name == "toxic":
        return "toxic", "toxic"
    if name.startswith("jailbreak__"):
        return "jailbreak", name.split("__", 1)[1]
    return "unknown", name


def _discover_phase2_runs(phase2_root, run_prefixes):
    if not phase2_root.exists():
        return []
    runs = [path for path in phase2_root.iterdir() if path.is_dir()]
    if run_prefixes:
        runs = [path for path in runs if any(path.name.startswith(prefix) for prefix in run_prefixes)]
    return sorted(runs, key=lambda item: item.name)


def _discover_result_summaries(result_root, run_prefixes):
    if not result_root.exists():
        return []
    summaries = sorted(result_root.glob("*/runs/*/summary.json"))
    if run_prefixes:
        summaries = [
            path
            for path in summaries
            if any(path.parent.name.startswith(prefix) for prefix in run_prefixes)
        ]
    return summaries


def _model_metadata(run_dir, state):
    metadata = _read_parquet(run_dir / "model_metadata.parquet", state)
    if metadata.empty:
        return {}
    return metadata.iloc[0].to_dict()


def _append_artifact_manifest(run_dir, run_id, model_name, state):
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file():
            continue
        rel_path = path.relative_to(run_dir).as_posix()
        size_bytes = _file_size(path)
        state.artifact_manifest.append(
            {
                "run_id": run_id,
                "model_name": model_name,
                "artifact_path": rel_path,
                "size_bytes": size_bytes,
                "size_mb": _mb(size_bytes),
            }
        )


def _append_score_tables(run_dir, run_id, model_name, state):
    scores = _read_parquet(run_dir / "jbshield_scores.parquet", state)
    if scores.empty:
        return 0
    scores = scores.copy()
    scores["attack_family"] = _column_or_default(scores, "attack_family", "unknown")
    scores["split"] = _column_or_default(scores, "split", "unknown")
    scores["label"] = _column_or_default(scores, "label", 0).astype(int)
    scores["prediction"] = _column_or_default(scores, "prediction", 0).astype(int)

    prompt_metadata = _read_parquet(run_dir / "prompts.parquet", state)
    if not prompt_metadata.empty and "prompt_id" in prompt_metadata:
        keep_columns = [
            column
            for column in ("prompt_id", "base_request_id", "source_group", "prompt_index", "task_type")
            if column in prompt_metadata.columns
        ]
        scores = scores.merge(
            prompt_metadata[keep_columns].drop_duplicates("prompt_id"),
            on="prompt_id",
            how="left",
        )

    enriched_rows = []
    for row in scores.to_dict(orient="records"):
        thresholds = _json_loads_maybe(row.get("thresholds_json"))
        layers = _json_loads_maybe(row.get("layers_json"))
        toxic_threshold = _as_float(thresholds.get("toxic"))
        jailbreak_threshold = _as_float(thresholds.get("jailbreak"))
        toxic_score = _as_float(row.get("toxic_score"))
        jailbreak_score = _as_float(row.get("jailbreak_score"))
        toxic_margin = None if toxic_score is None or toxic_threshold is None else toxic_score - toxic_threshold
        jailbreak_margin = (
            None
            if jailbreak_score is None or jailbreak_threshold is None
            else jailbreak_score - jailbreak_threshold
        )
        enriched = {
            "run_id": run_id,
            "model_name": model_name,
            "score_id": row.get("id"),
            "prompt_id": row.get("prompt_id"),
            "base_request_id": row.get("base_request_id"),
            "source_group": row.get("source_group"),
            "prompt_index": row.get("prompt_index"),
            "split": row.get("split"),
            "attack_family": row.get("attack_family"),
            "label": int(row.get("label", 0)),
            "prediction": int(row.get("prediction", 0)),
            "error_type": _error_type(row.get("label"), row.get("prediction")),
            "toxic_score": toxic_score,
            "jailbreak_score": jailbreak_score,
            "toxic_threshold": toxic_threshold,
            "jailbreak_threshold": jailbreak_threshold,
            "toxic_margin": toxic_margin,
            "jailbreak_margin": jailbreak_margin,
            "min_margin": _safe_min([toxic_margin, jailbreak_margin]),
            "toxic_layer": layers.get("toxic"),
            "jailbreak_layer": layers.get("jailbreak"),
        }
        enriched_rows.append(enriched)
        state.sample_scores.append(enriched)

    for (split, family), group in scores.groupby(["split", "attack_family"], dropna=False):
        metrics = _classification_metrics(group)
        state.score_metrics.append(
            {
                "run_id": run_id,
                "model_name": model_name,
                "split": split,
                "attack_family": family,
                **metrics,
            }
        )

    for (split, family, label), group in scores.groupby(["split", "attack_family", "label"], dropna=False):
        row = {
            "run_id": run_id,
            "model_name": model_name,
            "split": split,
            "attack_family": family,
            "label": int(label) if pd.notna(label) else None,
            "num_samples": int(len(group)),
            "prediction_rate": _as_float(group["prediction"].mean()) if "prediction" in group else None,
        }
        for column in ("toxic_score", "jailbreak_score"):
            if column in group:
                row[f"{column}_mean"] = _as_float(group[column].mean())
                row[f"{column}_std"] = _as_float(group[column].std())
                row[f"{column}_min"] = _as_float(group[column].min())
                row[f"{column}_max"] = _as_float(group[column].max())
        state.score_distributions.append(row)
    return int(len(scores))


def _append_calibration_tables(run_dir, run_id, model_name, state):
    thresholds = _read_parquet(run_dir / "calibration_artifacts" / "thresholds.parquet", state)
    if not thresholds.empty:
        for row in thresholds.to_dict(orient="records"):
            row.update({"run_id": run_id, "model_name": model_name})
            state.thresholds.append(row)

    concept_stats = _read_parquet(run_dir / "calibration_artifacts" / "concept_stats.parquet", state)
    if not concept_stats.empty:
        for row in concept_stats.to_dict(orient="records"):
            row.update({"run_id": run_id, "model_name": model_name})
            state.concept_stats.append(row)


def _append_phase2_behavior_labels(labels, run_id, model_name, state):
    if labels.empty:
        return
    behavior_columns = [
        column
        for column in ("jailbreak_success", "refusal", "judge_score", "output_text")
        if column in labels.columns
    ]
    if not behavior_columns:
        return
    labeled = labels.copy()
    has_behavior = labeled[behavior_columns].notna().any(axis=1)
    labeled = labeled[has_behavior]
    if labeled.empty:
        return
    keep_columns = [
        column
        for column in ("prompt_id", "attack_family", "jailbreak_success", "refusal", "judge_score", "output_text")
        if column in labeled.columns
    ]
    for row in labeled[keep_columns].to_dict(orient="records"):
        row.update(
            {
                "run_id": run_id,
                "model_name": model_name,
                "behavior_source_path": f"{run_id}:behavior_labels.parquet",
            }
        )
        state.phase2_behavior_labels.append(row)


def _append_span_quality(run_dir, run_id, model_name, state):
    spans = _read_parquet(run_dir / "token_spans.parquet", state)
    if spans.empty:
        return
    for (span_name, source), group in spans.groupby(["span_name", "source"], dropna=False):
        state.span_quality.append(
            {
                "run_id": run_id,
                "model_name": model_name,
                "span_name": span_name,
                "source": source,
                "num_spans": int(len(group)),
                "zero_token_spans": int((group["token_count"].fillna(0) <= 0).sum()),
                "token_count_mean": _as_float(group["token_count"].mean()),
                "token_count_min": _as_float(group["token_count"].min()),
                "token_count_max": _as_float(group["token_count"].max()),
                "sequence_length_mean": _as_float(group["sequence_length"].mean()),
                "sequence_length_max": _as_float(group["sequence_length"].max()),
            }
        )


def _append_concept_vector_cosine(run_dir, run_id, model_name, state):
    path = run_dir / "calibration_artifacts" / "concept_vectors.safetensors"
    if not path.exists():
        state.warnings.append(f"missing concept vectors: {path}")
        return
    vectors = load_file(str(path))
    names = sorted(vectors)
    for i, left_name in enumerate(names):
        left_type, left_family = _vector_kind(left_name)
        left = vectors[left_name].detach().cpu().float().flatten()
        for right_name in names[i:]:
            right_type, right_family = _vector_kind(right_name)
            right = vectors[right_name].detach().cpu().float().flatten()
            state.concept_vector_cosine.append(
                {
                    "run_id": run_id,
                    "model_name": model_name,
                    "vector_a": left_name,
                    "vector_a_type": left_type,
                    "vector_a_family": left_family,
                    "vector_b": right_name,
                    "vector_b_type": right_type,
                    "vector_b_family": right_family,
                    "cosine": _cosine(left, right),
                }
            )


def _safetensor_header(path):
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = f.read(header_len)
    return json.loads(header.decode("utf-8"))


def _hidden_metadata(run_dir, filename, tensor_name):
    path = run_dir / filename
    if not path.exists():
        return {"exists": False, "size_bytes": 0}
    header = _safetensor_header(path)
    tensor_info = header.get(tensor_name)
    if tensor_info is None:
        return {"exists": True, "size_bytes": _file_size(path), "tensor_missing": True}
    return {
        "exists": True,
        "size_bytes": _file_size(path),
        "shape": tensor_info.get("shape", []),
        "dtype": tensor_info.get("dtype"),
    }


def _aligned_prompt_labels(run_dir, state):
    prompts = _read_parquet(run_dir / "prompts.parquet", state)
    labels = _read_parquet(run_dir / "behavior_labels.parquet", state)
    if prompts.empty or labels.empty:
        return pd.DataFrame()
    prompts = prompts.reset_index(drop=True).reset_index(names="hidden_index")
    keep = ["prompt_id", "label", "attack_family", "harmful_intent", "jailbreak_present"]
    labels = labels[[column for column in keep if column in labels.columns]]
    merged = prompts.merge(labels, on="prompt_id", how="left", suffixes=("", "_label"))
    if "attack_family_label" in merged:
        merged["attack_family"] = merged["attack_family_label"].fillna(merged.get("attack_family"))
    merged["attack_family"] = merged["attack_family"].fillna("unknown")
    merged["label"] = merged["label"].fillna(0).astype(int)
    return merged


def _centroid(values, mask):
    selected = values[mask]
    if selected.shape[0] == 0:
        return None
    return selected.mean(dim=0)


def _append_centroid_row(state, run_id, model_name, source, layer_idx, comparison, left_name, left, right_name, right):
    if left is None or right is None:
        return
    cosine = _cosine(left, right)
    state.hidden_layer_metrics.append(
        {
            "run_id": run_id,
            "model_name": model_name,
            "source": source,
            "layer": int(layer_idx),
            "comparison": comparison,
            "left_group": left_name,
            "right_group": right_name,
            "centroid_cosine": cosine,
            "centroid_cosine_divergence": None if cosine is None else 1.0 - cosine,
            "left_norm": float(torch.linalg.norm(left)),
            "right_norm": float(torch.linalg.norm(right)),
            "norm_delta": float(torch.linalg.norm(left) - torch.linalg.norm(right)),
        }
    )


def _append_hidden_layer_metrics(run_dir, run_id, model_name, state):
    labels = _aligned_prompt_labels(run_dir, state)
    if labels.empty:
        return

    for source, filename, tensor_name in (
        ("last_token", "hidden_last.safetensors", "hidden_last"),
        ("user_prompt", "hidden_spans.safetensors", "user_prompt"),
    ):
        path = run_dir / filename
        if not path.exists():
            state.warnings.append(f"missing hidden tensor: {path}")
            continue
        tensors = load_file(str(path))
        if tensor_name not in tensors:
            state.warnings.append(f"missing tensor {tensor_name} in {path}")
            continue
        hidden = tensors[tensor_name].detach().cpu().float()
        if hidden.ndim != 3 or hidden.shape[0] != len(labels):
            state.warnings.append(
                f"hidden tensor shape mismatch for {path}: shape={list(hidden.shape)} labels={len(labels)}"
            )
            continue

        attack_family = labels["attack_family"].astype(str)
        label_values = torch.tensor(labels["label"].to_numpy(), dtype=torch.int64)
        harmless_mask = torch.tensor((attack_family == "none").to_numpy(), dtype=torch.bool)
        harmful_mask = torch.tensor((attack_family == "direct_harmful").to_numpy(), dtype=torch.bool)
        abnormal_mask = label_values == 1
        normal_mask = label_values == 0
        attack_families = sorted(
            family for family in attack_family.unique() if family not in {"none", "direct_harmful", "unknown"}
        )

        for layer_idx in range(hidden.shape[1]):
            layer_values = hidden[:, layer_idx, :]
            normal_centroid = _centroid(layer_values, normal_mask)
            abnormal_centroid = _centroid(layer_values, abnormal_mask)
            harmless_centroid = _centroid(layer_values, harmless_mask)
            harmful_centroid = _centroid(layer_values, harmful_mask)
            _append_centroid_row(
                state,
                run_id,
                model_name,
                source,
                layer_idx,
                "label_1_vs_label_0",
                "label_1",
                abnormal_centroid,
                "label_0",
                normal_centroid,
            )
            _append_centroid_row(
                state,
                run_id,
                model_name,
                source,
                layer_idx,
                "direct_harmful_vs_harmless",
                "direct_harmful",
                harmful_centroid,
                "none",
                harmless_centroid,
            )
            for family in attack_families:
                family_mask = torch.tensor((attack_family == family).to_numpy(), dtype=torch.bool)
                family_centroid = _centroid(layer_values, family_mask)
                _append_centroid_row(
                    state,
                    run_id,
                    model_name,
                    source,
                    layer_idx,
                    "attack_vs_direct_harmful",
                    family,
                    family_centroid,
                    "direct_harmful",
                    harmful_centroid,
                )
                _append_centroid_row(
                    state,
                    run_id,
                    model_name,
                    source,
                    layer_idx,
                    "attack_vs_harmless",
                    family,
                    family_centroid,
                    "none",
                    harmless_centroid,
                )


def _append_result_summary(path, state):
    try:
        summary = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        state.warnings.append(f"failed to read result summary {path}: {exc}")
        return
    metrics = summary.get("metrics", {}).get("overall", {})
    phase2 = summary.get("phase2", {})
    state.result_summaries.append(
        {
            "run_id": path.parent.name,
            "model_name": summary.get("model") or path.parents[2].name,
            "summary_path": str(path),
            "status": summary.get("status"),
            "elapsed_seconds": _as_float(summary.get("elapsed_seconds")),
            "accuracy": _as_float(metrics.get("accuracy")),
            "precision": _as_float(metrics.get("precision")),
            "recall": _as_float(metrics.get("recall")),
            "f1": _as_float(metrics.get("f1")),
            "num_samples": metrics.get("num_samples"),
            "phase2_enabled": bool(phase2.get("enabled")),
            "phase2_path": phase2.get("path"),
        }
    )


def _table_from_path(path):
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".jsonl":
        return pd.read_json(path, lines=True)
    if suffix == ".json":
        payload = _read_json(path)
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for key in ("rows", "records", "data", "labels"):
                if isinstance(payload.get(key), list):
                    return pd.DataFrame(payload[key])
            return pd.DataFrame([payload])
    raise ValueError(f"Unsupported behavior label format: {path}")


def _load_behavior_labels(paths, state):
    frames = []
    for path in paths or []:
        try:
            frame = _table_from_path(path)
        except Exception as exc:
            state.warnings.append(f"failed to read behavior labels {path}: {exc}")
            continue
        if frame.empty:
            state.warnings.append(f"empty behavior labels: {path}")
            continue
        frame = frame.copy()
        frame["behavior_source_path"] = str(path)
        if "id" in frame.columns and "score_id" not in frame.columns:
            frame["score_id"] = frame["id"]
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _combine_behavior_labels(phase2_rows, external_labels):
    frames = []
    phase2_frame = pd.DataFrame(phase2_rows)
    if not phase2_frame.empty:
        frames.append(phase2_frame)
    if external_labels is not None and not external_labels.empty:
        frames.append(external_labels)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _normalize_bool_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value != 0)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "success", "succeeded", "refusal", "refused"}:
        return 1
    if normalized in {"0", "false", "no", "n", "failed", "failure", "non_refusal", "not_refused"}:
        return 0
    return None


def _normalize_behavior_frame(frame):
    if frame.empty:
        return frame
    frame = frame.copy()
    for column in ("jailbreak_success", "refusal"):
        if column in frame.columns:
            frame[column] = frame[column].map(_normalize_bool_value)
    if "judge_score" in frame.columns:
        frame["judge_score"] = pd.to_numeric(frame["judge_score"], errors="coerce")
    return frame


def _merge_behavior(sample_scores, behavior_labels):
    if sample_scores.empty or behavior_labels.empty:
        return pd.DataFrame(), None
    sample_scores = sample_scores.copy()
    behavior_labels = _normalize_behavior_frame(behavior_labels)
    key_sets = [
        ["run_id", "score_id"],
        ["run_id", "prompt_id", "attack_family"],
        ["model_name", "prompt_id", "attack_family"],
        ["prompt_id", "attack_family"],
        ["model_name", "prompt_id"],
        ["prompt_id"],
        ["model_name", "base_request_id", "attack_family"],
        ["base_request_id", "attack_family"],
    ]
    behavior_columns = [
        column
        for column in (
            "jailbreak_success",
            "refusal",
            "judge_score",
            "output_text",
            "behavior_source_path",
        )
        if column in behavior_labels.columns
    ]
    for keys in key_sets:
        if not set(keys) <= set(sample_scores.columns) or not set(keys) <= set(behavior_labels.columns):
            continue
        right = behavior_labels[keys + behavior_columns].dropna(how="all", subset=keys)
        if right.empty:
            continue
        right = right.drop_duplicates(keys)
        merged = sample_scores.merge(right, on=keys, how="left")
        matched_columns = [column for column in ("jailbreak_success", "refusal", "judge_score") if column in merged]
        if any(merged[column].notna().any() for column in matched_columns):
            merged["behavior_join_key"] = "+".join(keys)
            return merged, "+".join(keys)
    return pd.DataFrame(), None


def _append_score_error_analysis(state):
    samples = pd.DataFrame(state.sample_scores)
    if samples.empty:
        return
    group_columns = ["model_name", "split", "attack_family", "error_type"]
    for keys, group in samples.groupby(group_columns, dropna=False):
        row = dict(zip(group_columns, keys))
        row.update(
            {
                "num_samples": int(len(group)),
                "prediction_rate": _as_float(group["prediction"].mean()),
                "toxic_score_mean": _as_float(group["toxic_score"].mean()),
                "jailbreak_score_mean": _as_float(group["jailbreak_score"].mean()),
                "toxic_margin_mean": _as_float(group["toxic_margin"].mean()),
                "jailbreak_margin_mean": _as_float(group["jailbreak_margin"].mean()),
                "min_margin_mean": _as_float(group["min_margin"].mean()),
                "toxic_margin_p10": _as_float(group["toxic_margin"].quantile(0.10)),
                "toxic_margin_p90": _as_float(group["toxic_margin"].quantile(0.90)),
                "jailbreak_margin_p10": _as_float(group["jailbreak_margin"].quantile(0.10)),
                "jailbreak_margin_p90": _as_float(group["jailbreak_margin"].quantile(0.90)),
            }
        )
        state.score_error_analysis.append(row)


def _append_critical_layer_summary(state):
    concept_stats = pd.DataFrame(state.concept_stats)
    if concept_stats.empty:
        return
    required = {"model_name", "attack_family", "toxic_layer_normalized_depth", "jailbreak_layer_normalized_depth"}
    if not required <= set(concept_stats.columns):
        return
    for model_name, group in concept_stats.groupby("model_name", dropna=False):
        state.critical_layer_summary.append(
            {
                "summary_scope": "model",
                "model_name": model_name,
                "attack_family": "__all__",
                "num_attack_families": int(group["attack_family"].nunique()),
                "toxic_layer_mean": _as_float(group["toxic_layer"].mean()) if "toxic_layer" in group else None,
                "jailbreak_layer_mean": _as_float(group["jailbreak_layer"].mean()) if "jailbreak_layer" in group else None,
                "toxic_depth_mean": _as_float(group["toxic_layer_normalized_depth"].mean()),
                "jailbreak_depth_mean": _as_float(group["jailbreak_layer_normalized_depth"].mean()),
                "jailbreak_depth_std": _as_float(group["jailbreak_layer_normalized_depth"].std()),
                "jailbreak_depth_min": _as_float(group["jailbreak_layer_normalized_depth"].min()),
                "jailbreak_depth_max": _as_float(group["jailbreak_layer_normalized_depth"].max()),
            }
        )
    for family, group in concept_stats.groupby("attack_family", dropna=False):
        state.critical_layer_summary.append(
            {
                "summary_scope": "attack_family",
                "model_name": "__all__",
                "attack_family": family,
                "num_models": int(group["model_name"].nunique()),
                "toxic_depth_mean": _as_float(group["toxic_layer_normalized_depth"].mean()),
                "jailbreak_depth_mean": _as_float(group["jailbreak_layer_normalized_depth"].mean()),
                "jailbreak_depth_std": _as_float(group["jailbreak_layer_normalized_depth"].std()),
                "jailbreak_depth_min": _as_float(group["jailbreak_layer_normalized_depth"].min()),
                "jailbreak_depth_max": _as_float(group["jailbreak_layer_normalized_depth"].max()),
            }
        )


def _append_hidden_source_summary(state):
    metrics = pd.DataFrame(state.hidden_layer_metrics)
    runs = pd.DataFrame(state.runs)
    if metrics.empty or "centroid_cosine_divergence" not in metrics:
        return
    layer_counts = {}
    if not runs.empty and {"model_name", "num_layers"} <= set(runs.columns):
        for row in runs.to_dict(orient="records"):
            try:
                layer_counts[row["model_name"]] = int(row["num_layers"])
            except (TypeError, ValueError):
                pass
    group_columns = ["model_name", "source", "comparison"]
    for keys, group in metrics.groupby(group_columns, dropna=False):
        group = group.dropna(subset=["centroid_cosine_divergence"])
        if group.empty:
            continue
        idx = group["centroid_cosine_divergence"].idxmax()
        peak = group.loc[idx]
        num_layers = layer_counts.get(keys[0])
        peak_layer = int(peak["layer"])
        state.hidden_source_summary.append(
            {
                "model_name": keys[0],
                "source": keys[1],
                "comparison": keys[2],
                "num_rows": int(len(group)),
                "mean_divergence": _as_float(group["centroid_cosine_divergence"].mean()),
                "max_divergence": _as_float(peak["centroid_cosine_divergence"]),
                "max_layer": peak_layer,
                "max_layer_normalized_depth": (
                    peak_layer / float(num_layers - 1) if num_layers and num_layers > 1 else None
                ),
                "mean_left_norm": _as_float(group["left_norm"].mean()) if "left_norm" in group else None,
                "mean_right_norm": _as_float(group["right_norm"].mean()) if "right_norm" in group else None,
            }
        )


def _append_behavior_analysis(state, behavior_labels):
    sample_scores = pd.DataFrame(state.sample_scores)
    if behavior_labels is None or behavior_labels.empty:
        return
    merged, join_key = _merge_behavior(sample_scores, behavior_labels)
    if sample_scores.empty:
        return
    behavior_columns = [column for column in ("jailbreak_success", "refusal", "judge_score") if column in merged]
    if merged.empty or not behavior_columns:
        state.behavior_coverage.append(
            {
                "join_key": join_key,
                "num_sample_scores": int(len(sample_scores)),
                "num_behavior_matched": 0,
                "coverage": 0.0,
            }
        )
        if behavior_labels is not None and not behavior_labels.empty:
            state.warnings.append("behavior labels were provided but did not match sample scores")
        return

    has_behavior = merged[behavior_columns].notna().any(axis=1)
    state.behavior_coverage.append(
        {
            "join_key": join_key,
            "num_sample_scores": int(len(sample_scores)),
            "num_behavior_matched": int(has_behavior.sum()),
            "coverage": _as_float(has_behavior.mean()),
            "jailbreak_success_labels": int(merged["jailbreak_success"].notna().sum())
            if "jailbreak_success" in merged
            else 0,
            "refusal_labels": int(merged["refusal"].notna().sum()) if "refusal" in merged else 0,
            "judge_score_labels": int(merged["judge_score"].notna().sum()) if "judge_score" in merged else 0,
        }
    )

    for behavior_column in ("jailbreak_success", "refusal"):
        if behavior_column not in merged:
            continue
        labeled = merged[merged[behavior_column].notna()].copy()
        if labeled.empty:
            continue
        labeled[behavior_column] = labeled[behavior_column].astype(int)
        for keys, group in labeled.groupby(["model_name", "attack_family", behavior_column], dropna=False):
            state.behavior_score_alignment.append(
                {
                    "behavior_column": behavior_column,
                    "model_name": keys[0],
                    "attack_family": keys[1],
                    "behavior_value": int(keys[2]),
                    "num_samples": int(len(group)),
                    "prediction_rate": _as_float(group["prediction"].mean()),
                    "toxic_score_mean": _as_float(group["toxic_score"].mean()),
                    "jailbreak_score_mean": _as_float(group["jailbreak_score"].mean()),
                    "toxic_margin_mean": _as_float(group["toxic_margin"].mean()),
                    "jailbreak_margin_mean": _as_float(group["jailbreak_margin"].mean()),
                    "judge_score_mean": _as_float(group["judge_score"].mean()) if "judge_score" in group else None,
                }
            )
        if behavior_column == "jailbreak_success":
            for keys, group in labeled.groupby(["model_name", "attack_family"], dropna=False):
                metrics = _classification_metrics(
                    pd.DataFrame(
                        {
                            "label": group["jailbreak_success"].astype(int),
                            "prediction": group["prediction"].astype(int),
                        }
                    )
                )
                state.behavior_detection_metrics.append(
                    {
                        "behavior_target": behavior_column,
                        "model_name": keys[0],
                        "attack_family": keys[1],
                        **metrics,
                    }
                )


def _append_derived_tables(state, behavior_labels):
    _append_score_error_analysis(state)
    _append_critical_layer_summary(state)
    _append_hidden_source_summary(state)
    _append_behavior_analysis(state, behavior_labels)


def _load_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Plot generation requires matplotlib. Run `uv add matplotlib` or "
            "`python -m pip install matplotlib` in the remote environment."
        ) from exc
    return plt


def _short_label(value, max_len=32):
    text = str(value)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def _plot_heatmap(frame, path, title, value_format=".2f", cmap="viridis"):
    if frame.empty:
        return None
    plt = _load_matplotlib()
    values = frame.astype(float)
    fig_width = max(7.0, min(18.0, 0.8 * len(values.columns) + 3.0))
    fig_height = max(4.5, min(16.0, 0.45 * len(values.index) + 2.5))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), constrained_layout=True)
    image = ax.imshow(values.to_numpy(), aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xticks(range(len(values.columns)))
    ax.set_xticklabels([_short_label(column, 22) for column in values.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(values.index)))
    ax.set_yticklabels([_short_label(index, 36) for index in values.index])
    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            value = values.iat[row_idx, col_idx]
            if pd.notna(value):
                ax.text(col_idx, row_idx, format(value, value_format), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_f1_heatmap(score_metrics, figures_dir):
    if score_metrics.empty or "f1" not in score_metrics:
        return []
    frame = score_metrics.copy()
    frame = frame[frame["attack_family"].notna()]
    if frame.empty:
        return []
    pivot = frame.pivot_table(
        index="model_name",
        columns="attack_family",
        values="f1",
        aggfunc="mean",
    )
    path = figures_dir / "f1_by_model_attack.png"
    result = _plot_heatmap(pivot, path, "JBShield-D F1 by model and attack family", cmap="magma")
    return [result] if result else []


def _plot_critical_layers(concept_stats, figures_dir):
    required = {"model_name", "toxic_layer_normalized_depth", "jailbreak_layer_normalized_depth"}
    if concept_stats.empty or not required <= set(concept_stats.columns):
        return []
    plt = _load_matplotlib()
    summary = (
        concept_stats.groupby("model_name", dropna=False)[
            ["toxic_layer_normalized_depth", "jailbreak_layer_normalized_depth"]
        ]
        .mean()
        .reset_index()
        .sort_values("model_name")
    )
    if summary.empty:
        return []
    x = range(len(summary))
    fig_width = max(8.0, min(18.0, 0.75 * len(summary) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0), constrained_layout=True)
    ax.plot(x, summary["toxic_layer_normalized_depth"], marker="o", label="toxic")
    ax.plot(x, summary["jailbreak_layer_normalized_depth"], marker="s", label="jailbreak")
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("Normalized layer depth")
    ax.set_title("Mean critical layer depth by model")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_short_label(value, 24) for value in summary["model_name"]], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    path = figures_dir / "critical_layer_depth_by_model.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def _plot_concept_vector_cosine(concept_vector_cosine, figures_dir):
    required = {
        "vector_a_type",
        "vector_b_type",
        "vector_a_family",
        "vector_b_family",
        "cosine",
    }
    if concept_vector_cosine.empty or not required <= set(concept_vector_cosine.columns):
        return []
    frame = concept_vector_cosine.copy()
    frame = frame[
        (frame["vector_a_type"] == "jailbreak")
        & (frame["vector_b_type"] == "jailbreak")
        & (frame["vector_a_family"] != frame["vector_b_family"])
    ]
    if frame.empty:
        return []
    families = sorted(set(frame["vector_a_family"]) | set(frame["vector_b_family"]))
    matrix = pd.DataFrame(float("nan"), index=families, columns=families)
    for family in families:
        matrix.loc[family, family] = 1.0
    grouped = (
        frame.groupby(["vector_a_family", "vector_b_family"], dropna=False)["cosine"]
        .mean()
        .reset_index()
    )
    for row in grouped.to_dict(orient="records"):
        left = row["vector_a_family"]
        right = row["vector_b_family"]
        value = row["cosine"]
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    path = figures_dir / "jailbreak_concept_vector_cosine.png"
    result = _plot_heatmap(matrix, path, "Mean jailbreak concept-vector cosine", cmap="coolwarm")
    return [result] if result else []


def _plot_score_distribution_summary(score_distributions, figures_dir):
    required = {"attack_family", "toxic_score_mean", "jailbreak_score_mean"}
    if score_distributions.empty or not required <= set(score_distributions.columns):
        return []
    plt = _load_matplotlib()
    frame = (
        score_distributions.groupby("attack_family", dropna=False)[
            ["toxic_score_mean", "jailbreak_score_mean"]
        ]
        .mean()
        .reset_index()
        .sort_values("attack_family")
    )
    if frame.empty:
        return []
    x = range(len(frame))
    width = 0.38
    fig_width = max(8.0, min(18.0, 0.7 * len(frame) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.0), constrained_layout=True)
    ax.bar([item - width / 2 for item in x], frame["toxic_score_mean"], width=width, label="toxic")
    ax.bar([item + width / 2 for item in x], frame["jailbreak_score_mean"], width=width, label="jailbreak")
    ax.set_title("Mean detector scores by attack family")
    ax.set_ylabel("Mean score")
    ax.set_xticks(list(x))
    ax.set_xticklabels([_short_label(value, 18) for value in frame["attack_family"]], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    path = figures_dir / "score_means_by_attack_family.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def _plot_error_breakdown(score_error_analysis, figures_dir):
    required = {"model_name", "error_type", "num_samples"}
    if score_error_analysis.empty or not required <= set(score_error_analysis.columns):
        return []
    plt = _load_matplotlib()
    frame = (
        score_error_analysis.groupby(["model_name", "error_type"], dropna=False)["num_samples"]
        .sum()
        .reset_index()
    )
    if frame.empty:
        return []
    pivot = frame.pivot_table(
        index="model_name",
        columns="error_type",
        values="num_samples",
        aggfunc="sum",
        fill_value=0,
    )
    ordered_columns = [column for column in ["tp", "fp", "fn", "tn", "unknown"] if column in pivot.columns]
    pivot = pivot[ordered_columns]
    fig_width = max(8.0, min(18.0, 0.75 * len(pivot.index) + 3.0))
    fig, ax = plt.subplots(figsize=(fig_width, 5.5), constrained_layout=True)
    bottom = [0] * len(pivot)
    for column in pivot.columns:
        values = pivot[column].to_numpy()
        ax.bar(range(len(pivot.index)), values, bottom=bottom, label=column.upper())
        bottom = [current + value for current, value in zip(bottom, values)]
    ax.set_title("JBShield-D confusion breakdown by model")
    ax.set_ylabel("Samples")
    ax.set_xticks(range(len(pivot.index)))
    ax.set_xticklabels([_short_label(value, 24) for value in pivot.index], rotation=45, ha="right")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    path = figures_dir / "confusion_breakdown_by_model.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def _plot_behavior_alignment(behavior_score_alignment, figures_dir):
    required = {"behavior_column", "model_name", "attack_family", "behavior_value", "prediction_rate"}
    if behavior_score_alignment.empty or not required <= set(behavior_score_alignment.columns):
        return []
    frame = behavior_score_alignment[
        (behavior_score_alignment["behavior_column"] == "jailbreak_success")
        & (behavior_score_alignment["behavior_value"] == 1)
    ]
    if frame.empty:
        return []
    pivot = frame.pivot_table(
        index="model_name",
        columns="attack_family",
        values="prediction_rate",
        aggfunc="mean",
    )
    path = figures_dir / "prediction_rate_on_successful_jailbreaks.png"
    result = _plot_heatmap(
        pivot,
        path,
        "JBShield prediction rate on behavior-labeled jailbreak successes",
        cmap="viridis",
    )
    return [result] if result else []


def _plot_hidden_divergence(hidden_layer_metrics, figures_dir):
    required = {"source", "comparison", "layer", "centroid_cosine_divergence"}
    if hidden_layer_metrics.empty or not required <= set(hidden_layer_metrics.columns):
        return []
    plt = _load_matplotlib()
    frame = hidden_layer_metrics[
        hidden_layer_metrics["comparison"].isin(
            ["label_1_vs_label_0", "attack_vs_direct_harmful", "attack_vs_harmless"]
        )
    ].copy()
    if frame.empty:
        return []
    summary = (
        frame.groupby(["source", "comparison", "layer"], dropna=False)["centroid_cosine_divergence"]
        .mean()
        .reset_index()
        .sort_values(["source", "comparison", "layer"])
    )
    if summary.empty:
        return []
    fig, ax = plt.subplots(figsize=(10.0, 6.0), constrained_layout=True)
    for (source, comparison), group in summary.groupby(["source", "comparison"], dropna=False):
        ax.plot(
            group["layer"],
            group["centroid_cosine_divergence"],
            marker="o",
            linewidth=1.4,
            label=f"{source}:{comparison}",
        )
    ax.set_title("Layer-wise hidden centroid divergence")
    ax.set_xlabel("Layer")
    ax.set_ylabel("1 - centroid cosine")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    path = figures_dir / "hidden_layer_centroid_divergence.png"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path]


def _write_plots(output_dir, frames):
    figures_dir = output_dir / "figures"
    plot_files = []
    plot_files.extend(_plot_f1_heatmap(frames["score_metrics"], figures_dir))
    plot_files.extend(_plot_critical_layers(frames["concept_stats"], figures_dir))
    plot_files.extend(_plot_concept_vector_cosine(frames["concept_vector_cosine"], figures_dir))
    plot_files.extend(_plot_score_distribution_summary(frames["score_distributions"], figures_dir))
    plot_files.extend(_plot_error_breakdown(frames["score_error_analysis"], figures_dir))
    plot_files.extend(_plot_hidden_divergence(frames["hidden_layer_metrics"], figures_dir))
    plot_files.extend(_plot_behavior_alignment(frames["behavior_score_alignment"], figures_dir))
    return [str(path.relative_to(output_dir)) for path in plot_files if path is not None]


def _analyze_phase2_run(run_dir, state, hidden_analysis):
    run_id = run_dir.name
    metadata = _model_metadata(run_dir, state)
    model_name = metadata.get("model_name") or metadata.get("model_id") or "unknown"
    missing = [rel_path for rel_path in PHASE2_REQUIRED_FILES if not (run_dir / rel_path).exists()]

    prompts = _read_parquet(run_dir / "prompts.parquet", state)
    labels = _read_parquet(run_dir / "behavior_labels.parquet", state)
    hidden_last_meta = _hidden_metadata(run_dir, "hidden_last.safetensors", "hidden_last")
    hidden_spans_meta = _hidden_metadata(run_dir, "hidden_spans.safetensors", "user_prompt")
    num_scores = _append_score_tables(run_dir, run_id, model_name, state)
    _append_calibration_tables(run_dir, run_id, model_name, state)
    _append_phase2_behavior_labels(labels, run_id, model_name, state)
    _append_span_quality(run_dir, run_id, model_name, state)
    _append_concept_vector_cosine(run_dir, run_id, model_name, state)
    _append_artifact_manifest(run_dir, run_id, model_name, state)
    if hidden_analysis:
        _append_hidden_layer_metrics(run_dir, run_id, model_name, state)

    state.runs.append(
        {
            "run_id": run_id,
            "model_name": model_name,
            "model_id": metadata.get("model_id"),
            "model_path": metadata.get("model_path"),
            "chat_template": metadata.get("chat_template"),
            "quantization": metadata.get("quantization"),
            "load_dtype": metadata.get("load_dtype"),
            "device_map": metadata.get("device_map"),
            "num_layers": metadata.get("num_layers"),
            "hidden_size": metadata.get("hidden_size"),
            "phase2_path": str(run_dir),
            "phase2_complete": len(missing) == 0,
            "missing_files": ",".join(missing),
            "num_prompts": int(len(prompts)) if not prompts.empty else 0,
            "num_labels": int(len(labels)) if not labels.empty else 0,
            "num_score_rows": num_scores,
            "hidden_last_shape": json.dumps(hidden_last_meta.get("shape", [])),
            "hidden_last_dtype": hidden_last_meta.get("dtype"),
            "hidden_last_mb": _mb(hidden_last_meta.get("size_bytes", 0)),
            "hidden_spans_shape": json.dumps(hidden_spans_meta.get("shape", [])),
            "hidden_spans_dtype": hidden_spans_meta.get("dtype"),
            "hidden_spans_mb": _mb(hidden_spans_meta.get("size_bytes", 0)),
        }
    )


def _write_summary_md(
    path,
    state,
    hidden_analysis,
    plot_files=None,
    csv_export=False,
    behavior_labels_provided=False,
):
    runs = pd.DataFrame(state.runs)
    result_summaries = pd.DataFrame(state.result_summaries)
    total_phase2_mb = sum(row["size_mb"] for row in state.artifact_manifest)
    plot_files = plot_files or []
    lines = [
        "# JBShield Remote Analysis Summary",
        "",
        f"- Phase2 runs: {len(state.runs)}",
        f"- Result summaries: {len(state.result_summaries)}",
        f"- Artifact size scanned: {total_phase2_mb:.3f} MB",
        f"- Hidden tensor layer analysis: {'enabled' if hidden_analysis else 'disabled'}",
        f"- Behavior label alignment: {'enabled' if behavior_labels_provided else 'not provided'}",
        f"- CSV export: {'enabled' if csv_export else 'disabled'}",
        f"- Figures: {len(plot_files)}",
        f"- Warnings: {len(state.warnings)}",
        "",
    ]
    if plot_files:
        lines.extend(["## Figures", ""])
        lines.extend(f"- `{plot_file}`" for plot_file in plot_files)
        lines.append("")
    if not runs.empty:
        compact = runs[
            [
                "run_id",
                "model_name",
                "phase2_complete",
                "num_prompts",
                "num_score_rows",
                "hidden_last_mb",
                "hidden_spans_mb",
            ]
        ].head(30)
        lines.extend(["## Runs", "", "```text", compact.to_string(index=False), "```", ""])
    if not result_summaries.empty:
        compact = result_summaries[
            ["run_id", "model_name", "status", "accuracy", "f1", "num_samples"]
        ].head(30)
        lines.extend(["## Result Metrics", "", "```text", compact.to_string(index=False), "```", ""])
    if state.warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in state.warnings[:100])
        if len(state.warnings) > 100:
            lines.append(f"- ... {len(state.warnings) - 100} more")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _zip_output(output_dir):
    archive = output_dir.with_suffix(".zip")
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(output_dir.parent))
    return archive


def analyze_remote_artifacts(
    phase2_root,
    result_root,
    output_dir,
    run_prefixes=None,
    behavior_label_paths=None,
    hidden_analysis=False,
    plots=False,
    export_csv=False,
    zip_output=False,
):
    phase2_root = Path(phase2_root)
    result_root = Path(result_root)
    output_dir = Path(output_dir)
    run_prefixes = [prefix for prefix in (run_prefixes or []) if prefix]
    state = AnalysisState(output_dir=output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    external_behavior_labels = _load_behavior_labels(behavior_label_paths or [], state)

    phase2_runs = _discover_phase2_runs(phase2_root, run_prefixes)
    if not phase2_runs:
        state.warnings.append(f"no Phase2 runs found under {phase2_root}")
    for run_dir in phase2_runs:
        _analyze_phase2_run(run_dir, state, hidden_analysis=hidden_analysis)

    result_summaries = _discover_result_summaries(result_root, run_prefixes)
    if not result_summaries:
        state.warnings.append(f"no result summaries found under {result_root}")
    for summary_path in result_summaries:
        _append_result_summary(summary_path, state)

    behavior_labels = _combine_behavior_labels(state.phase2_behavior_labels, external_behavior_labels)
    _append_derived_tables(state, behavior_labels)
    export_csv = bool(export_csv or plots)
    frames = _state_frames(state)
    _write_frames(output_dir, frames, export_csv=export_csv)
    plot_files = _write_plots(output_dir, frames) if plots else []
    _write_json(
        output_dir / "summary.json",
        {
            "phase2_root": str(phase2_root),
            "result_root": str(result_root),
            "output_dir": str(output_dir),
            "run_prefixes": run_prefixes,
            "behavior_label_paths": [str(path) for path in (behavior_label_paths or [])],
            "behavior_labels_provided": bool(behavior_label_paths) or bool(state.phase2_behavior_labels),
            "phase2_behavior_label_rows": len(state.phase2_behavior_labels),
            "external_behavior_label_rows": int(len(external_behavior_labels))
            if external_behavior_labels is not None
            else 0,
            "hidden_analysis": hidden_analysis,
            "plots": plots,
            "plot_files": plot_files,
            "csv_export": export_csv,
            "num_phase2_runs": len(state.runs),
            "num_result_summaries": len(state.result_summaries),
            "num_warnings": len(state.warnings),
            "warnings": state.warnings,
        },
    )
    _write_summary_md(
        output_dir / "summary.md",
        state,
        hidden_analysis=hidden_analysis,
        plot_files=plot_files,
        csv_export=export_csv,
        behavior_labels_provided=bool(behavior_label_paths) or bool(state.phase2_behavior_labels),
    )
    archive = _zip_output(output_dir) if zip_output else None
    return {
        "output_dir": str(output_dir),
        "archive": str(archive) if archive else None,
        "plots": plot_files,
        "num_phase2_runs": len(state.runs),
        "num_result_summaries": len(state.result_summaries),
        "num_warnings": len(state.warnings),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Summarize large JBShield result/Phase2 artifacts on a remote machine."
    )
    parser.add_argument("--phase2-root", default="outputs/phase2")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--output-dir", default="analysis/jbshield_remote")
    parser.add_argument(
        "--run-prefix",
        action="append",
        default=[],
        help="Only analyze runs whose run_id starts with this prefix. Can be repeated.",
    )
    parser.add_argument(
        "--behavior-labels",
        action="append",
        default=[],
        help=(
            "Optional CSV/JSONL/JSON/parquet behavior labels to join with sample scores. "
            "Expected columns include prompt_id or score_id, and optionally "
            "jailbreak_success, refusal, judge_score, output_text."
        ),
    )
    parser.add_argument(
        "--hidden-analysis",
        action="store_true",
        help="Read hidden_last/hidden_spans safetensors and compute layer-level centroid divergence.",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Write PNG figures under output-dir/figures. Implies --csv.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Write CSV copies of all summary tables under output-dir/tables.",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a compact zip archive of the analysis output directory.",
    )
    args = parser.parse_args()

    summary = analyze_remote_artifacts(
        phase2_root=args.phase2_root,
        result_root=args.result_root,
        output_dir=args.output_dir,
        run_prefixes=args.run_prefix,
        behavior_label_paths=args.behavior_labels,
        hidden_analysis=args.hidden_analysis,
        plots=args.plots,
        export_csv=args.csv,
        zip_output=args.zip,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
