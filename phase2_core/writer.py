import json
from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import save_file


def _to_jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _json(value):
    return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True)


def _as_scalar(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _stack_layer_major(layer_major):
    if not layer_major:
        return torch.empty(0)
    per_layer = [torch.stack(prompts, dim=0) for prompts in layer_major]
    return torch.stack(per_layer, dim=0).transpose(0, 1).contiguous()


def _concat_layer_major(groups, key):
    combined = None
    for group in groups:
        layer_major = group.get(key)
        if layer_major is None:
            continue
        if combined is None:
            combined = [[] for _ in range(len(layer_major))]
        for layer_idx, values in enumerate(layer_major):
            combined[layer_idx].extend(values)
    return combined


def _split_from_group(group_name):
    if group_name.startswith("calibration_"):
        return "calibration"
    if group_name.startswith("test_"):
        return "test"
    return "unknown"


def _label_metadata(group_name):
    if "harmless" in group_name:
        return {
            "attack_family": "none",
            "harmful_intent": False,
            "jailbreak_present": False,
            "label": 0,
        }
    if "harmful" in group_name:
        return {
            "attack_family": "direct_harmful",
            "harmful_intent": True,
            "jailbreak_present": False,
            "label": 0,
        }
    family = group_name.replace("calibration_", "").replace("test_", "")
    return {
        "attack_family": family,
        "harmful_intent": True,
        "jailbreak_present": True,
        "label": 1,
    }


def _prompt_id(model_name, group_name, idx):
    return f"{model_name}:{group_name}:{idx}"


def _write_parquet(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def _depth(layer, num_layers):
    if num_layers <= 1:
        return 0.0
    return float(layer) / float(num_layers - 1)


def write_phase2_outputs(
    output_dir,
    run_id,
    model_metadata,
    groups,
    sample_records,
    thresholds,
    critical_layers,
    calibration_vectors,
    deltas,
):
    out_dir = Path(output_dir) / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prompt_rows = []
    label_rows = []
    generation_rows = []
    span_rows = []
    for group in groups:
        group_name = group["group_name"]
        split = _split_from_group(group_name)
        label_meta = _label_metadata(group_name)
        prompts = group["prompts"]
        span_records = group.get("span_records") or []
        for idx, prompt in enumerate(prompts):
            prompt_id = _prompt_id(model_metadata["model_name"], group_name, idx)
            prompt_rows.append(
                {
                    "model_id": model_metadata["model_id"],
                    "prompt_id": prompt_id,
                    "base_request_id": f"{group_name}:{idx}",
                    "task_type": "jailbreak",
                    "split": split,
                    "source_group": group_name,
                    "prompt_index": idx,
                    "messages_json": _json([{"role": "user", "name": "user_prompt", "content": prompt}]),
                    "metadata_json": _json({"attack_family": label_meta["attack_family"]}),
                }
            )
            label_rows.append(
                {
                    "model_id": model_metadata["model_id"],
                    "prompt_id": prompt_id,
                    "harmful_intent": label_meta["harmful_intent"],
                    "jailbreak_present": label_meta["jailbreak_present"],
                    "jailbreak_success": None,
                    "refusal": None,
                    "label": label_meta["label"],
                    "attack_family": label_meta["attack_family"],
                }
            )
            generation_rows.append(
                {
                    "model_id": model_metadata["model_id"],
                    "prompt_id": prompt_id,
                    "output_text": None,
                    "max_new_tokens": None,
                    "temperature": None,
                    "top_p": None,
                    "seed": None,
                }
            )
            span = span_records[idx] if idx < len(span_records) else {}
            start = span.get("user_prompt_start", 0)
            end = span.get("user_prompt_end", span.get("sequence_length", 0))
            span_rows.append(
                {
                    "model_id": model_metadata["model_id"],
                    "prompt_id": prompt_id,
                    "span_name": "user_prompt",
                    "start": int(start),
                    "end": int(end),
                    "token_count": int(max(0, end - start)),
                    "source": span.get("span_source", "unknown"),
                    "sequence_length": int(span.get("sequence_length", 0)),
                }
            )
            last_idx = int(max(0, span.get("sequence_length", 1) - 1))
            span_rows.append(
                {
                    "model_id": model_metadata["model_id"],
                    "prompt_id": prompt_id,
                    "span_name": "last_input_token",
                    "start": last_idx,
                    "end": last_idx + 1,
                    "token_count": 1 if span.get("sequence_length", 0) else 0,
                    "source": "last_input_token",
                    "sequence_length": int(span.get("sequence_length", 0)),
                }
            )

    _write_parquet(out_dir / "prompts.parquet", prompt_rows)
    _write_parquet(out_dir / "behavior_labels.parquet", label_rows)
    _write_parquet(out_dir / "generation_outputs.parquet", generation_rows)
    _write_parquet(out_dir / "token_spans.parquet", span_rows)
    _write_parquet(out_dir / "model_metadata.parquet", [model_metadata])

    hidden_last = _stack_layer_major(_concat_layer_major(groups, "embeddings"))
    save_file({"hidden_last": hidden_last}, str(out_dir / "hidden_last.safetensors"))

    span_layer_major = _concat_layer_major(groups, "span_embeddings")
    if span_layer_major is not None:
        hidden_spans = _stack_layer_major(span_layer_major)
    else:
        hidden_spans = torch.empty(0)
    save_file({"user_prompt": hidden_spans}, str(out_dir / "hidden_spans.safetensors"))

    score_rows = []
    for row in sample_records:
        record = dict(row)
        record["model_id"] = model_metadata["model_id"]
        record.setdefault("prompt_id", record.get("id"))
        record["thresholds_json"] = _json(record.pop("thresholds", {}))
        record["layers_json"] = _json(record.pop("layers", {}))
        score_rows.append(record)
    _write_parquet(out_dir / "jbshield_scores.parquet", score_rows)

    num_layers = int(model_metadata.get("num_layers") or 0)
    threshold_rows = []
    concept_rows = []
    toxic_layer = int(critical_layers["toxic"])
    for family, family_thresholds in thresholds.items():
        jb_layer = int(critical_layers["jailbreak"][family])
        threshold_rows.extend(
            [
                {
                    "model_id": model_metadata["model_id"],
                    "attack_family": family,
                    "threshold_type": "toxic",
                    "threshold": float(_as_scalar(family_thresholds["toxic"])),
                    "layer": toxic_layer,
                    "normalized_depth": _depth(toxic_layer, num_layers),
                },
                {
                    "model_id": model_metadata["model_id"],
                    "attack_family": family,
                    "threshold_type": "jailbreak",
                    "threshold": float(_as_scalar(family_thresholds["jailbreak"])),
                    "layer": jb_layer,
                    "normalized_depth": _depth(jb_layer, num_layers),
                },
            ]
        )
        concept_rows.append(
            {
                "model_id": model_metadata["model_id"],
                "attack_family": family,
                "toxic_layer": toxic_layer,
                "jailbreak_layer": jb_layer,
                "toxic_layer_normalized_depth": _depth(toxic_layer, num_layers),
                "jailbreak_layer_normalized_depth": _depth(jb_layer, num_layers),
                "toxic_delta": float(_as_scalar(deltas.get("toxic", 0.0))),
                "jailbreak_delta": float(_as_scalar(deltas.get("jailbreak", {}).get(family, 0.0))),
            }
        )
    artifact_dir = out_dir / "calibration_artifacts"
    _write_parquet(artifact_dir / "thresholds.parquet", threshold_rows)
    _write_parquet(artifact_dir / "concept_stats.parquet", concept_rows)

    vector_tensors = {}
    if calibration_vectors.get("toxic") is not None:
        vector_tensors["toxic"] = calibration_vectors["toxic"].detach().cpu()
    for family, vector in calibration_vectors.get("jailbreak", {}).items():
        vector_tensors[f"jailbreak__{family}"] = vector.detach().cpu()
    save_file(vector_tensors, str(artifact_dir / "concept_vectors.safetensors"))

    return out_dir
