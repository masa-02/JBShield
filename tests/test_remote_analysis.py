import json
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from analyze_remote_artifacts import analyze_remote_artifacts
from phase2_core.writer import write_phase2_outputs


def _layer_major(base, count, layers=2, hidden=3):
    values = []
    for layer_idx in range(layers):
        layer_values = []
        for sample_idx in range(count):
            layer_values.append(torch.ones(hidden) * (base + layer_idx + sample_idx * 0.1))
        values.append(layer_values)
    return values


def _span_records(count):
    return [
        {
            "prompt_index": idx,
            "sequence_length": 5 + idx,
            "user_prompt_start": 0,
            "user_prompt_end": 5 + idx,
            "span_source": "subsequence",
        }
        for idx in range(count)
    ]


def test_remote_analysis_writes_compact_tables():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        phase2_root = root / "outputs" / "phase2"
        result_root = root / "result"
        run_id = "qwen-smoke-run"
        groups = [
            {
                "group_name": "calibration_harmless",
                "prompts": ["safe a", "safe b"],
                "embeddings": _layer_major(1.0, 2),
                "span_embeddings": _layer_major(1.5, 2),
                "span_records": _span_records(2),
            },
            {
                "group_name": "calibration_harmful",
                "prompts": ["harm a", "harm b"],
                "embeddings": _layer_major(3.0, 2),
                "span_embeddings": _layer_major(3.5, 2),
                "span_records": _span_records(2),
            },
            {
                "group_name": "test_gcg",
                "prompts": ["jb a", "jb b"],
                "embeddings": _layer_major(5.0, 2),
                "span_embeddings": _layer_major(5.5, 2),
                "span_records": _span_records(2),
            },
        ]
        write_phase2_outputs(
            phase2_root,
            run_id,
            {
                "model_id": "Qwen/test",
                "model_name": "qwen-test",
                "model_path": "Qwen/test",
                "chat_template": "hf",
                "tokenizer_class": "FakeTokenizer",
                "model_class": "FakeModel",
                "num_layers": 2,
                "hidden_size": 3,
                "vocab_size": 128,
                "dtype": "torch.float16",
                "device": "cuda:0",
                "quantization": "8bit",
                "load_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "quant_type": None,
                "double_quant": None,
                "device_map": "auto",
                "trust_remote_code": True,
                "extraction_version": "test",
            },
            groups,
            [
                {
                    "id": "qwen-test:gcg:test:jailbreak:0",
                    "prompt_id": "qwen-test:test_gcg:0",
                    "split": "test",
                    "attack_family": "gcg",
                    "label": 1,
                    "toxic_score": 0.8,
                    "jailbreak_score": 0.9,
                    "prediction": 1,
                    "thresholds": {"toxic": 0.5, "jailbreak": 0.6},
                    "layers": {"toxic": 0, "jailbreak": 1},
                },
                {
                    "id": "qwen-test:gcg:test:jailbreak:1",
                    "prompt_id": "qwen-test:test_gcg:1",
                    "split": "test",
                    "attack_family": "gcg",
                    "label": 1,
                    "toxic_score": 0.2,
                    "jailbreak_score": 0.4,
                    "prediction": 0,
                    "thresholds": {"toxic": 0.5, "jailbreak": 0.6},
                    "layers": {"toxic": 0, "jailbreak": 1},
                },
            ],
            {"gcg": {"toxic": 0.5, "jailbreak": 0.6}},
            {"toxic": 0, "jailbreak": {"gcg": 1}},
            {
                "toxic": torch.tensor([1.0, 0.0, 0.0]),
                "jailbreak": {"gcg": torch.tensor([0.0, 1.0, 0.0])},
            },
            {"toxic": torch.tensor(0.1), "jailbreak": {"gcg": torch.tensor(0.2)}},
        )

        summary_dir = result_root / "qwen-test" / "runs" / run_id
        summary_dir.mkdir(parents=True)
        (summary_dir / "summary.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "model": "qwen-test",
                    "elapsed_seconds": 12.0,
                    "metrics": {
                        "overall": {
                            "accuracy": 0.5,
                            "precision": 1.0,
                            "recall": 0.5,
                            "f1": 2 / 3,
                            "num_samples": 2,
                        }
                    },
                    "phase2": {"enabled": True, "path": str(phase2_root / run_id)},
                }
            ),
            encoding="utf-8",
        )

        output_dir = root / "analysis"
        summary = analyze_remote_artifacts(
            phase2_root=phase2_root,
            result_root=result_root,
            output_dir=output_dir,
            run_prefixes=["qwen-smoke"],
            hidden_analysis=True,
            export_csv=True,
        )

        assert summary["num_phase2_runs"] == 1
        assert summary["num_result_summaries"] == 1
        assert (output_dir / "summary.md").exists()
        assert (output_dir / "tables" / "score_metrics.csv").exists()

        runs = pd.read_parquet(output_dir / "runs.parquet")
        assert runs.loc[0, "model_name"] == "qwen-test"
        assert bool(runs.loc[0, "phase2_complete"]) is True

        score_metrics = pd.read_parquet(output_dir / "score_metrics.parquet")
        assert score_metrics.loc[0, "attack_family"] == "gcg"
        assert score_metrics.loc[0, "recall"] == 0.5

        hidden_metrics = pd.read_parquet(output_dir / "hidden_layer_metrics.parquet")
        assert {"last_token", "user_prompt"} <= set(hidden_metrics["source"])
        assert "attack_vs_harmless" in set(hidden_metrics["comparison"])

        vector_cosine = pd.read_parquet(output_dir / "concept_vector_cosine.parquet")
        assert {"toxic", "jailbreak__gcg"} <= set(vector_cosine["vector_a"]) | set(vector_cosine["vector_b"])


if __name__ == "__main__":
    test_remote_analysis_writes_compact_tables()
