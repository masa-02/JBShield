import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2_core.spans import AmbiguousSpanError, SpanMappingError, resolve_user_prompt_span
from phase2_core.writer import write_phase2_outputs
from utils import (
    _as_embedding_list,
    _bitsandbytes_config,
    _to_model_inputs,
    collect_hidden_summaries,
    get_model_hidden_size,
    get_model_num_hidden_layers,
    get_model_vocab_size,
    interpret_difference_matrix,
)


class FakeTokenizer:
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = [ord(ch) for ch in text]
        if add_special_tokens:
            ids = [1] + ids + [2]
        return {"input_ids": ids}

    def encode(self, text, add_special_tokens=False):
        ids = [ord(ch) for ch in text]
        if add_special_tokens:
            return [1] + ids + [2]
        return ids


class BatchEncodingLike:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __contains__(self, key):
        return key == "input_ids"

    def __getitem__(self, key):
        if key != "input_ids":
            raise KeyError(key)
        return self.input_ids


class FakeBatchEncodingTokenizer(FakeTokenizer):
    def __call__(self, text, add_special_tokens=False, return_tensors=None):
        ids = self.encode(text, add_special_tokens=add_special_tokens)
        return BatchEncodingLike(ids)


class BatchEncodingTensorLike:
    def __init__(self, input_ids):
        self.input_ids = input_ids

    def __contains__(self, key):
        return key == "input_ids"

    def __getitem__(self, key):
        if key != "input_ids":
            raise KeyError(key)
        return self.input_ids

    def keys(self):
        return ["input_ids"]

    def to(self, device):
        self.input_ids = self.input_ids.to(device)
        return self


class AddLayer(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, hidden_states):
        return hidden_states + self.value


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.layers = nn.ModuleList([AddLayer(1.0), AddLayer(2.0)])

    def forward(self, input_ids, attention_mask=None, return_dict=True, output_hidden_states=False, use_cache=False):
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return {"last_hidden_state": hidden_states}


class FakeSummaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"num_hidden_layers": 2, "hidden_size": 4})()
        self.model = FakeBackbone()


class FakeGemma3Config:
    def __init__(self):
        self.text_config = type(
            "TextConfig",
            (),
            {"num_hidden_layers": 2, "hidden_size": 4, "vocab_size": 16},
        )()

    def get_text_config(self):
        return self.text_config


class FakeGemma3Outer(nn.Module):
    def __init__(self):
        super().__init__()
        self.language_model = FakeBackbone()


class FakeGemma3SummaryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = FakeGemma3Config()
        self.model = FakeGemma3Outer()


def test_span_resolution_success_and_failures():
    tokenizer = FakeTokenizer()
    input_ids = [10] + tokenizer.encode("hello", add_special_tokens=False) + [11]
    span = resolve_user_prompt_span(input_ids, tokenizer, "hello", strict=True)
    assert span["start"] == 1
    assert span["end"] == 6
    assert span["source"] == "subsequence"

    ambiguous_ids = tokenizer.encode("aa aa", add_special_tokens=False)
    try:
        resolve_user_prompt_span(ambiguous_ids, tokenizer, "aa", strict=True)
    except AmbiguousSpanError:
        pass
    else:
        raise AssertionError("ambiguous span should raise")

    try:
        resolve_user_prompt_span(input_ids, tokenizer, "missing", strict=True)
    except SpanMappingError:
        pass
    else:
        raise AssertionError("missing span should raise")


def test_span_resolution_accepts_batch_encoding_like_tokenizer_output():
    tokenizer = FakeBatchEncodingTokenizer()
    input_ids = tokenizer.encode("prefix hello suffix", add_special_tokens=False)
    span = resolve_user_prompt_span(input_ids, tokenizer, "hello", strict=True)
    assert span["source"] == "subsequence"
    assert span["start"] == len("prefix ")
    assert span["end"] == len("prefix hello")


def test_phase2_writer_outputs_expected_schema():
    with tempfile.TemporaryDirectory() as tmp:
        groups = [
            {
                "group_name": "test_gcg",
                "prompts": ["hello"],
                "embeddings": [
                    [torch.ones(3)],
                    [torch.ones(3) * 2],
                ],
                "span_embeddings": [
                    [torch.ones(3) * 3],
                    [torch.ones(3) * 4],
                ],
                "span_records": [
                    {
                        "prompt_index": 0,
                        "sequence_length": 5,
                        "user_prompt_start": 0,
                        "user_prompt_end": 5,
                        "span_source": "subsequence",
                    }
                ],
            }
        ]
        out_dir = write_phase2_outputs(
            tmp,
            "run-a",
            {
                "model_id": "fake/model",
                "model_name": "fake",
                "model_path": "fake/model",
                "chat_template": "hf",
                "tokenizer_class": "FakeTokenizer",
                "model_class": "FakeModel",
                "num_layers": 2,
                "hidden_size": 3,
                "vocab_size": 128,
                "dtype": "torch.float32",
                "device": "cpu",
                "quantization": "8bit",
                "load_dtype": "bfloat16",
                "compute_dtype": "bfloat16",
                "quant_type": None,
                "double_quant": None,
                "device_map": "auto",
                "trust_remote_code": False,
                "extraction_version": "test",
            },
            groups,
            [
                {
                    "id": "fake:gcg:test:jailbreak:0",
                    "prompt_id": "fake:test_gcg:0",
                    "split": "test",
                    "attack_family": "gcg",
                    "label": 1,
                    "toxic_score": 0.7,
                    "jailbreak_score": 0.8,
                    "prediction": 1,
                    "thresholds": {"toxic": 0.5, "jailbreak": 0.6},
                    "layers": {"toxic": 0, "jailbreak": 1},
                }
            ],
            {"gcg": {"toxic": 0.5, "jailbreak": 0.6}},
            {"toxic": 0, "jailbreak": {"gcg": 1}},
            {"toxic": torch.ones(3), "jailbreak": {"gcg": torch.ones(3) * 2}},
            {"toxic": torch.tensor(0.1), "jailbreak": {"gcg": torch.tensor(0.2)}},
        )

        expected = [
            "prompts.parquet",
            "model_metadata.parquet",
            "token_spans.parquet",
            "generation_outputs.parquet",
            "behavior_labels.parquet",
            "hidden_last.safetensors",
            "hidden_spans.safetensors",
            "jbshield_scores.parquet",
            "calibration_artifacts/thresholds.parquet",
            "calibration_artifacts/concept_vectors.safetensors",
            "calibration_artifacts/concept_stats.parquet",
        ]
        for rel_path in expected:
            assert (out_dir / rel_path).exists(), rel_path

        spans = pd.read_parquet(out_dir / "token_spans.parquet")
        assert set(spans["span_name"]) == {"user_prompt", "last_input_token"}

        scores = pd.read_parquet(out_dir / "jbshield_scores.parquet")
        assert list(scores["prompt_id"]) == ["fake:test_gcg:0"]

        metadata = pd.read_parquet(out_dir / "model_metadata.parquet")
        assert metadata.loc[0, "quantization"] == "8bit"
        assert metadata.loc[0, "device_map"] == "auto"

        hidden_last = load_file(str(out_dir / "hidden_last.safetensors"))["hidden_last"]
        assert tuple(hidden_last.shape) == (1, 2, 3)

        concept_vectors = load_file(str(out_dir / "calibration_artifacts" / "concept_vectors.safetensors"))
        assert set(concept_vectors) == {"toxic", "jailbreak__gcg"}


def test_bitsandbytes_quantization_config():
    assert _bitsandbytes_config({"quantization": "none"}) is None

    int8_config = _bitsandbytes_config({"quantization": "8bit"})
    assert int8_config.load_in_8bit is True
    assert int8_config.load_in_4bit is False

    int4_config = _bitsandbytes_config(
        {
            "quantization": "4bit",
            "compute_dtype": "bfloat16",
            "quant_type": "nf4",
            "double_quant": True,
        }
    )
    assert int4_config.load_in_4bit is True
    assert int4_config.bnb_4bit_quant_type == "nf4"
    assert int4_config.bnb_4bit_use_double_quant is True


def test_model_input_normalization_accepts_tensor_and_batch_encoding_like():
    tensor_inputs = _to_model_inputs(torch.tensor([[1, 2, 3]]), torch.device("cpu"))
    assert set(tensor_inputs) == {"input_ids"}
    assert tuple(tensor_inputs["input_ids"].shape) == (1, 3)

    batch_inputs = BatchEncodingTensorLike(torch.tensor([[4, 5]]))
    normalized = _to_model_inputs(batch_inputs, torch.device("cpu"))
    assert normalized is batch_inputs
    assert tuple(normalized["input_ids"].shape) == (1, 2)


def test_interpret_difference_matrix_accepts_single_embedding_vectors():
    first = torch.arange(3584, dtype=torch.float32)
    second = torch.zeros(3584, dtype=torch.float32)
    vector, delta = interpret_difference_matrix(
        None,
        None,
        first,
        second,
        return_tokens=False,
    )
    assert tuple(vector.shape) == (3584,)
    assert delta.ndim == 0

    matrix_rows = _as_embedding_list(torch.stack([first, second]))
    assert len(matrix_rows) == 2
    assert tuple(matrix_rows[0].shape) == (3584,)


def test_collect_hidden_summaries_avoids_full_hidden_state_return():
    model = FakeSummaryModel()
    model_inputs = {"input_ids": torch.tensor([[1, 2, 3, 4]])}
    summary = collect_hidden_summaries(
        model,
        model_inputs,
        span=(1, 3),
        last_k=2,
        return_tail=True,
        return_spans=True,
        return_audit=True,
    )
    assert len(summary["last"]) == 3
    assert len(summary["tail"]) == 3
    assert len(summary["span"]) == 3
    assert tuple(summary["last"][0].shape) == (4,)
    assert tuple(summary["tail"][0].shape) == (2, 4)
    assert tuple(summary["span"][0].shape) == (4,)
    assert summary["audit"]["audit_scope"] == "captured_hidden_summaries"


def test_gemma3_style_text_config_and_language_backbone_are_supported():
    model = FakeGemma3SummaryModel()
    assert get_model_num_hidden_layers(model) == 2
    assert get_model_hidden_size(model) == 4
    assert get_model_vocab_size(model) == 16

    summary = collect_hidden_summaries(
        model,
        {"input_ids": torch.tensor([[1, 2, 3, 4]])},
        return_tail=True,
        return_spans=True,
        span=(1, 3),
    )
    assert len(summary["last"]) == 3
    assert tuple(summary["last"][0].shape) == (4,)
    assert tuple(summary["span"][0].shape) == (4,)


if __name__ == "__main__":
    test_span_resolution_success_and_failures()
    test_span_resolution_accepts_batch_encoding_like_tokenizer_output()
    test_phase2_writer_outputs_expected_schema()
    test_bitsandbytes_quantization_config()
    test_model_input_normalization_accepts_tensor_and_batch_encoding_like()
    test_interpret_difference_matrix_accepts_single_embedding_vectors()
    test_collect_hidden_summaries_avoids_full_hidden_state_return()
    test_gemma3_style_text_config_and_language_backbone_are_supported()
