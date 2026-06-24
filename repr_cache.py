"""
Hidden-state representation cache for JBShield.

Persists per-prompt hidden states (last token and trailing last-k tokens) so that
downstream methods (RTV, Residual Stream Activation Analysis, CKA/RSA, ...) can be
computed from a shared cache without re-running model forwards.

Layout (procedure section 7.1):

    representations/{model_id_sanitized}/{split}/
        hidden_last.safetensors   # [N, L+1, d]
        hidden_tail.safetensors   # [N, L+1, k, d]  (only when tails are provided)
        metadata.parquet          # one row per prompt
"""

from pathlib import Path

import pandas as pd
import torch
from safetensors.torch import save_file

EXTRACTION_VERSION = "jbshield-hidden-cache-v1"


def sanitize_model_id(model_id):
    return str(model_id).replace("/", "__")


def stack_layer_major(layer_major):
    """Stack a layer-major list (len L+1, each a list of N per-prompt tensors)
    into a single tensor of shape [N, L+1, ...]."""
    # [L+1, N, ...] -> [N, L+1, ...]
    per_layer = [torch.stack(prompts, dim=0) for prompts in layer_major]
    stacked = torch.stack(per_layer, dim=0)
    return stacked.transpose(0, 1).contiguous()


def write_hidden_cache(
    cache_dir,
    model_id,
    split,
    embeddings_last_token,
    metadata_rows,
    tail_embeddings=None,
    extraction_version=EXTRACTION_VERSION,
):
    """Write hidden-state tensors and metadata for one split.

    Args:
    - cache_dir: root directory (e.g. "representations")
    - model_id: HF model id; sanitized for the directory name
    - split: split / group name (e.g. "calibration_harmless", "test_gcg")
    - embeddings_last_token: layer-major list, each [d] per prompt
    - metadata_rows: list of dicts, one per prompt (must align with prompt order)
    - tail_embeddings: optional layer-major list, each [k, d] per prompt

    Returns the split output directory as a Path.
    """
    split_dir = Path(cache_dir) / sanitize_model_id(model_id) / str(split)
    split_dir.mkdir(parents=True, exist_ok=True)

    hidden_last = stack_layer_major(embeddings_last_token)  # [N, L+1, d]
    last_path = split_dir / "hidden_last.safetensors"
    save_file({"hidden_last": hidden_last}, str(last_path))

    tail_path = None
    if tail_embeddings is not None:
        hidden_tail = stack_layer_major(tail_embeddings)  # [N, L+1, k, d]
        tail_path = split_dir / "hidden_tail.safetensors"
        save_file({"hidden_tail": hidden_tail}, str(tail_path))

    num_layers = hidden_last.shape[1]
    hidden_size = hidden_last.shape[-1]
    dtype = str(hidden_last.dtype)
    enriched = []
    for row in metadata_rows:
        enriched.append({
            **row,
            "model_id": model_id,
            "split": split,
            "num_layers": int(num_layers),
            "hidden_size": int(hidden_size),
            "dtype": dtype,
            "representation_type": "hidden_state_last_token",
            "tensor_path": str(last_path),
            "tail_tensor_path": str(tail_path) if tail_path is not None else None,
            "extraction_version": extraction_version,
        })
    df = pd.DataFrame(enriched)
    df.to_parquet(split_dir / "metadata.parquet", index=False)

    return split_dir
