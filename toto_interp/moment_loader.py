"""
MOMENT (CMU/AutonLab) backbone loader for the cross-model probing comparison.

This is a thin adapter to run the structural-concept probing
protocol on a *second* time-series foundation model. We use MOMENT-base (~125M
params, T5-encoder) as the comparison model.

Install (via the official package):

    pip install --no-deps momentfm==0.1.4

The model is downloaded on first call from
https://huggingface.co/AutonLab/MOMENT-1-base.

Why MOMENT for the comparison:
  - Different architecture family from TOTO (encoder-only T5 vs. decoder-only)
  - Different patch tokenization (patch_size=8 vs. TOTO's 64)
  - Different pretraining objective (masked patch reconstruction vs. TOTO's
    autoregressive forecasting)
  - Therefore, probing MOMENT's residual stream for the same 4 structural
    concepts (metric_type, domain, frequency_bucket, cardinality_bucket)
    tests whether the TOTO readouts recur in a second model.
"""
from __future__ import annotations

import logging
import os
from typing import Literal

import torch

logger = logging.getLogger(__name__)

WeightSource = Literal["pretrained", "random_init"]
MomentTask = Literal["embedding", "reconstruction"]
MOMENT_BASE_ID = "AutonLab/MOMENT-1-base"
MOMENT_SMALL_ID = "AutonLab/MOMENT-1-small"
MOMENT_LARGE_ID = "AutonLab/MOMENT-1-large"


def _import_moment():
    try:
        from momentfm import MOMENTPipeline  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "MOMENT is not installed. Install with:\n\n"
            "    pip install --no-deps momentfm==0.1.4\n\n"
            "(The MOMENT-base checkpoint, ~125M params, will download on first use.)"
        ) from exc
    from momentfm import MOMENTPipeline
    return MOMENTPipeline


def load_moment_with_fallback(
    model_id: str = MOMENT_BASE_ID,
    *,
    device: str = "cpu",
    weight_source: WeightSource = "pretrained",
    task_name: MomentTask = "embedding",
    seq_len: int = 512,
    dtype: torch.dtype = torch.float32,
):
    """
    Load a MOMENT model in embedding or reconstruction mode.

    `seq_len` is the maximum context length MOMENT will accept (must be a
    multiple of patch_size=8). MOMENT-1-base supports up to 512.

    Returns a `MOMENTPipeline` instance ready for forward passes. The
    reconstruction task retains MOMENT's pretrained reconstruction head while
    still allowing callers to invoke ``pipeline.embed(...)`` explicitly. This
    is used by the matched-interchange runner to share one frozen encoder
    between the probe check and the future-MAE forecast endpoint computed with
    MOMENT's official short-forecast head.

    `weight_source="random_init"` mirrors the TOTO loader's random-control:
    it loads the architecture and calls ``reset_parameters()`` on every
    submodule that defines it.
    Useful as a control arm separating trained weights from architecture
    alone.
    """
    MOMENTPipeline = _import_moment()

    logger.info(
        "Loading MOMENT model %s (weight_source=%s, task_name=%s, seq_len=%d, device=%s)",
        model_id,
        weight_source,
        task_name,
        seq_len,
        device,
    )

    pipeline = MOMENTPipeline.from_pretrained(
        model_id,
        model_kwargs={
            "task_name": task_name,
            "seq_len": seq_len,
            # `freeze_encoder` is irrelevant for inference but kept for clarity
            "freeze_encoder": True,
            "freeze_embedder": True,
            "enable_gradient_checkpointing": False,
        },
    )
    pipeline.init()

    if weight_source == "random_init":
        logger.info("Re-initializing MOMENT modules via reset_parameters (random-init control)")
        for m in pipeline.modules():
            if hasattr(m, "reset_parameters"):
                m.reset_parameters()

    pipeline = pipeline.to(device=device, dtype=dtype)
    pipeline.eval()
    return pipeline


def moment_patch_size(pipeline) -> int:
    """MOMENT uses fixed patch_size=8 across all variants."""
    cfg = getattr(pipeline, "config", None)
    if cfg is not None and hasattr(cfg, "patch_len"):
        return int(cfg.patch_len)
    # Fallback: introspect the patch_embedding module
    try:
        return int(pipeline.normalizer.patch_len)
    except AttributeError:
        return 8


def moment_hidden_dim(pipeline) -> int:
    """Returns the encoder hidden dimension (768 for base, 1024 for large)."""
    enc = pipeline.encoder
    # T5 encoder: pipeline.encoder.encoder.config.d_model
    inner = getattr(enc, "encoder", enc)
    cfg = getattr(inner, "config", None)
    if cfg is not None and hasattr(cfg, "d_model"):
        return int(cfg.d_model)
    raise RuntimeError("Could not introspect MOMENT hidden dim.")


def moment_num_layers(pipeline) -> int:
    enc = pipeline.encoder
    inner = getattr(enc, "encoder", enc)
    blocks = getattr(inner, "block", None)
    if blocks is None:
        raise RuntimeError("Could not introspect MOMENT layer blocks.")
    return len(blocks)


def moment_layer(pipeline, idx: int):
    """Return the i-th T5 encoder block for hook installation."""
    enc = pipeline.encoder
    inner = getattr(enc, "encoder", enc)
    return inner.block[idx]
