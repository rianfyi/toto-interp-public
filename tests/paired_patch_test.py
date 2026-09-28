"""
Unit tests for the on-manifold paired-patch primitive.

These exercise the hook mechanics on a tiny mock Toto and don't require
network/GPU. They're intentionally narrow: they verify the patch is applied,
that random patches differ from real patches, and that the no-op blend (0.0)
leaves the forward pass unchanged.
"""
from __future__ import annotations

import torch

from toto_interp import (
    PairedPatchConfig,
    apply_paired_patch,
    capture_source,
)

from .test_helpers import make_tiny_toto, make_window_example


def _context_inputs(window):
    inputs = window.context.unsqueeze(0)
    padding_mask = torch.ones_like(inputs, dtype=torch.bool)
    id_mask = torch.zeros_like(inputs, dtype=torch.long)
    return inputs, padding_mask, id_mask


def _forward(backbone, window):
    inputs, padding_mask, id_mask = _context_inputs(window)
    backbone(
        inputs=inputs,
        input_padding_mask=padding_mask,
        id_mask=id_mask,
        scaling_prefix_length=inputs.shape[-1],
    )


def _capture_layer(backbone, window, layer_idx: int) -> torch.Tensor:
    captured: dict[int, torch.Tensor] = {}

    def hook(_module, _args, output, layer_idx=layer_idx):
        captured.setdefault(layer_idx, output.detach().clone())

    handle = backbone.transformer.layers[layer_idx].register_forward_hook(hook)
    try:
        _forward(backbone, window)
    finally:
        handle.remove()
    return captured[layer_idx]


def test_capture_source_grabs_residual():
    model = make_tiny_toto()
    backbone = model.model
    window = make_window_example()

    with capture_source(backbone, (1,)) as captured:
        _forward(backbone, window)

    assert 1 in captured
    # Shape should be (1, num_variates, num_patches, embed_dim)
    assert captured[1].dim() == 4
    assert captured[1].shape[0] == 1
    assert captured[1].shape[1] == window.context.shape[0]


def test_paired_patch_with_full_blend_replaces_residual():
    model = make_tiny_toto()
    backbone = model.model
    window = make_window_example()

    # Capture from a *different* synthetic source. Note: TOTO standardizes
    # inputs (StdMeanScaler), so a pure additive/multiplicative shift would be
    # normalized away. We replace the context with a different *pattern* so
    # the post-scaler residual differs meaningfully from the target's.
    source_window = make_window_example()
    torch.manual_seed(0)
    source_window.context.copy_(torch.randn_like(source_window.context) * 2.0)

    with capture_source(backbone, (1,)) as src_captured:
        _forward(backbone, source_window)
    src_residual = src_captured[1]

    # Without patch: capture the natural layer-1 output for the target
    natural = _capture_layer(backbone, window, layer_idx=1)

    # With patch (full replace): layer-1 output should match src_residual on
    # the patched token positions (here, all_context).
    cfg = PairedPatchConfig(
        layer_indices=(1,),
        token_position="all_context",
        source_residuals={1: src_residual},
        blend=1.0,
    )

    patched_capture: dict[int, torch.Tensor] = {}

    def hook(_module, _args, output):
        patched_capture.setdefault(1, output.detach().clone())

    # Register the capture hook *inside* the patch context so it fires after
    # the patch hook (PyTorch fires forward hooks in registration order).
    with apply_paired_patch(backbone, cfg):
        handle = backbone.transformer.layers[1].register_forward_hook(hook)
        try:
            _forward(backbone, window)
        finally:
            handle.remove()

    patched = patched_capture[1]
    # Should differ from natural (because we replaced the residual)
    assert not torch.allclose(patched, natural, atol=1e-5)
    # Should be (close to) src_residual on the patched positions
    # We use atol that's generous since dtype/device transit can introduce drift
    assert torch.allclose(patched, src_residual, atol=1e-4)


def test_paired_patch_with_zero_blend_is_noop():
    model = make_tiny_toto()
    backbone = model.model
    window = make_window_example()
    source_window = make_window_example()
    torch.manual_seed(1)
    source_window.context.copy_(torch.randn_like(source_window.context) * 2.0)

    with capture_source(backbone, (0,)) as src_captured:
        _forward(backbone, source_window)
    src_residual = src_captured[0]

    natural = _capture_layer(backbone, window, layer_idx=0)

    cfg = PairedPatchConfig(
        layer_indices=(0,),
        token_position="all_context",
        source_residuals={0: src_residual},
        blend=0.0,  # no-op
    )

    patched_capture: dict[int, torch.Tensor] = {}

    def hook(_module, _args, output):
        patched_capture.setdefault(0, output.detach().clone())

    with apply_paired_patch(backbone, cfg):
        handle = backbone.transformer.layers[0].register_forward_hook(hook)
        try:
            _forward(backbone, window)
        finally:
            handle.remove()

    patched = patched_capture[0]
    assert torch.allclose(patched, natural, atol=1e-5)


def test_paired_patch_aligns_mismatched_batch_dims():
    """
    Regression: when capture_source runs inside a sample-expanded forecast,
    the captured residual carries the samples_per_batch dim. The patch hook
    must still apply cleanly to a single-batch target forward (e.g. the one
    extract_activations does), without raising on shape mismatch.
    """
    model = make_tiny_toto()
    backbone = model.model
    window = make_window_example()

    # Simulate a sample-expanded source residual: capture once at batch=1, then
    # synthetically expand it to batch=4 so the patch hook sees the same shape
    # mismatch the runner script triggers in production.
    with capture_source(backbone, (1,)) as src_captured:
        _forward(backbone, window)
    src_residual_b1 = src_captured[1]
    assert src_residual_b1.shape[0] == 1, "capture_source should normalize to batch=1"

    src_residual_b4 = src_residual_b1.expand(4, -1, -1, -1).contiguous()

    cfg = PairedPatchConfig(
        layer_indices=(1,),
        token_position="all_context",
        source_residuals={1: src_residual_b4},
        blend=1.0,
    )

    # Apply the patch around a single-batch target forward. With the alignment
    # fix this should not raise; without it, this raised
    # "expanded size of the tensor (1) must match the existing size (4) ..."
    with apply_paired_patch(backbone, cfg):
        _forward(backbone, window)


def test_paired_patch_only_touches_requested_layers():
    model = make_tiny_toto()
    backbone = model.model
    window = make_window_example()
    source_window = make_window_example()
    torch.manual_seed(2)
    source_window.context.copy_(torch.randn_like(source_window.context) * 2.0)

    # Capture src at layer 2
    with capture_source(backbone, (2,)) as src_captured:
        _forward(backbone, source_window)
    src_residual = src_captured[2]

    natural_layer0 = _capture_layer(backbone, window, layer_idx=0)
    natural_layer1 = _capture_layer(backbone, window, layer_idx=1)

    cfg = PairedPatchConfig(
        layer_indices=(2,),
        token_position="all_context",
        source_residuals={2: src_residual},
        blend=1.0,
    )

    cap0: dict[int, torch.Tensor] = {}
    cap1: dict[int, torch.Tensor] = {}

    h0 = backbone.transformer.layers[0].register_forward_hook(
        lambda _m, _a, out: cap0.setdefault(0, out.detach().clone())
    )
    h1 = backbone.transformer.layers[1].register_forward_hook(
        lambda _m, _a, out: cap1.setdefault(1, out.detach().clone())
    )
    try:
        with apply_paired_patch(backbone, cfg):
            _forward(backbone, window)
    finally:
        h0.remove()
        h1.remove()

    # Layer 0 and 1 should be unchanged (we only patched layer 2)
    assert torch.allclose(cap0[0], natural_layer0, atol=1e-5)
    assert torch.allclose(cap1[1], natural_layer1, atol=1e-5)
