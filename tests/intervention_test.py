from __future__ import annotations

from contextlib import nullcontext

import torch

from toto_interp import InterventionConfig, apply_intervention
from toto.inference.forecaster import TotoForecaster

from .test_helpers import make_tiny_toto, make_window_example


def _context_inputs(window):
    inputs = window.context.unsqueeze(0)
    padding_mask = torch.ones_like(inputs, dtype=torch.bool)
    id_mask = torch.zeros_like(inputs, dtype=torch.long)
    return inputs, padding_mask, id_mask


def _capture_layer_outputs(backbone, window, *, intervention=None):
    inputs, padding_mask, id_mask = _context_inputs(window)
    captured = {}
    manager = apply_intervention(backbone, intervention) if intervention is not None else nullcontext()
    with manager:
        handles = []
        for layer_idx in (0, 2):
            handles.append(
                backbone.transformer.layers[layer_idx].register_forward_hook(
                    lambda _module, _args, output, layer_idx=layer_idx: captured.setdefault(
                        layer_idx, output.detach().clone()
                    )
                )
            )
        backbone(
            inputs=inputs,
            input_padding_mask=padding_mask,
            id_mask=id_mask,
            scaling_prefix_length=inputs.shape[-1],
        )
        for handle in handles:
            handle.remove()
    return captured


def _capture_context_and_decode(backbone, window, *, intervention=None):
    inputs, padding_mask, id_mask = _context_inputs(window)
    kv_cache = backbone.allocate_kv_cache(
        batch_size=1,
        num_variates=window.context.shape[0],
        max_time_steps=window.context.shape[-1] + window.patch_size,
        device=inputs.device,
        dtype=inputs.dtype,
    )

    captured = {0: [], 2: []}
    manager = apply_intervention(backbone, intervention) if intervention is not None else nullcontext()
    with manager:
        handles = []
        for layer_idx in (0, 2):
            handles.append(
                backbone.transformer.layers[layer_idx].register_forward_hook(
                    lambda _module, _args, output, layer_idx=layer_idx: captured[layer_idx].append(output.detach().clone())
                )
            )
        base_distr, loc, scale = backbone(
            inputs=inputs,
            input_padding_mask=padding_mask,
            id_mask=id_mask,
            kv_cache=kv_cache,
            scaling_prefix_length=inputs.shape[-1],
        )
        mean_patch = TotoForecaster.create_affine_transformed(base_distr, loc, scale).mean[:, :, -window.patch_size :]
        decode_padding = torch.ones_like(mean_patch, dtype=torch.bool)
        decode_id_mask = id_mask[:, :, -1:].expand(-1, -1, window.patch_size)
        backbone(
            inputs=torch.cat([inputs, mean_patch], dim=-1),
            input_padding_mask=torch.cat([padding_mask, decode_padding], dim=-1),
            id_mask=torch.cat([id_mask, decode_id_mask], dim=-1),
            kv_cache=kv_cache,
            scaling_prefix_length=inputs.shape[-1],
        )
        for handle in handles:
            handle.remove()
    return captured


def test_apply_intervention_only_edits_requested_layer_and_token_position():
    model = make_tiny_toto()
    backbone = model.model
    window = make_window_example()
    vector = torch.randn(backbone.embed_dim)

    baseline = _capture_layer_outputs(backbone, window)
    steered = _capture_layer_outputs(
        backbone,
        window,
        intervention=InterventionConfig(
            layer_indices=(2,),
            token_position="final_context",
            mode="steer",
            vector=vector,
            strength=0.5,
        ),
    )

    assert torch.allclose(baseline[0], steered[0])
    assert torch.allclose(baseline[2][:, :, :-1, :], steered[2][:, :, :-1, :])
    assert not torch.allclose(baseline[2][:, :, -1, :], steered[2][:, :, -1, :])


def test_first_decode_intervention_only_applies_on_decode_step():
    model = make_tiny_toto()
    backbone = model.model
    window = make_window_example()
    vector = torch.randn(backbone.embed_dim)

    baseline = _capture_context_and_decode(backbone, window)
    steered = _capture_context_and_decode(
        backbone,
        window,
        intervention=InterventionConfig(
            layer_indices=(2,),
            token_position="first_decode",
            mode="steer",
            vector=vector,
            strength=0.5,
            decode_steps=(1,),
        ),
    )

    assert torch.allclose(baseline[2][0], steered[2][0])
    assert not torch.allclose(baseline[2][1], steered[2][1])
