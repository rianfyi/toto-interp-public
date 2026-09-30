from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Mapping

import torch

from toto.model.backbone import TotoBackbone
from toto.model.toto import Toto
from .types import InterventionConfig, TokenPosition


def _resolve_backbone(model: Toto | TotoBackbone) -> TotoBackbone:
    if isinstance(model, TotoBackbone):
        return model
    if isinstance(model, Toto):
        return model.model
    raise TypeError(f"Unsupported model type for interpretability intervention: {type(model)!r}")


def _token_selector(
    seq_len: int,
    *,
    token_position: str,
    decode_step: int,
    decode_steps: tuple[int, ...] | None,
) -> slice | list[int] | None:
    if token_position == "all_context":
        return slice(None) if seq_len > 1 else None
    if token_position == "final_context":
        return [seq_len - 1] if seq_len > 1 else None
    if token_position == "first_decode":
        if seq_len != 1:
            return None
        if decode_steps is None or decode_step in decode_steps:
            return [0]
        return None
    raise ValueError(f"Unsupported token position: {token_position}")


def _ablate_direction(
    activations: torch.Tensor,
    vector: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    denom = vector.pow(2).sum().clamp_min(1e-12)
    projection = (activations * vector).sum(dim=-1, keepdim=True) / denom
    return activations - strength * projection * vector


def _steer_direction(
    activations: torch.Tensor,
    vector: torch.Tensor,
    *,
    strength: float,
    normalize_by_residual: bool,
) -> torch.Tensor:
    if normalize_by_residual:
        unit_vector = vector / vector.norm().clamp_min(1e-12)
        residual_norm = activations.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        delta = strength * residual_norm * unit_vector
    else:
        delta = strength * vector
    return activations + delta


@contextmanager
def apply_intervention(
    model: Toto | TotoBackbone,
    intervention_config: InterventionConfig,
) -> Iterator[None]:
    """
    Apply a residual-stream intervention to Toto transformer layer outputs.

    The context manager is intentionally narrow: it only edits the requested
    transformer-layer outputs and requested token positions, leaving Toto's
    default forecasting behavior unchanged outside the scope of the context.
    """

    backbone = _resolve_backbone(model)
    transformer_layers = backbone.transformer.layers
    if not transformer_layers:
        yield
        return

    state = {
        "decode_step": 0,
        "seq_len": None,
    }
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def pre_hook(_module: torch.nn.Module, inputs: tuple[object, ...]) -> None:
        hidden_states = inputs[1]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("Expected transformer layer hidden states as the second positional input.")
        seq_len = int(hidden_states.shape[2])
        state["seq_len"] = seq_len
        if seq_len == 1:
            state["decode_step"] += 1

    handles.append(transformer_layers[0].register_forward_pre_hook(pre_hook))

    def make_layer_hook(layer_idx: int):
        def layer_hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: torch.Tensor,
        ) -> torch.Tensor:
            if layer_idx not in intervention_config.layer_indices:
                return output

            seq_len = state["seq_len"]
            if seq_len is None:
                return output

            selected_tokens = _token_selector(
                seq_len,
                token_position=intervention_config.token_position,
                decode_step=int(state["decode_step"]),
                decode_steps=intervention_config.decode_steps,
            )
            if selected_tokens is None:
                return output

            modified = output.clone()
            vector = intervention_config.vector.to(device=output.device, dtype=output.dtype).view(1, 1, 1, -1)

            if isinstance(selected_tokens, slice):
                selected = modified[:, :, selected_tokens, :]
            else:
                selected = modified[:, :, selected_tokens, :]

            if intervention_config.mode == "ablate":
                updated = _ablate_direction(selected, vector, strength=intervention_config.strength)
            elif intervention_config.mode == "steer":
                updated = _steer_direction(
                    selected,
                    vector,
                    strength=intervention_config.strength,
                    normalize_by_residual=intervention_config.normalize_by_residual,
                )
            else:
                raise ValueError(f"Unsupported intervention mode: {intervention_config.mode}")

            if isinstance(selected_tokens, slice):
                modified[:, :, selected_tokens, :] = updated
            else:
                modified[:, :, selected_tokens, :] = updated
            return modified

        return layer_hook

    for layer_idx, layer in enumerate(transformer_layers):
        if layer_idx in intervention_config.layer_indices:
            handles.append(layer.register_forward_hook(make_layer_hook(layer_idx)))

    try:
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()


# ---------------------------------------------------------------------------
# Paired source/target patching (interchange intervention)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedPatchConfig:
    """
    Blend residuals captured from a source (donor) forward pass into the target
    forward pass's residual stream at a chosen (layer, token_position).

    Unlike :class:`InterventionConfig`, which steers along or ablates a
    direction vector, this is an *interchange intervention*: the source values
    come from a real forward pass on a real donor input, and
    :func:`apply_paired_patch` writes ``blend * source + (1 - blend) * target``.
    ``scripts/run_toto_paired_patch.py`` uses it for the Toto donor exchange.

    `source_residuals[layer_idx]` must have shape
    ``(1, num_variates, source_seq_len, hidden_dim)`` — i.e. the raw output of
    the matching transformer layer in the source pass.
    """

    layer_indices: tuple[int, ...]
    token_position: TokenPosition
    source_residuals: Mapping[int, torch.Tensor]
    decode_steps: tuple[int, ...] | None = None
    blend: float = 1.0  # 1.0 = full replace; 0.0 = no-op; (0,1) = interpolate


def capture_source_residuals(
    model: Toto | TotoBackbone,
    *,
    layer_indices: tuple[int, ...],
) -> dict[int, torch.Tensor]:
    """
    Placeholder that directs callers to :func:`capture_source`, which yields a
    dict of the captured residuals::

        with capture_source(model, layer_indices) as residuals:
            <run forward pass on source window>
    """
    raise NotImplementedError(
        "Use the `capture_source` context manager below."
    )


@contextmanager
def capture_source(
    model: Toto | TotoBackbone,
    layer_indices: tuple[int, ...],
) -> Iterator[dict[int, torch.Tensor]]:
    """
    Context manager that captures post-layer residuals at the given layer
    indices into a fresh dict. Usage::

        with capture_source(backbone, (11,)) as captured:
            forecaster.forecast(source_window, ...)
        # captured[11] now holds the layer-11 output from the source pass

    Captured tensors live on the model's device; convert/clone as needed.
    """
    backbone = _resolve_backbone(model)
    captured: dict[int, torch.Tensor] = {}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def make_hook(layer_idx: int):
        def hook(_module, _inputs, output):
            # Only keep the *first* call per layer (the context pass), not the
            # subsequent decode-step calls. The runner script controls this by
            # entering the context manager around the forecast call we care
            # about; the first call per layer is the pass over the full
            # context (seq_len > 1).
            if layer_idx in captured:
                return
            if not isinstance(output, torch.Tensor):
                return
            if output.dim() != 4:
                return
            # Normalize to single-sample batch. When `forecaster.forecast` is
            # called with sample expansion (samples_per_batch > 1), TOTO runs
            # the context pass with batch = samples_per_batch and the residual
            # stream carries that dim. Context passes over identical inputs
            # are deterministic across replicas, so slicing [0:1] is exact and
            # gives us the documented (1, V, T, H) shape that
            # `PairedPatchConfig` expects.
            captured[layer_idx] = output[0:1].detach().clone()

        return hook

    for idx, layer in enumerate(backbone.transformer.layers):
        if idx in layer_indices:
            handles.append(layer.register_forward_hook(make_hook(idx)))

    try:
        yield captured
    finally:
        for handle in reversed(handles):
            handle.remove()


@contextmanager
def apply_paired_patch(
    model: Toto | TotoBackbone,
    patch_config: PairedPatchConfig,
) -> Iterator[None]:
    """
    Interchange intervention: blend previously captured source residuals into
    the residual stream at chosen (layer, token_position) as
    ``blend * source + (1 - blend) * target``.

    Unlike :func:`apply_intervention`, the source values come from a real
    forward pass on a donor input rather than from a direction vector.

    Shape handling:
      - target output:  ``(B, V_t, T_t, H)``
      - source residual: ``(1, V_s, T_s, H)``
      - For matching variate counts the patch is direct.
      - For mismatched variates, the source is averaged over its variates and
        the mean is broadcast across the target variate dim. Callers that need
        per-variate correspondence should pass a source with the target's
        variate count.
    """
    backbone = _resolve_backbone(model)
    transformer_layers = backbone.transformer.layers
    if not transformer_layers:
        yield
        return

    state = {"decode_step": 0, "seq_len": None}
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def pre_hook(_module, inputs):
        hidden_states = inputs[1]
        if not isinstance(hidden_states, torch.Tensor):
            raise TypeError("Expected transformer layer hidden states as the second positional input.")
        seq_len = int(hidden_states.shape[2])
        state["seq_len"] = seq_len
        if seq_len == 1:
            state["decode_step"] += 1

    handles.append(transformer_layers[0].register_forward_pre_hook(pre_hook))

    def make_layer_hook(layer_idx: int):
        def layer_hook(_module, _inputs, output: torch.Tensor) -> torch.Tensor:
            if layer_idx not in patch_config.layer_indices:
                return output
            seq_len = state["seq_len"]
            if seq_len is None:
                return output
            if layer_idx not in patch_config.source_residuals:
                return output

            selected_tokens = _token_selector(
                seq_len,
                token_position=patch_config.token_position,
                decode_step=int(state["decode_step"]),
                decode_steps=patch_config.decode_steps,
            )
            if selected_tokens is None:
                return output

            source = patch_config.source_residuals[layer_idx].to(
                device=output.device, dtype=output.dtype
            )
            # source shape: (1, V_s, T_s, H); output shape: (B, V_t, T_t, H)
            # We always patch a contiguous slice along the seq axis; align by
            # taking the *same selector* on source. If source is shorter, we
            # broadcast/repeat. If longer, we truncate.
            if isinstance(selected_tokens, slice):
                tgt_slice = output[:, :, selected_tokens, :]
                # Take same span from source
                src_slice = source[:, :, : tgt_slice.shape[2], :]
                if src_slice.shape[2] < tgt_slice.shape[2]:
                    # pad-by-repeat-last
                    pad = tgt_slice.shape[2] - src_slice.shape[2]
                    src_slice = torch.cat(
                        [src_slice, src_slice[:, :, -1:, :].expand(-1, -1, pad, -1)],
                        dim=2,
                    )
            else:
                tgt_slice = output[:, :, selected_tokens, :]
                # For final_context / first_decode we patch one position; align to
                # source's *last* context-token residual.
                src_slice = source[:, :, -1:, :].expand(-1, -1, tgt_slice.shape[2], -1)

            # Variate alignment: broadcast source across target variates if mismatched
            if src_slice.shape[1] != tgt_slice.shape[1]:
                src_slice = src_slice.mean(dim=1, keepdim=True).expand(
                    -1, tgt_slice.shape[1], -1, -1
                )

            # Batch alignment. Two cases worth supporting cleanly:
            #   - target is a single deterministic forward (batch=1) but source
            #     was captured under sample-expansion (batch=samples_per_batch);
            #     reduce by mean (same input -> identical replicas).
            #   - target is sample-expanded (batch=N) and source is single
            #     (batch=1); broadcast source across the sample dim.
            if src_slice.shape[0] != tgt_slice.shape[0]:
                if src_slice.shape[0] > tgt_slice.shape[0]:
                    src_slice = src_slice.mean(dim=0, keepdim=True)
                src_slice = src_slice.expand(tgt_slice.shape[0], -1, -1, -1)

            blended = patch_config.blend * src_slice + (1.0 - patch_config.blend) * tgt_slice

            modified = output.clone()
            if isinstance(selected_tokens, slice):
                modified[:, :, selected_tokens, :] = blended
            else:
                modified[:, :, selected_tokens, :] = blended
            return modified

        return layer_hook

    for layer_idx, layer in enumerate(transformer_layers):
        if layer_idx in patch_config.layer_indices:
            handles.append(layer.register_forward_hook(make_layer_hook(layer_idx)))

    try:
        yield
    finally:
        for handle in reversed(handles):
            handle.remove()
