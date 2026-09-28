from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import torch

from toto.data.util.dataset import replace_extreme_values
from toto.inference.forecaster import TotoForecaster
from toto.model.backbone import TotoBackbone
from toto.model.toto import Toto
from .boom import raw_features_for_window
from .labels import RAW_FEATURE_NAMES
from .types import ActivationBatch, TraceConfig, WindowExample


def _resolve_backbone(model: Toto | TotoBackbone) -> TotoBackbone:
    if isinstance(model, TotoBackbone):
        return model
    if isinstance(model, Toto):
        return model.model
    raise TypeError(f"Unsupported model type for activation extraction: {type(model)!r}")


def _device_of(model: Toto | TotoBackbone) -> torch.device:
    if isinstance(model, Toto):
        return model.device
    return model.device


class _ActivationRecorder:
    def __init__(self, trace_config: TraceConfig):
        self.trace_config = trace_config
        self.phase = "context"
        self.records: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)

    def clear_phase(self, phase: str) -> None:
        self.records[phase] = {}

    def set_phase(self, phase: str) -> None:
        self.phase = phase

    def patch_hook(
        self,
        _module: torch.nn.Module,
        _inputs: tuple[object, ...],
        output: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        if not self.trace_config.capture_patch_embedding:
            return
        self.records[self.phase][-1] = output[0].detach().cpu()

    def layer_hook(self, layer_idx: int):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[object, ...],
            output: torch.Tensor,
        ) -> None:
            if layer_idx in self.trace_config.layers:
                self.records[self.phase][layer_idx] = output.detach().cpu()

        return hook


def _ensure_sequence(inputs: WindowExample | Sequence[WindowExample]) -> list[WindowExample]:
    if isinstance(inputs, WindowExample):
        return [inputs]
    return list(inputs)


def _build_masks(
    inputs: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    padding_mask = torch.ones_like(inputs, dtype=torch.bool, device=device)
    id_mask = torch.zeros_like(inputs, dtype=torch.long, device=device)
    return padding_mask, id_mask


def _mean_patch_prediction(
    backbone: TotoBackbone,
    *,
    inputs: torch.Tensor,
    padding_mask: torch.Tensor,
    id_mask: torch.Tensor,
    kv_cache: object | None,
    scaling_prefix_length: int,
) -> torch.Tensor:
    base_distr, loc, scale = backbone(
        inputs=inputs,
        input_padding_mask=padding_mask,
        id_mask=id_mask,
        kv_cache=kv_cache,
        scaling_prefix_length=scaling_prefix_length,
        num_exogenous_variables=0,
    )
    transformed = TotoForecaster.create_affine_transformed(base_distr, loc, scale)
    return replace_extreme_values(transformed.mean[:, :, -backbone.patch_embed.patch_size :])


def _run_context_backbone(
    backbone: TotoBackbone,
    *,
    inputs: torch.Tensor,
    padding_mask: torch.Tensor,
    id_mask: torch.Tensor,
    kv_cache: object | None,
    scaling_prefix_length: int,
) -> None:
    """Run the residual stream without constructing an unused output distribution."""
    backbone.backbone(
        inputs=inputs,
        input_padding_mask=padding_mask,
        id_mask=id_mask,
        kv_cache=kv_cache,
        scaling_prefix_length=scaling_prefix_length,
        num_exogenous_variables=0,
    )


def _append_decode_inputs(
    predicted_patch: torch.Tensor,
    *,
    context_inputs: torch.Tensor,
    context_padding_mask: torch.Tensor,
    context_id_mask: torch.Tensor,
    use_kv_cache: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    patch_size = int(predicted_patch.shape[-1])
    decode_padding = torch.ones_like(predicted_patch, dtype=torch.bool, device=predicted_patch.device)
    decode_id_mask = context_id_mask[:, :, -1:].expand(-1, -1, patch_size)
    return (
        torch.cat([context_inputs, predicted_patch], dim=-1),
        torch.cat([context_padding_mask, decode_padding], dim=-1),
        torch.cat([context_id_mask, decode_id_mask], dim=-1),
    )


def _add_records(
    *,
    storage: dict[str, list[object]],
    tensor: torch.Tensor,
    layer_idx: int,
    token_position: str,
    pooling_modes: tuple[str, ...],
    window: WindowExample,
    patch_index: int | None = None,
    patch_offset: int = 0,
    raw_features: torch.Tensor,
) -> None:
    batch, variates, seq_len, embed_dim = tensor.shape
    if batch != 1:
        raise ValueError("extract_activations currently expects single-example tracing.")

    data = tensor[0]

    if token_position == "all_context":
        patch_indices = list(range(patch_offset, patch_offset + seq_len))
        selected = data
    else:
        selected = data[:, -1:, :]
        patch_indices = [patch_index if patch_index is not None else patch_offset + seq_len - 1]

    if "per_variate" in pooling_modes:
        selected_per_variate = selected.permute(1, 0, 2).reshape(-1, embed_dim)
        repeated_patch_indices = [patch for patch in patch_indices for _ in range(variates)]
        repeated_variate_indices = [variate for _ in patch_indices for variate in range(variates)]
        for row_idx in range(selected_per_variate.shape[0]):
            storage["activations"].append(selected_per_variate[row_idx])
            storage["raw_features"].append(raw_features)
            storage["layer_indices"].append(layer_idx)
            storage["patch_indices"].append(repeated_patch_indices[row_idx])
            storage["variate_indices"].append(repeated_variate_indices[row_idx])
            storage["token_positions"].append(token_position)
            storage["pooling_modes"].append("per_variate")
            storage["series_ids"].append(window.series_id)
            storage["window_ids"].append(window.window_id)
            storage["splits"].append(window.split)
            for label_name, label_value in window.labels.items():
                storage["labels"][label_name].append(label_value)

    if "series_mean" in pooling_modes:
        selected_series = selected.mean(dim=0)
        for row_idx in range(selected_series.shape[0]):
            storage["activations"].append(selected_series[row_idx])
            storage["raw_features"].append(raw_features)
            storage["layer_indices"].append(layer_idx)
            storage["patch_indices"].append(patch_indices[row_idx])
            storage["variate_indices"].append(-1)
            storage["token_positions"].append(token_position)
            storage["pooling_modes"].append("series_mean")
            storage["series_ids"].append(window.series_id)
            storage["window_ids"].append(window.window_id)
            storage["splits"].append(window.split)
            for label_name, label_value in window.labels.items():
                storage["labels"][label_name].append(label_value)


@torch.no_grad()
def extract_activations(
    model: Toto | TotoBackbone,
    inputs: WindowExample | Sequence[WindowExample],
    trace_config: TraceConfig,
) -> ActivationBatch:
    """
    Extract patch-token activations from Toto's residual stream.

    `inputs` is expected to be a `WindowExample` or a sequence of them. The
    implementation traces one window at a time to keep memory bounded and to
    keep the decode-step accounting deterministic across runs.
    """

    backbone = _resolve_backbone(model)
    device = _device_of(model)
    windows = _ensure_sequence(inputs)

    recorder = _ActivationRecorder(trace_config)
    storage: dict[str, list[object]] = {
        "activations": [],
        "raw_features": [],
        "layer_indices": [],
        "patch_indices": [],
        "variate_indices": [],
        "token_positions": [],
        "pooling_modes": [],
        "series_ids": [],
        "window_ids": [],
        "splits": [],
        "labels": defaultdict(list),
    }

    handles: list[torch.utils.hooks.RemovableHandle] = []
    if trace_config.capture_patch_embedding:
        handles.append(backbone.patch_embed.register_forward_hook(recorder.patch_hook))
    for layer_idx, layer in enumerate(backbone.transformer.layers):
        if layer_idx in trace_config.layers:
            handles.append(layer.register_forward_hook(recorder.layer_hook(layer_idx)))

    try:
        for window in windows:
            context_inputs = window.context.unsqueeze(0).to(device=device, dtype=torch.float32)
            context_padding_mask, context_id_mask = _build_masks(context_inputs, device=device)
            patch_size = int(window.patch_size)
            context_length = int(context_inputs.shape[-1])
            context_patch_count = context_length // patch_size
            raw_features = raw_features_for_window(window).cpu()

            kv_cache = None
            if trace_config.use_kv_cache:
                kv_cache = backbone.allocate_kv_cache(
                    batch_size=1,
                    num_variates=int(context_inputs.shape[1]),
                    max_time_steps=context_length + patch_size,
                    device=device,
                    dtype=context_inputs.dtype,
                )

            recorder.clear_phase("context")
            recorder.set_phase("context")
            if "first_decode" in trace_config.token_positions:
                predicted_patch = _mean_patch_prediction(
                    backbone,
                    inputs=context_inputs,
                    padding_mask=context_padding_mask,
                    id_mask=context_id_mask,
                    kv_cache=kv_cache,
                    scaling_prefix_length=context_length,
                )
            else:
                _run_context_backbone(
                    backbone,
                    inputs=context_inputs,
                    padding_mask=context_padding_mask,
                    id_mask=context_id_mask,
                    kv_cache=kv_cache,
                    scaling_prefix_length=context_length,
                )
                predicted_patch = None

            for layer_idx, tensor in recorder.records["context"].items():
                if "all_context" in trace_config.token_positions:
                    _add_records(
                        storage=storage,
                        tensor=tensor,
                        layer_idx=layer_idx,
                        token_position="all_context",
                        pooling_modes=trace_config.pooling_modes,
                        window=window,
                        patch_offset=0,
                        raw_features=raw_features,
                    )
                if "final_context" in trace_config.token_positions:
                    _add_records(
                        storage=storage,
                        tensor=tensor,
                        layer_idx=layer_idx,
                        token_position="final_context",
                        pooling_modes=trace_config.pooling_modes,
                        window=window,
                        patch_index=context_patch_count - 1,
                        raw_features=raw_features,
                    )

            if "first_decode" not in trace_config.token_positions:
                continue

            assert predicted_patch is not None
            decode_inputs, decode_padding_mask, decode_id_mask = _append_decode_inputs(
                predicted_patch,
                context_inputs=context_inputs,
                context_padding_mask=context_padding_mask,
                context_id_mask=context_id_mask,
                use_kv_cache=trace_config.use_kv_cache,
            )

            recorder.clear_phase("decode")
            recorder.set_phase("decode")
            _mean_patch_prediction(
                backbone,
                inputs=decode_inputs,
                padding_mask=decode_padding_mask,
                id_mask=decode_id_mask,
                kv_cache=kv_cache,
                scaling_prefix_length=context_length,
            )

            for layer_idx, tensor in recorder.records["decode"].items():
                _add_records(
                    storage=storage,
                    tensor=tensor,
                    layer_idx=layer_idx,
                    token_position="first_decode",
                    pooling_modes=trace_config.pooling_modes,
                    window=window,
                    patch_index=context_patch_count,
                    raw_features=raw_features,
                )
    finally:
        for handle in reversed(handles):
            handle.remove()

    if not storage["activations"]:
        raise ValueError("No activations were captured. Check the trace configuration and input windows.")

    labels = {name: list(values) for name, values in storage["labels"].items()}
    return ActivationBatch(
        activations=torch.stack(storage["activations"]).to(torch.float32),
        raw_features=torch.stack(storage["raw_features"]).to(torch.float32),
        raw_feature_names=RAW_FEATURE_NAMES,
        layer_indices=torch.tensor(storage["layer_indices"], dtype=torch.long),
        patch_indices=torch.tensor(storage["patch_indices"], dtype=torch.long),
        variate_indices=torch.tensor(storage["variate_indices"], dtype=torch.long),
        token_positions=list(storage["token_positions"]),
        pooling_modes=list(storage["pooling_modes"]),
        series_ids=list(storage["series_ids"]),
        window_ids=list(storage["window_ids"]),
        splits=list(storage["splits"]),
        labels=labels,
    )
