from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Literal

import torch

from .moment_loader import moment_layer
from .types import ProbeArtifact

MomentTokenPosition = Literal["all_context", "final_context"]


def _hidden_state(output: object) -> torch.Tensor:
    """Return the hidden-state tensor from a Hugging Face T5 block output."""

    hidden = output[0] if isinstance(output, (tuple, list)) else output
    if not isinstance(hidden, torch.Tensor):
        raise TypeError(
            "Expected the MOMENT encoder block to return a tensor or a tuple/list "
            "whose first element is a tensor."
        )
    if hidden.ndim != 3:
        raise ValueError(
            "Expected a MOMENT block hidden state with shape "
            f"(batch*channels, patches, hidden), received {tuple(hidden.shape)}."
        )
    return hidden


def _replace_hidden_state(output: object, hidden: torch.Tensor) -> object:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    if isinstance(output, list):
        return [hidden, *output[1:]]
    return hidden


@dataclass(frozen=True)
class MomentInterchangeConfig:
    """A single-site MOMENT encoder residual interchange."""

    layer: int
    token_position: MomentTokenPosition
    source_residual: torch.Tensor
    blend: float

    def __post_init__(self) -> None:
        if self.token_position not in ("all_context", "final_context"):
            raise ValueError(
                "MOMENT interchange supports only all_context and final_context "
                f"views, received {self.token_position!r}."
            )
        if not 0.0 <= float(self.blend) <= 1.0:
            raise ValueError(f"blend must lie in [0, 1], received {self.blend}.")
        _hidden_state(self.source_residual)


@contextmanager
def capture_moment_residual(
    pipeline,
    *,
    layer: int,
) -> Iterator[dict[str, torch.Tensor]]:
    """
    Capture the first post-block hidden state from one MOMENT forward pass.

    MOMENT flattens batch and channel into the leading dimension before its T5
    encoder, so the captured shape is ``(batch*channels, patches, hidden)``.
    Callers deliberately retain that exact shape and only pair windows with the
    same channel count; the intervention primitive never silently broadcasts a
    donor across unmatched channels.
    """

    captured: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        if "residual" not in captured:
            captured["residual"] = _hidden_state(output).detach().clone()

    handle = moment_layer(pipeline, layer).register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


@contextmanager
def apply_moment_interchange(
    pipeline,
    config: MomentInterchangeConfig,
) -> Iterator[None]:
    """
    Blend a real donor residual into the matching MOMENT block output.

    Source and target hidden states must agree exactly in channel, patch, and
    hidden dimensions, so each blended value pairs the target's residual with
    the donor residual at the same channel and patch position.
    """

    source = _hidden_state(config.source_residual)

    def hook(_module, _inputs, output):
        target = _hidden_state(output)
        if tuple(target.shape) != tuple(source.shape):
            raise ValueError(
                "MOMENT source and target residual shapes must match exactly; "
                f"source={tuple(source.shape)}, target={tuple(target.shape)}."
            )

        aligned_source = source.to(device=target.device, dtype=target.dtype)
        modified = target.clone()
        if config.token_position == "all_context":
            target_slice = target
            source_slice = aligned_source
            modified = (1.0 - config.blend) * target_slice + config.blend * source_slice
        else:
            modified[:, -1:, :] = (
                (1.0 - config.blend) * target[:, -1:, :]
                + config.blend * aligned_source[:, -1:, :]
            )
        return _replace_hidden_state(output, modified)

    handle = moment_layer(pipeline, config.layer).register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def pool_moment_residual(
    residual: torch.Tensor,
    *,
    num_variates: int,
    token_position: MomentTokenPosition,
    pooling_mode: str,
) -> torch.Tensor:
    """Pool a captured MOMENT block output exactly as the activation dumper."""

    hidden = _hidden_state(residual)
    if num_variates <= 0:
        raise ValueError(f"num_variates must be positive, received {num_variates}.")
    if hidden.shape[0] != num_variates:
        raise ValueError(
            "Expected a single-window MOMENT residual whose leading dimension "
            f"equals num_variates={num_variates}, received {tuple(hidden.shape)}."
        )

    data = hidden
    if token_position == "final_context":
        data = data[:, -1:, :]
    elif token_position != "all_context":
        raise ValueError(f"Unsupported MOMENT token position: {token_position!r}.")

    if pooling_mode == "series_mean":
        return data.mean(dim=0)
    if pooling_mode == "per_variate":
        return data.permute(1, 0, 2).reshape(-1, data.shape[-1])
    raise ValueError(f"Unsupported MOMENT pooling mode: {pooling_mode!r}.")


def score_moment_probe(
    residual: torch.Tensor,
    *,
    num_variates: int,
    probe: ProbeArtifact,
) -> tuple[float, float, int]:
    """
    Return mean, standard deviation, and record count of a continuous probe.

    The mean is the predeclared scalar probe score that the probe check
    compares between the real-donor and matched-null interchanges. Retaining
    the within-window dispersion and record count makes the reduction
    auditable for both ``series_mean`` and ``per_variate`` probe views.
    """

    if probe.label_spec.task_type != "continuous":
        raise ValueError("MOMENT future-burstiness interchange requires a continuous probe.")
    if probe.method != "linear_probe":
        raise ValueError("MOMENT interchange requires an activation-backed linear probe.")
    if probe.feature_mean is None or probe.feature_std is None:
        raise ValueError("Probe artifact is missing activation standardization statistics.")
    if probe.coef.ndim != 2 or probe.coef.shape[0] != 1:
        raise ValueError(
            "Expected one continuous linear-probe coefficient row, "
            f"received shape {tuple(probe.coef.shape)}."
        )

    pooled = pool_moment_residual(
        residual,
        num_variates=num_variates,
        token_position=probe.token_position,
        pooling_mode=probe.pooling_mode,
    ).detach().cpu().float()
    feature_mean = probe.feature_mean.detach().cpu().float()
    feature_std = probe.feature_std.detach().cpu().float()
    if pooled.shape[-1] != feature_mean.numel() or pooled.shape[-1] != probe.coef.shape[-1]:
        raise ValueError(
            "MOMENT residual/probe feature dimensions disagree: "
            f"residual={pooled.shape[-1]}, mean={feature_mean.numel()}, "
            f"coef={probe.coef.shape[-1]}."
        )

    standardized = (pooled - feature_mean) / feature_std
    standardized = torch.nan_to_num(
        standardized, nan=0.0, posinf=1e3, neginf=-1e3
    ).clamp(-1e3, 1e3)
    predictions = (
        standardized @ probe.coef[0].detach().cpu().float()
        + probe.intercept[0].detach().cpu().float()
    )
    return (
        float(predictions.mean().item()),
        float(predictions.std(unbiased=False).item()),
        int(predictions.numel()),
    )
