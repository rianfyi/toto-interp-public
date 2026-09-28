from __future__ import annotations

from typing import Any

import numpy as np
import torch

PRIMARY_METRIC_TYPES = ("gauge", "rate", "distribution")
PRIMARY_DOMAINS = ("Application Usage", "Infrastructure", "Database")

RAW_FEATURE_NAMES = (
    "last_patch_mean",
    "last_patch_std",
    "last_patch_zero_fraction",
    "last_patch_abs_mean",
    "last_patch_abs_max",
    "last_patch_variate_mean_std",
)


def bucket_num_variates(num_variates: int) -> str:
    if num_variates == 1:
        return "univariate"
    if num_variates <= 8:
        return "small_mv"
    if num_variates <= 32:
        return "medium_mv"
    return "high_mv"


def canonical_metric_type(metric_types: list[str]) -> str:
    if len(metric_types) == 1 and metric_types[0] in PRIMARY_METRIC_TYPES:
        return metric_types[0]
    if len(metric_types) == 1 and metric_types[0] == "count":
        return "auxiliary"
    return "auxiliary"


def canonical_domain(domains: list[str]) -> str:
    if len(domains) == 1 and domains[0] in PRIMARY_DOMAINS:
        return domains[0]
    return "auxiliary"


def robust_scale(values: torch.Tensor) -> torch.Tensor:
    values = values.float()
    median = values.median(dim=-1, keepdim=True).values
    mad = (values - median).abs().median(dim=-1, keepdim=True).values
    return 1.4826 * mad.clamp_min(1e-6)


def compute_dynamic_regime_labels(
    context: torch.Tensor,
    next_patch: torch.Tensor,
    *,
    num_target_variates: int | None = None,
) -> dict[str, float]:
    """
    Compute observability-style operating regime scores from a context window and
    the aligned next true patch.
    """

    context = context.float()
    next_patch = next_patch.float()
    if num_target_variates is None:
        num_target_variates = int(context.shape[0])

    target_context = context[:num_target_variates]
    target_next = next_patch[:num_target_variates]

    last_context_patch = target_context[:, -target_next.shape[-1] :]
    scale = robust_scale(target_context)
    context_median = target_context.median(dim=-1, keepdim=True).values

    current_sparsity = float((last_context_patch == 0).float().mean().item())
    future_sparsity = float((target_next == 0).float().mean().item())

    current_burstiness = float((((last_context_patch - context_median) / scale).abs().amax()).item())
    future_burstiness = float((((target_next - context_median) / scale).abs().amax()).item())

    last_patch_mean = last_context_patch.mean(dim=-1)
    next_patch_mean = target_next.mean(dim=-1)
    shift_risk = float(((next_patch_mean - last_patch_mean).abs() / scale.squeeze(-1)).mean().item())

    direction_delta = next_patch_mean - last_patch_mean
    active_threshold = 0.5 * scale.squeeze(-1)
    active_mask = direction_delta.abs() > active_threshold
    active_deltas = direction_delta[active_mask]
    if active_deltas.numel() == 0:
        coordination = 0.0
    else:
        positive = (active_deltas > 0).float().mean().item()
        negative = (active_deltas < 0).float().mean().item()
        coordination = float(max(positive, negative))

    return {
        "current_sparsity": current_sparsity,
        "future_sparsity": future_sparsity,
        "current_burstiness": current_burstiness,
        "future_burstiness": future_burstiness,
        "shift_risk": shift_risk,
        "coordination": coordination,
    }


def build_raw_baseline_features(context: torch.Tensor, next_patch: torch.Tensor) -> torch.Tensor:
    context = context.float()
    last_patch = context[:, -next_patch.shape[-1] :]
    per_variate_means = last_patch.mean(dim=-1)
    features = torch.tensor(
        [
            float(last_patch.mean().item()),
            float(last_patch.std(unbiased=False).item()),
            float((last_patch == 0).float().mean().item()),
            float(last_patch.abs().mean().item()),
            float(last_patch.abs().amax().item()),
            float(per_variate_means.std(unbiased=False).item()),
        ],
        dtype=torch.float32,
    )
    return features


def build_taxonomy_labels(series_id: str, taxonomy_meta: dict[str, Any]) -> dict[str, Any]:
    num_variates = int(taxonomy_meta["num_variates"])
    return {
        "series_id": series_id,
        "metric_type": canonical_metric_type(list(taxonomy_meta["type"])),
        "domain": canonical_domain(list(taxonomy_meta["domain"])),
        "frequency_bucket": str(taxonomy_meta["frequency"]),
        "cardinality_bucket": bucket_num_variates(num_variates),
        "num_variates": num_variates,
        "term_bucket": str(taxonomy_meta["term"]),
    }
