from __future__ import annotations

import torch


def wape(target: torch.Tensor, prediction: torch.Tensor) -> float:
    numerator = (target - prediction).abs().sum().item()
    denominator = target.abs().sum().clamp_min(1e-6).item()
    return float(numerator / denominator)


def mase(context: torch.Tensor, target: torch.Tensor, prediction: torch.Tensor) -> float:
    naive_scale = context.diff(dim=-1).abs().mean().clamp_min(1e-6).item()
    numerator = (target - prediction).abs().mean().item()
    return float(numerator / naive_scale)


def quantile_loss(target: torch.Tensor, prediction: torch.Tensor, q: float) -> torch.Tensor:
    error = target - prediction
    return torch.maximum(q * error, (q - 1.0) * error)


def weighted_quantile_loss(
    target: torch.Tensor,
    quantile_predictions: dict[float, torch.Tensor],
) -> float:
    denominator = target.abs().sum().clamp_min(1e-6)
    losses = [
        2.0 * quantile_loss(target, prediction, q).sum() / denominator
        for q, prediction in sorted(quantile_predictions.items())
    ]
    return float(torch.stack(losses).mean().item())
