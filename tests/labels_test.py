from __future__ import annotations

import torch

from toto_interp.boom import split_boom_series_ids
from toto_interp.labels import compute_dynamic_regime_labels


def test_dynamic_regime_labelers_follow_signal_structure():
    zeros = torch.zeros(2, 8)
    zeros_next = torch.zeros(2, 4)
    sparse_labels = compute_dynamic_regime_labels(zeros, zeros_next, num_target_variates=2)
    assert sparse_labels["current_sparsity"] == 1.0
    assert sparse_labels["future_sparsity"] == 1.0

    spiky_context = torch.ones(2, 8)
    calm_next = torch.ones(2, 4)
    spiky_next = calm_next.clone()
    spiky_next[0, -1] = 50.0
    calm_labels = compute_dynamic_regime_labels(spiky_context, calm_next, num_target_variates=2)
    spiky_labels = compute_dynamic_regime_labels(spiky_context, spiky_next, num_target_variates=2)
    assert spiky_labels["future_burstiness"] > calm_labels["future_burstiness"]

    stable_context = torch.arange(16, dtype=torch.float32).view(2, 8)
    stable_next = stable_context[:, -4:].clone()
    shifted_next = stable_next + 20.0
    stable_labels = compute_dynamic_regime_labels(stable_context, stable_next, num_target_variates=2)
    shifted_labels = compute_dynamic_regime_labels(stable_context, shifted_next, num_target_variates=2)
    assert shifted_labels["shift_risk"] > stable_labels["shift_risk"]

    mixed_next = stable_next.clone()
    mixed_next[0] += 10.0
    mixed_next[1] -= 10.0
    coordinated_next = stable_next + 10.0
    mixed_labels = compute_dynamic_regime_labels(stable_context, mixed_next, num_target_variates=2)
    coordinated_labels = compute_dynamic_regime_labels(stable_context, coordinated_next, num_target_variates=2)
    assert coordinated_labels["coordination"] > mixed_labels["coordination"]


def test_boom_split_is_series_disjoint():
    taxonomy = {f"series-{idx}": {"dummy": True} for idx in range(100)}
    splits = split_boom_series_ids(taxonomy, seed=7)

    train = set(splits["train"])
    val = set(splits["val"])
    test = set(splits["test"])

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert len(train | val | test) == len(taxonomy)
