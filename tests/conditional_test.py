from __future__ import annotations

import torch

from toto_interp.conditional import aggregate_series_records, conditional_subset, default_conditional_probe_specs
from toto_interp.labels import RAW_FEATURE_NAMES
from toto_interp.types import ActivationBatch


def _conditional_batch() -> ActivationBatch:
    rows: list[tuple[str, str, str, str, str]] = []
    for split in ("train", "val", "test"):
        for repeat in range(2):
            for metric in ("gauge", "rate", "distribution"):
                rows.append((f"metric-{split}-{repeat}-{metric}", split, metric, "Application Usage", "Short"))
            for domain in ("Application Usage", "Infrastructure", "Database"):
                rows.append((f"domain-{split}-{repeat}-{domain}", split, "gauge", domain, "Short"))
            for frequency in ("Short", "Medium"):
                rows.append((f"frequency-{split}-{repeat}-{frequency}", split, "gauge", "Infrastructure", frequency))

    # Repeat each series twice to verify the conditional path aggregates windows before balancing.
    repeated = [row for row in rows for _ in range(2)]
    total = len(repeated)
    return ActivationBatch(
        activations=torch.arange(total * 3, dtype=torch.float32).view(total, 3),
        raw_features=torch.ones(total, len(RAW_FEATURE_NAMES)),
        raw_feature_names=RAW_FEATURE_NAMES,
        layer_indices=torch.full((total,), 5, dtype=torch.long),
        patch_indices=torch.zeros(total, dtype=torch.long),
        variate_indices=torch.full((total,), -1, dtype=torch.long),
        token_positions=["all_context"] * total,
        pooling_modes=["series_mean"] * total,
        series_ids=[row[0] for row in repeated],
        window_ids=[f"{row[0]}:{index % 2}" for index, row in enumerate(repeated)],
        splits=[row[1] for row in repeated],
        labels={
            "metric_type": [row[2] for row in repeated],
            "domain": [row[3] for row in repeated],
            "frequency_bucket": [row[4] for row in repeated],
        },
    )


def test_conditional_subsets_use_series_level_balance_and_fixed_nuisance_labels():
    batch = aggregate_series_records(_conditional_batch())
    assert len(batch) == 48

    for spec in default_conditional_probe_specs():
        conditioned, counts = conditional_subset(batch, spec, seed=11)
        assert all(conditioned.labels[name][index] == value for name, value in spec.conditions.items() for index in range(len(conditioned)))
        assert len(set(conditioned.series_ids)) == len(conditioned)
        for split_counts in counts.values():
            assert len(set(split_counts.values())) == 1
