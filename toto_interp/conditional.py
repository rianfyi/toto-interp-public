from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .types import ActivationBatch, LabelSpec


@dataclass(frozen=True)
class ConditionalProbeSpec:
    """A common-support structural probe conditioned on the other taxonomy axes."""

    key: str
    target: LabelSpec
    conditions: dict[str, Any]
    description: str


def default_conditional_probe_specs() -> tuple[ConditionalProbeSpec, ...]:
    """Predeclared, nonempty BOOM strata that condition on both other labels.

    BOOM lacks every domain--metric combination (for example, there are no
    Infrastructure/rate series), so a fully crossed factorial test is not
    identifiable. These three strata retain all classes of the target while
    fixing the two remaining structural labels.
    """

    return (
        ConditionalProbeSpec(
            key="metric_type_given_app_short",
            target=LabelSpec(
                name="metric_type",
                task_type="categorical",
                classes=("gauge", "rate", "distribution"),
            ),
            conditions={"domain": "Application Usage", "frequency_bucket": "Short"},
            description="Metric type conditional on Application Usage domain and Short cadence.",
        ),
        ConditionalProbeSpec(
            key="domain_given_gauge_short",
            target=LabelSpec(
                name="domain",
                task_type="categorical",
                classes=("Application Usage", "Infrastructure", "Database"),
            ),
            conditions={"metric_type": "gauge", "frequency_bucket": "Short"},
            description="Domain conditional on gauge metric type and Short cadence.",
        ),
        ConditionalProbeSpec(
            key="frequency_given_infra_gauge",
            target=LabelSpec(
                name="frequency_bucket",
                task_type="categorical",
                classes=("Short", "Medium"),
            ),
            conditions={"domain": "Infrastructure", "metric_type": "gauge"},
            description="Frequency conditional on Infrastructure domain and gauge metric type.",
        ),
    )


def _matches_conditions(labels: dict[str, list[Any]], index: int, conditions: dict[str, Any]) -> bool:
    return all(labels[name][index] == value for name, value in conditions.items())


def balanced_conditional_series_ids(
    batch: ActivationBatch,
    spec: ConditionalProbeSpec,
    *,
    seed: int,
) -> tuple[set[str], dict[str, dict[str, int]]]:
    """Choose balanced target classes at the series level within each split.

    Activation dumps have one row per view, so balancing rows would duplicate
    some series. Selecting identifiers first keeps every series and every view
    aligned and preserves the paper's series-disjoint split policy.
    """

    by_split_class: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for index, (series_id, split) in enumerate(zip(batch.series_ids, batch.splits)):
        if not _matches_conditions(batch.labels, index, spec.conditions):
            continue
        target = str(batch.labels[spec.target.name][index])
        if spec.target.classes is None or target not in spec.target.classes:
            continue
        by_split_class[split][target].add(series_id)

    rng = np.random.default_rng(seed)
    selected: set[str] = set()
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        class_to_series = by_split_class.get(split, {})
        if spec.target.classes is None:
            raise ValueError(f"Conditional probe {spec.key} requires categorical target classes.")
        missing = [name for name in spec.target.classes if not class_to_series.get(name)]
        if missing:
            raise ValueError(f"Conditional probe {spec.key} has no {split} examples for classes {missing}.")
        per_class = min(len(class_to_series[name]) for name in spec.target.classes)
        if per_class < 2:
            raise ValueError(
                f"Conditional probe {spec.key} has only {per_class} series per class in {split}; insufficient support."
            )
        counts[split] = {}
        for class_name in spec.target.classes:
            candidates = sorted(class_to_series[class_name])
            chosen = rng.choice(candidates, size=per_class, replace=False).tolist()
            selected.update(chosen)
            counts[split][class_name] = per_class
    return selected, counts


def conditional_subset(
    batch: ActivationBatch,
    spec: ConditionalProbeSpec,
    *,
    seed: int,
) -> tuple[ActivationBatch, dict[str, dict[str, int]]]:
    """Return the fixed-stratum, split-balanced activation batch for a condition."""

    selected_series, counts = balanced_conditional_series_ids(batch, spec, seed=seed)
    indices = [
        index
        for index, series_id in enumerate(batch.series_ids)
        if series_id in selected_series and _matches_conditions(batch.labels, index, spec.conditions)
    ]
    result = batch.subset_indices(indices)
    result.source_metadata.update(
        {
            "conditional_probe_key": spec.key,
            "conditional_conditions": dict(spec.conditions),
            "conditional_balanced_series_counts": counts,
        }
    )
    return result, counts


def aggregate_series_records(batch: ActivationBatch) -> ActivationBatch:
    """Average repeated windows before a conditional series-level probe.

    The original audit intentionally evaluates window-level readouts. The
    conditional stress test instead uses series as its statistical unit so
    that a few series with many windows cannot inflate its apparent support.
    """

    groups: dict[tuple[int, int, int, str, str, str, str], list[int]] = defaultdict(list)
    for index, series_id in enumerate(batch.series_ids):
        key = (
            int(batch.layer_indices[index]),
            int(batch.patch_indices[index]),
            int(batch.variate_indices[index]),
            batch.token_positions[index],
            batch.pooling_modes[index],
            series_id,
            batch.splits[index],
        )
        groups[key].append(index)

    ordered_groups = sorted(groups.values(), key=lambda indices: min(indices))
    first_indices = [indices[0] for indices in ordered_groups]
    activations = torch.stack([batch.activations[indices].mean(dim=0) for indices in ordered_groups])
    raw_features = torch.stack([batch.raw_features[indices].mean(dim=0) for indices in ordered_groups])
    result = ActivationBatch(
        activations=activations,
        raw_features=raw_features,
        raw_feature_names=batch.raw_feature_names,
        layer_indices=batch.layer_indices[first_indices],
        patch_indices=batch.patch_indices[first_indices],
        variate_indices=batch.variate_indices[first_indices],
        token_positions=[batch.token_positions[index] for index in first_indices],
        pooling_modes=[batch.pooling_modes[index] for index in first_indices],
        series_ids=[batch.series_ids[index] for index in first_indices],
        window_ids=[batch.window_ids[index] for index in first_indices],
        splits=[batch.splits[index] for index in first_indices],
        labels={name: [values[index] for index in first_indices] for name, values in batch.labels.items()},
        source_metadata={**batch.source_metadata, "conditional_series_aggregation": True},
    )
    return result
