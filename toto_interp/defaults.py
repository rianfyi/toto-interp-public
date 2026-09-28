from __future__ import annotations

from .types import LabelSpec


def default_taxonomy_label_specs() -> list[LabelSpec]:
    return [
        LabelSpec(
            name="metric_type",
            task_type="categorical",
            classes=("gauge", "rate", "distribution"),
        ),
        LabelSpec(
            name="domain",
            task_type="categorical",
            classes=("Application Usage", "Infrastructure", "Database"),
        ),
        LabelSpec(
            name="frequency_bucket",
            task_type="categorical",
            classes=("Short", "Medium", "Long"),
        ),
        LabelSpec(
            name="cardinality_bucket",
            task_type="categorical",
            classes=("univariate", "small_mv", "medium_mv", "high_mv"),
        ),
    ]


def default_dynamic_label_specs() -> list[LabelSpec]:
    return [
        LabelSpec(name="current_sparsity", task_type="continuous"),
        LabelSpec(name="future_sparsity", task_type="continuous"),
        LabelSpec(name="current_burstiness", task_type="continuous"),
        LabelSpec(name="future_burstiness", task_type="continuous"),
        LabelSpec(name="shift_risk", task_type="continuous"),
        LabelSpec(name="coordination", task_type="continuous"),
    ]


def default_operational_label_specs() -> list[LabelSpec]:
    return [
        LabelSpec(name="current_burstiness", task_type="continuous"),
        LabelSpec(name="future_burstiness", task_type="continuous"),
        LabelSpec(name="shift_risk", task_type="continuous"),
        LabelSpec(name="coordination", task_type="continuous"),
    ]


def default_label_specs() -> list[LabelSpec]:
    return default_taxonomy_label_specs() + default_dynamic_label_specs()
