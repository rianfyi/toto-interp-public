from __future__ import annotations

from itertools import combinations
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from .types import ActivationBatch

STRUCTURAL_CONFOUND_LABELS = ("domain", "metric_type", "frequency_bucket")


def series_label_frame(batch: ActivationBatch, labels: Iterable[str] = STRUCTURAL_CONFOUND_LABELS) -> pd.DataFrame:
    """Collapse repeated activation rows to one audited label record per series.

    Activation dumps contain many layers, views, and sometimes windows for a
    series.  Computing a contingency table from those rows would make the
    association depend on tracing configuration, so this routine verifies label
    consistency and uses the series as the observational unit.
    """

    requested = tuple(labels)
    missing = set(requested).difference(batch.labels)
    if missing:
        raise KeyError(f"Activation dump is missing structural labels: {sorted(missing)}")
    records: dict[str, dict[str, Any]] = {}
    for index, series_id in enumerate(batch.series_ids):
        record = {"series_id": series_id, "split": batch.splits[index]}
        record.update({name: batch.labels[name][index] for name in requested})
        existing = records.get(series_id)
        if existing is None:
            records[series_id] = record
        elif existing != record:
            raise ValueError(
                f"Series {series_id!r} has inconsistent split or structural labels across activation rows."
            )
    return pd.DataFrame(records.values()).sort_values("series_id").reset_index(drop=True)


def cramers_v(left: Iterable[Any], right: Iterable[Any]) -> tuple[float, int, pd.DataFrame]:
    """Compute uncorrected Cramér's V and return the exact contingency table."""

    table = pd.crosstab(pd.Series(list(left), name="left"), pd.Series(list(right), name="right"), dropna=False)
    n = int(table.to_numpy().sum())
    if n == 0 or min(table.shape) < 2:
        return float("nan"), n, table
    statistic, _, _, _ = chi2_contingency(table, correction=False)
    denominator = min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt((statistic / n) / denominator)), n, table


def pairwise_cramers_v(
    frame: pd.DataFrame,
    labels: Iterable[str] = STRUCTURAL_CONFOUND_LABELS,
    *,
    split: str = "all",
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    """Report Cramér's V for every requested structural-label pair."""

    requested = tuple(labels)
    missing = set(requested).difference(frame.columns)
    if missing:
        raise KeyError(f"Structural label frame is missing columns: {sorted(missing)}")
    subset = frame if split == "all" else frame[frame["split"] == split]
    rows: list[dict[str, Any]] = []
    tables: dict[str, dict[str, Any]] = {}
    for left, right in combinations(requested, 2):
        valid = subset[[left, right]].dropna()
        score, n, table = cramers_v(valid[left], valid[right])
        key = f"{left}__{right}__{split}"
        rows.append(
            {
                "left_label": left,
                "right_label": right,
                "split": split,
                "series_count": n,
                "left_cardinality": int(table.shape[0]),
                "right_cardinality": int(table.shape[1]),
                "cramers_v": score,
                "formula": "sqrt((chi2 / n) / min(r - 1, c - 1))",
                "unit": "series",
            }
        )
        tables[key] = {
            "left_label": left,
            "right_label": right,
            "split": split,
            "series_count": n,
            "row_labels": [str(value) for value in table.index.tolist()],
            "column_labels": [str(value) for value in table.columns.tolist()],
            "values": table.astype(int).to_numpy().tolist(),
        }
    return pd.DataFrame(rows), tables
