from __future__ import annotations

import pytest
import torch

from toto_interp.confounds import cramers_v, pairwise_cramers_v, series_label_frame
from toto_interp.labels import RAW_FEATURE_NAMES
from toto_interp.types import ActivationBatch


def _batch_with_repeated_views() -> ActivationBatch:
    rows = [
        ("a", "train", "Application Usage", "gauge", "Short"),
        ("b", "train", "Application Usage", "rate", "Medium"),
        ("c", "test", "Infrastructure", "gauge", "Short"),
        ("d", "test", "Infrastructure", "rate", "Medium"),
    ]
    repeated = [row for row in rows for _ in range(3)]
    return ActivationBatch(
        activations=torch.ones(len(repeated), 2),
        raw_features=torch.ones(len(repeated), len(RAW_FEATURE_NAMES)),
        raw_feature_names=RAW_FEATURE_NAMES,
        layer_indices=torch.zeros(len(repeated), dtype=torch.long),
        patch_indices=torch.zeros(len(repeated), dtype=torch.long),
        variate_indices=torch.full((len(repeated),), -1, dtype=torch.long),
        token_positions=["final_context"] * len(repeated),
        pooling_modes=["series_mean"] * len(repeated),
        series_ids=[row[0] for row in repeated],
        window_ids=[f"{row[0]}:{index}" for index, row in enumerate(repeated)],
        splits=[row[1] for row in repeated],
        labels={
            "domain": [row[2] for row in repeated],
            "metric_type": [row[3] for row in repeated],
            "frequency_bucket": [row[4] for row in repeated],
        },
    )


def test_cramers_v_uses_a_series_level_contingency_table():
    frame = series_label_frame(_batch_with_repeated_views())
    assert len(frame) == 4

    score, count, table = cramers_v(frame["domain"], frame["frequency_bucket"])
    assert count == 4
    assert table.to_numpy().sum() == 4
    assert score == pytest.approx(0.0)

    rows, _ = pairwise_cramers_v(frame, split="all")
    assert len(rows) == 3
    assert set(rows["unit"]) == {"series"}


def test_series_label_frame_rejects_inconsistent_metadata():
    batch = _batch_with_repeated_views()
    batch.labels["domain"][1] = "Database"

    with pytest.raises(ValueError, match="inconsistent"):
        series_label_frame(batch)
