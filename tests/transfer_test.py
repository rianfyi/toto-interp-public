from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
from datasets import Dataset

from scripts.run_toto_transfer import finite_windows, resolve_probe_paths
from toto_interp import LabelSpec, ProbeArtifact
from toto_interp.transfer import build_fev_windows, build_fev_windows_from_dataset, build_lsf_windows

from .test_helpers import make_window_example


def test_build_fev_windows_from_dataset_supports_auto_target_detection():
    dataset = Dataset.from_dict(
        {
            "id": ["series-a", "series-b"],
            "timestamp": [
                pd.date_range("2024-01-01", periods=24, freq="h").to_numpy(),
                pd.date_range("2024-01-01", periods=24, freq="h").to_numpy(),
            ],
            "cpu": [list(range(24)), list(range(24, 48))],
            "mem": [list(range(100, 124)), list(range(124, 148))],
        }
    )
    dataset.set_format("numpy")

    windows = build_fev_windows_from_dataset(
        dataset,
        dataset_name="synthetic_fev",
        context_length=8,
        patch_size=4,
        max_series=1,
        max_windows_per_series=3,
    )

    assert len(windows) == 2
    assert windows[0].num_target_variates == 2
    assert windows[0].labels["benchmark_name"] == "fev"
    assert windows[0].labels["dataset_name"] == "synthetic_fev"


def test_build_lsf_windows_reads_local_custom_dataset(tmp_path: Path):
    data_dir = tmp_path / "electricity"
    data_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
            {
            "date": pd.date_range("2024-01-01", periods=240, freq="h"),
            "load": list(range(240)),
            "OT": list(range(500, 740)),
        }
    )
    frame.to_csv(data_dir / "electricity.csv", index=False)

    windows = build_lsf_windows(
        dataset_name="electricity",
        lsf_path=tmp_path,
        context_length=32,
        patch_size=8,
        max_series=1,
        max_windows_per_series=3,
    )

    assert len(windows) == 2
    assert windows[0].labels["benchmark_name"] == "lsf"
    assert windows[0].labels["dataset_name"] == "electricity"
    assert windows[0].num_target_variates == 2


def test_build_fev_windows_uses_official_task_metadata(monkeypatch):
    dataset = Dataset.from_dict(
        {
            "id": ["series-a"],
            "timestamp": [pd.date_range("2024-01-01", periods=24, freq="h").to_numpy()],
            "target": [list(range(24))],
            "Generation forecast": [list(range(100, 124))],
            "System load forecast": [list(range(200, 224))],
        }
    )
    dataset.set_format("numpy")

    monkeypatch.setattr("toto_interp.transfer.load_fev_dataset", lambda config_name, split="train": dataset)

    windows = build_fev_windows(
        config_name="epf_be",
        task_name="epf_be",
        context_length=8,
        patch_size=4,
        max_series=1,
        max_windows_per_series=3,
    )

    assert len(windows) == 2
    assert windows[0].num_target_variates == 1


def test_finite_windows_uses_complete_case_rule():
    valid = make_window_example()
    invalid = make_window_example()
    object.__setattr__(invalid, "window_id", "series-0:bad")
    object.__setattr__(
        invalid,
        "context",
        torch.tensor([[0.0, float("nan"), 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
    )

    eligible, excluded = finite_windows([valid, invalid])

    assert eligible == [valid]
    assert excluded == ["series-0:bad"]


def test_validation_selection_returns_one_best_view_per_label(tmp_path: Path):
    probe_dir = tmp_path / "probes"
    artifact_dir = probe_dir / "artifacts"
    artifact_dir.mkdir(parents=True)
    rows = []
    for label, layer, val_r2 in (
        ("future_burstiness", 1, 0.1),
        ("future_burstiness", 2, 0.2),
        ("shift_risk", 3, 0.3),
    ):
        path = artifact_dir / f"{label}_{layer}.pt"
        ProbeArtifact(
            label_spec=LabelSpec(label, "continuous"),
            layer=layer,
            token_position="all_context",
            pooling_mode="series_mean",
            coef=torch.zeros((1, 1)),
            intercept=torch.zeros(1),
            metrics={},
            baseline_metrics={},
            shuffled_metrics={},
            method="linear_probe",
            model_id="model",
            weight_source="pretrained",
            seed=42,
            artifact_metadata={},
        ).save(path)
        rows.append(
            {
                "label": label,
                "task_type": "continuous",
                "method": "linear_probe",
                "layer": layer,
                "token_position": "all_context",
                "pooling_mode": "series_mean",
                "val_r2": val_r2,
                "artifact_path": f"/stale/cluster/path/{path.name}",
            }
        )
    pd.DataFrame(rows).to_csv(probe_dir / "probe_results.csv", index=False)
    args = SimpleNamespace(
        probe_paths=None,
        probe_dir=probe_dir,
        validation_select_one_per_label=True,
    )

    selected = resolve_probe_paths(args)

    assert [path.name for path in selected] == [
        "future_burstiness_2.pt",
        "shift_risk_3.pt",
    ]
