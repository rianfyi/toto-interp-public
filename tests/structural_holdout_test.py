from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from toto_interp.labels import RAW_FEATURE_NAMES
from toto_interp.structural_holdout import (
    build_structural_holdout_batch,
    choose_structural_holdout,
)
from toto_interp.types import ActivationBatch


def _structural_batch() -> ActivationBatch:
    rows = []
    for split in ("train", "val", "test"):
        for domain in ("Application Usage", "Infrastructure", "Database"):
            for metric in ("gauge", "rate", "distribution"):
                for frequency in ("Short", "Medium"):
                    for replica in range(2):
                        rows.append(
                            (
                                f"{split}-{domain}-{metric}-{frequency}-{replica}",
                                split,
                                domain,
                                metric,
                                frequency,
                            )
                        )
    return ActivationBatch(
        activations=torch.arange(len(rows) * 3, dtype=torch.float32).view(len(rows), 3),
        raw_features=torch.ones(len(rows), len(RAW_FEATURE_NAMES)),
        raw_feature_names=RAW_FEATURE_NAMES,
        layer_indices=torch.full((len(rows),), 5, dtype=torch.long),
        patch_indices=torch.zeros(len(rows), dtype=torch.long),
        variate_indices=torch.full((len(rows),), -1, dtype=torch.long),
        token_positions=["final_context"] * len(rows),
        pooling_modes=["series_mean"] * len(rows),
        series_ids=[row[0] for row in rows],
        window_ids=[f"{row[0]}:0" for row in rows],
        splits=[row[1] for row in rows],
        labels={
            "domain": [row[2] for row in rows],
            "metric_type": [row[3] for row in rows],
            "frequency_bucket": [row[4] for row in rows],
        },
    )


def test_combination_holdout_excludes_heldout_tuple_from_train_and_validation():
    batch = _structural_batch()
    spec = choose_structural_holdout(
        batch,
        target_label="frequency_bucket",
        mode="combination",
        rotation=1,
    )
    heldout = build_structural_holdout_batch(batch, spec)

    def key(index: int) -> tuple[str, ...]:
        return tuple(str(heldout.labels[axis][index]) for axis in spec.holdout_axes)

    assert all(key(index) != spec.holdout_values for index, split in enumerate(heldout.splits) if split == "train")
    assert all(key(index) != spec.holdout_values for index, split in enumerate(heldout.splits) if split == "val")
    assert all(key(index) == spec.holdout_values for index, split in enumerate(heldout.splits) if split == "test")
    assert {
        heldout.labels["frequency_bucket"][index] for index, split in enumerate(heldout.splits) if split == "test"
    } == {
        "Short",
        "Medium",
    }


def test_domain_holdout_removes_domain_from_train_and_validation():
    batch = _structural_batch()
    spec = choose_structural_holdout(
        batch,
        target_label="metric_type",
        mode="domain",
        rotation=2,
    )
    heldout = build_structural_holdout_batch(batch, spec)

    heldout_domain = spec.holdout_values[0]
    assert heldout_domain not in {
        heldout.labels["domain"][index] for index, split in enumerate(heldout.splits) if split == "train"
    }
    assert heldout_domain not in {
        heldout.labels["domain"][index] for index, split in enumerate(heldout.splits) if split == "val"
    }
    assert {heldout.labels["domain"][index] for index, split in enumerate(heldout.splits) if split == "test"} == {
        heldout_domain
    }


def test_structural_holdout_runner_reuses_activation_dumps_and_records_leakage_guards(tmp_path: Path, monkeypatch):
    batch = _structural_batch()
    activation_files = []
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}_activations.pt"
        batch.subset(split=split).save(path)
        activation_files.append(path)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_structural_holdout_probes.py"
    spec = importlib.util.spec_from_file_location("run_structural_holdout_probes", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output_dir = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_structural_holdout_probes.py",
            "--activation-files",
            *(str(path) for path in activation_files),
            "--output-dir",
            str(output_dir),
            "--seed",
            "9",
            "--target-label",
            "frequency_bucket",
            "--holdout-mode",
            "combination",
            "--rotation",
            "4",
            "--token-positions",
            "final_context",
        ],
    )

    module.main()

    selected = pd.read_csv(output_dir / "structural_holdout_selected.csv")
    assert len(selected) == 1
    assert selected["no_heldout_value_in_train_or_val"].all()
    assert selected["source_test_only_evaluation"].all()


def test_holdout_grid_loads_each_activation_file_once(tmp_path: Path, monkeypatch):
    batch = _structural_batch()
    activation_files = []
    for split in ("train", "val", "test"):
        path = tmp_path / f"{split}_activations.pt"
        batch.subset(split=split).save(path)
        activation_files.append(path)

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_holdout_grid.py"
    spec = importlib.util.spec_from_file_location("run_holdout_grid", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls: list[Path] = []
    original_load = ActivationBatch.load

    def tracked_load(path: Path) -> ActivationBatch:
        calls.append(Path(path))
        return original_load(path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_holdout_grid.py",
            "--activation-files",
            *(str(path) for path in activation_files),
            "--output-root",
            str(tmp_path / "output"),
            "--conditional-output-dir",
            str(tmp_path / "conditional"),
            "--source",
            "pretrained",
            "--seed",
            "9",
            "--rotation",
            "1",
            "--token-positions",
            "final_context",
        ],
    )
    with patch.object(module.ActivationBatch, "load", side_effect=tracked_load):
        module.main()

    assert calls == activation_files
    for target_label, holdout_mode in module.HOLDOUT_GRID:
        selected_path = (
            tmp_path
            / "output"
            / holdout_mode
            / target_label
            / "rotation_1"
            / "pretrained"
            / "structural_holdout_selected.csv"
        )
        assert selected_path.exists()
    assert (
        tmp_path / "conditional" / "conditional_probe_selected.csv"
    ).exists()
