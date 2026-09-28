from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def _load_audit():
    path = Path(__file__).resolve().parents[1] / "scripts" / "audit_rebuttal_e2e.py"
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_require_cell_replication_accepts_exact_cell_seed_cross_product():
    module = _load_audit()
    frame = pd.DataFrame(
        [
            {"cell": cell, "seed": seed}
            for cell in ("a", "b")
            for seed in (42, 43, 44, 45, 46)
        ]
    )

    assert (
        module.require_cell_replication(
            frame,
            cell_columns=["cell"],
            seed_column="seed",
            expected_n=5,
            expected_cells=2,
            expected_cell_values={("a",), ("b",)},
            label="test",
        )
        == 2
    )


def test_require_cell_replication_rejects_duplicate_rows_masking_missing_cell():
    module = _load_audit()
    frame = pd.DataFrame(
        [
            {"cell": "a", "seed": seed}
            for seed in (42, 43, 44, 45, 46)
        ]
        + [
            {"cell": "a", "seed": seed}
            for seed in (42, 43, 44, 45, 46)
        ]
    )

    with pytest.raises(ValueError, match="expected exactly 2"):
        module.require_cell_replication(
            frame,
            cell_columns=["cell"],
            seed_column="seed",
            expected_n=5,
            expected_cells=2,
            expected_cell_values={("a",), ("b",)},
            label="test",
        )


def test_require_cell_replication_rejects_duplicate_cell_seed_row():
    module = _load_audit()
    rows = [
        {"cell": cell, "seed": seed}
        for cell in ("a", "b")
        for seed in (42, 43, 44, 45, 46)
    ]
    rows.append({"cell": "a", "seed": 42})
    frame = pd.DataFrame(rows)

    with pytest.raises(ValueError, match="exactly one row"):
        module.require_cell_replication(
            frame,
            cell_columns=["cell"],
            seed_column="seed",
            expected_n=5,
            expected_cells=2,
            expected_cell_values={("a",), ("b",)},
            label="test",
        )


def test_require_cell_replication_rejects_wrong_cell_identity():
    module = _load_audit()
    frame = pd.DataFrame(
        [
            {"cell": cell, "seed": seed}
            for cell in ("a", "wrong")
            for seed in (42, 43, 44, 45, 46)
        ]
    )

    with pytest.raises(ValueError, match="wrong cell identities"):
        module.require_cell_replication(
            frame,
            cell_columns=["cell"],
            seed_column="seed",
            expected_n=5,
            expected_cells=2,
            expected_cell_values={("a",), ("b",)},
            label="test",
        )
