from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

CANONICAL_SEEDS = [42, 43, 44, 45, 46]
STRUCTURAL_LABELS = (
    "cardinality_bucket",
    "domain",
    "frequency_bucket",
    "metric_type",
)
DYNAMIC_LABELS = (
    "coordination",
    "current_burstiness",
    "current_sparsity",
    "future_burstiness",
    "future_sparsity",
    "shift_risk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail closed unless every reported five-resplit cell is complete."
    )
    parser.add_argument("--moment-exchange-transfer-summary-dir", type=Path, required=True)
    parser.add_argument("--taxonomy-summary-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=CANONICAL_SEEDS
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checked(
    path: Path,
    *,
    seed_column: str,
    seeds: list[int],
    expected_rows: int,
    required_finite_columns: tuple[str, ...],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required aggregate: {path}")
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise ValueError(
            f"{path} has {len(frame)} rows; expected exactly {expected_rows}."
        )
    observed = set(pd.to_numeric(frame[seed_column], errors="raise").astype(int))
    if observed != set(seeds):
        raise ValueError(f"{path} has seeds {sorted(observed)}, expected {seeds}.")
    missing = set(required_finite_columns).difference(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required finite metrics: {sorted(missing)}"
        )
    for column in required_finite_columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy()).all():
            raise ValueError(f"{path} has non-finite values in required metric {column}")
    return frame


def require_cell_replication(
    frame: pd.DataFrame,
    *,
    cell_columns: list[str],
    seed_column: str,
    expected_n: int,
    expected_cells: int,
    expected_cell_values: set[tuple[object, ...]],
    label: str,
) -> int:
    groups = frame.groupby(cell_columns, dropna=False)
    distinct_seed_counts = groups[seed_column].nunique()
    row_counts = groups.size()
    if len(distinct_seed_counts) != expected_cells:
        raise ValueError(
            f"{label} has {len(distinct_seed_counts)} cells; expected exactly "
            f"{expected_cells}: {distinct_seed_counts.to_dict()}"
        )
    observed_values = {
        tuple(row)
        for row in frame[cell_columns].drop_duplicates().itertuples(
            index=False, name=None
        )
    }
    if observed_values != expected_cell_values:
        raise ValueError(
            f"{label} has the wrong cell identities; missing "
            f"{sorted(expected_cell_values - observed_values, key=str)}, "
            f"unexpected {sorted(observed_values - expected_cell_values, key=str)}"
        )
    if not (distinct_seed_counts == expected_n).all() or not (
        row_counts == expected_n
    ).all():
        raise ValueError(
            f"{label} cells must each contain exactly one row for each of "
            f"{expected_n} distinct resplits; distinct seeds "
            f"{distinct_seed_counts.to_dict()}, rows {row_counts.to_dict()}"
        )
    return int(len(distinct_seed_counts))


def main() -> None:
    args = parse_args()
    if args.seeds != CANONICAL_SEEDS:
        raise ValueError(
            "The end-to-end audit requires exactly the canonical five ordered "
            f"resplits {CANONICAL_SEEDS}; received {args.seeds}"
        )
    n = len(args.seeds)
    replication = args.moment_exchange_transfer_summary_dir
    taxonomy = args.taxonomy_summary_dir

    checks: dict[str, dict[str, object]] = {}

    def record(
        name: str,
        path: Path,
        frame: pd.DataFrame,
        *,
        cell_columns: list[str],
        seed_column: str,
        expected_cells: int,
        expected_cell_values: set[tuple[object, ...]],
    ) -> None:
        checks[name] = {
            "status": "complete",
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
            "replicated_cells": require_cell_replication(
                frame,
                cell_columns=cell_columns,
                seed_column=seed_column,
                expected_n=n,
                expected_cells=expected_cells,
                expected_cell_values=expected_cell_values,
                label=name,
            ),
            "seeds": args.seeds,
        }

    path = replication / "paired_patch_per_resplit.csv"
    frame = load_checked(
        path,
        seed_column="split_seed",
        seeds=args.seeds,
        expected_rows=3 * n,
        required_finite_columns=(
            "probe_test_r2",
            "probe_raw_test_r2",
            "burst_win_fraction",
            "probe_win_fraction",
            "wape_real_over_null_median",
        ),
    )
    record(
        "toto_matched_patch",
        path,
        frame,
        cell_columns=["blend"],
        seed_column="split_seed",
        expected_cells=3,
        expected_cell_values={(0.25,), (0.5,), (1.0,)},
    )

    path = replication / "moment_per_resplit.csv"
    frame = load_checked(
        path,
        seed_column="split_seed",
        seeds=args.seeds,
        expected_rows=4 * n,
        required_finite_columns=(
            "val_macro_f1",
            "test_macro_f1",
            "raw_test_macro_f1",
            "shuffled_test_macro_f1",
        ),
    )
    record(
        "moment_structural",
        path,
        frame,
        cell_columns=["label"],
        seed_column="split_seed",
        expected_cells=4,
        expected_cell_values={(label,) for label in STRUCTURAL_LABELS},
    )

    path = replication / "moment_dynamic_per_resplit.csv"
    frame = load_checked(
        path,
        seed_column="split_seed",
        seeds=args.seeds,
        expected_rows=6 * n,
        required_finite_columns=(
            "val_r2",
            "test_r2",
            "raw_test_r2",
            "shuffled_test_r2",
        ),
    )
    record(
        "moment_dynamic",
        path,
        frame,
        cell_columns=["label"],
        seed_column="split_seed",
        expected_cells=6,
        expected_cell_values={(label,) for label in DYNAMIC_LABELS},
    )

    for filename, name, cell_count in (
        (
            "moment_random_structural_per_resplit.csv",
            "moment_random_structural",
            4,
        ),
        ("moment_random_dynamic_per_resplit.csv", "moment_random_dynamic", 6),
    ):
        path = replication / filename
        frame = load_checked(
            path,
            seed_column="split_seed",
            seeds=args.seeds,
            expected_rows=cell_count * n,
            required_finite_columns=(
                "random_validation_score",
                "random_test_score",
            ),
        )
        record(
            name,
            path,
            frame,
            cell_columns=["label"],
            seed_column="split_seed",
            expected_cells=cell_count,
            expected_cell_values={
                (label,)
                for label in (
                    STRUCTURAL_LABELS
                    if cell_count == len(STRUCTURAL_LABELS)
                    else DYNAMIC_LABELS
                )
            },
        )

    path = replication / "moment_interchange_per_resplit.csv"
    frame = load_checked(
        path,
        seed_column="split_seed",
        seeds=args.seeds,
        expected_rows=3 * n,
        required_finite_columns=(
            "probe_win_fraction",
            "future_mae_real_better_fraction",
            "future_mae_real_minus_null_median",
        ),
    )
    record(
        "moment_matched_interchange",
        path,
        frame,
        cell_columns=["blend"],
        seed_column="split_seed",
        expected_cells=3,
        expected_cell_values={(0.25,), (0.5,), (1.0,)},
    )

    path = replication / "transfer_per_resplit.csv"
    frame = load_checked(
        path,
        seed_column="split_seed",
        seeds=args.seeds,
        expected_rows=12 * n,
        required_finite_columns=(
            "dataset_count",
            "transfer_r2_macro",
            "raw_transfer_r2_macro",
        ),
    )
    record(
        "external_transfer",
        path,
        frame,
        cell_columns=["benchmark", "label"],
        seed_column="split_seed",
        expected_cells=12,
        expected_cell_values={
            (benchmark, label)
            for benchmark in ("fev", "lsf")
            for label in DYNAMIC_LABELS
        },
    )

    structural_sources = {
        (label, source)
        for label in STRUCTURAL_LABELS
        for source in ("pretrained", "random_init")
    }
    taxonomy_specs = (
        (
            "raw_control_all_seeds.csv",
            "taxonomy_raw_controls",
            16,
            ["label", "method"],
            {
                (label, method)
                for label in STRUCTURAL_LABELS
                for method in ("cnn", "fno", "transformer", "gbdt")
            },
            ("val_accuracy", "test_accuracy", "test_macro_f1"),
        ),
        (
            "unconditional_selected_all_seeds.csv",
            "taxonomy_toto_pretrained_random",
            8,
            ["label", "source"],
            structural_sources,
            (
                "val_accuracy",
                "val_macro_f1",
                "test_accuracy",
                "test_macro_f1",
                "baseline_test_macro_f1",
                "shuffled_test_macro_f1",
            ),
        ),
        (
            "layer_permuted_selected_all_seeds.csv",
            "taxonomy_layer_permuted",
            4,
            ["label"],
            {(label,) for label in STRUCTURAL_LABELS},
            (
                "val_accuracy",
                "val_macro_f1",
                "test_accuracy",
                "test_macro_f1",
            ),
        ),
        (
            "conditional_all_seeds.csv",
            "taxonomy_conditional",
            6,
            ["conditional_key", "source"],
            {
                (key, source)
                for key in (
                    "domain_given_gauge_short",
                    "frequency_given_infra_gauge",
                    "metric_type_given_app_short",
                )
                for source in ("pretrained", "random_init")
            },
            (
                "val_macro_f1",
                "test_macro_f1",
                "baseline_test_macro_f1",
                "shuffled_test_macro_f1",
            ),
        ),
        (
            "pairwise_cramers_v_all_seeds.csv",
            "taxonomy_confounding",
            12,
            ["left_label", "right_label", "split"],
            {
                (left, right, split)
                for left, right in (
                    ("domain", "metric_type"),
                    ("domain", "frequency_bucket"),
                    ("metric_type", "frequency_bucket"),
                )
                for split in ("all", "train", "val", "test")
            },
            ("cramers_v", "series_count"),
        ),
        (
            "structural_holdout_all_seeds.csv",
            "taxonomy_structural_holdout",
            8,
            ["target", "holdout_mode", "source"],
            {
                (target, mode, source)
                for target, mode in (
                    ("frequency_bucket", "combination"),
                    ("metric_type", "combination"),
                    ("frequency_bucket", "domain"),
                    ("metric_type", "domain"),
                )
                for source in ("pretrained", "random_init")
            },
            (
                "val_accuracy",
                "test_accuracy",
                "test_macro_f1",
                "baseline_test_accuracy",
                "shuffled_test_accuracy",
            ),
        ),
    )
    for (
        filename,
        name,
        cells,
        cell_columns,
        expected_cell_values,
        required_finite_columns,
    ) in taxonomy_specs:
        path = taxonomy / filename
        frame = load_checked(
            path,
            seed_column="suite_seed",
            seeds=args.seeds,
            expected_rows=cells * n,
            required_finite_columns=required_finite_columns,
        )
        record(
            name,
            path,
            frame,
            cell_columns=cell_columns,
            seed_column="suite_seed",
            expected_cells=cells,
            expected_cell_values=expected_cell_values,
        )

    payload = {
        "schema_version": 1,
        "status": "complete",
        "seeds": args.seeds,
        "replication_unit": "seeded_series_disjoint_boom_resplit",
        "interval_role": "descriptive_between_resplit_variability",
        "checks": checks,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
