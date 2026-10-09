from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import t as student_t

CANONICAL_SEEDS = [42, 43, 44, 45, 46]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the five-seed taxonomy-control raw-control and conditional suites."
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--layer-permuted-runs-root",
        type=Path,
        default=None,
        help="Optional root containing the five layer-permuted-pretrained controls.",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=CANONICAL_SEEDS
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_requested_seeds(seeds: list[int]) -> None:
    if seeds != CANONICAL_SEEDS:
        raise ValueError(
            "Taxonomy-control aggregation requires exactly the canonical five ordered "
            f"resplits {CANONICAL_SEEDS}; received {seeds}"
        )


def validate_activation_provenance(
    runs_root: Path, seeds: list[int]
) -> dict[int, dict[str, str]]:
    """Require current eval-mode, disjoint, seed-matched activation summaries."""
    split_hashes_by_seed: dict[int, dict[str, str]] = {}
    for seed in seeds:
        source_hashes: dict[str, dict[str, str]] = {}
        for source in ("pretrained", "random_init"):
            path = (
                runs_root
                / f"seed_{seed}"
                / f"{source}_activations"
                / "activation_dump_summary.json"
            )
            if not path.exists():
                raise FileNotFoundError(f"Missing activation provenance: {path}")
            summary = json.loads(path.read_text())
            if int(summary.get("seed", -1)) != seed:
                raise ValueError(f"{path} records the wrong split seed")
            if summary.get("weight_source") != source:
                raise ValueError(f"{path} records the wrong weight source")
            if summary.get("model_training") is not False:
                raise ValueError(f"{path} does not record evaluation mode")
            overlaps = summary.get("split_overlap_counts")
            if (
                not isinstance(overlaps, dict)
                or set(overlaps) != {"train_val", "train_test", "val_test"}
                or any(int(value) != 0 for value in overlaps.values())
            ):
                raise ValueError(f"{path} does not prove series-disjoint splits")
            hashes = summary.get("series_id_sha256_by_split")
            if (
                not isinstance(hashes, dict)
                or set(hashes) != {"train", "val", "test"}
                or any(
                    not isinstance(value, str) or len(value) != 64
                    for value in hashes.values()
                )
                or len(set(hashes.values())) != 3
            ):
                raise ValueError(f"{path} has invalid split hashes")
            source_hashes[source] = hashes
        if source_hashes["pretrained"] != source_hashes["random_init"]:
            raise ValueError(
                f"Seed {seed} pretrained/random controls do not share exact splits"
            )
        split_hashes_by_seed[seed] = source_hashes["pretrained"]
    return split_hashes_by_seed


def require_true_flags(frame: pd.DataFrame, column: str, path: Path) -> None:
    def parse(value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        normalized = str(value).strip().lower()
        if normalized == "true":
            return True
        if normalized == "false":
            return False
        return None

    parsed = frame[column].map(parse)
    if parsed.isna().any() or not parsed.all():
        raise ValueError(f"{path} does not record {column}=true for every row")


def mean_ci(series: pd.Series) -> tuple[float, float, int]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    count = len(values)
    if not count:
        return float("nan"), float("nan"), 0
    mean = float(values.mean())
    ci = float(student_t.ppf(0.975, df=count - 1) * values.sem()) if count > 1 else 0.0
    return mean, ci, count


def value(mean: float, ci: float) -> str:
    return "--" if pd.isna(mean) else f"{mean:.3f} +/- {ci:.3f}"


def read_csvs(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing expected suite output: {path}")
        frames.append(pd.read_csv(path))
    return pd.concat(frames, ignore_index=True)


def validate_suite_frame(
    frame: pd.DataFrame,
    *,
    path: Path,
    expected_seed: int,
    expected_method: str,
    expected_weight_source: str,
) -> None:
    required = {"seed", "method", "weight_source"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Suite output {path} is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"Suite output {path} is empty")

    seeds = pd.to_numeric(frame["seed"], errors="coerce")
    if seeds.isna().any() or not (seeds == expected_seed).all():
        raise ValueError(f"Suite output {path} does not contain only seed {expected_seed}")
    methods = set(frame["method"].dropna().astype(str))
    if methods != {expected_method}:
        raise ValueError(f"Suite output {path} has methods {sorted(methods)}, expected {expected_method!r}")
    sources = set(frame["weight_source"].dropna().astype(str))
    if sources != {expected_weight_source}:
        raise ValueError(
            f"Suite output {path} has weight sources {sorted(sources)}, expected {expected_weight_source!r}"
        )


def read_suite_frame(
    path: Path,
    *,
    expected_seed: int,
    expected_method: str,
    expected_weight_source: str,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected suite output: {path}")
    frame = pd.read_csv(path)
    validate_suite_frame(
        frame,
        path=path,
        expected_seed=expected_seed,
        expected_method=expected_method,
        expected_weight_source=expected_weight_source,
    )
    return frame


def relativize_artifact_paths(frame: pd.DataFrame, runs_root: Path) -> pd.DataFrame:
    frame = frame.copy()
    if "artifact_path" not in frame.columns:
        return frame

    def relative(value: object) -> object:
        if pd.isna(value):
            return value
        path = Path(str(value))
        try:
            return str(path.relative_to(runs_root))
        except ValueError:
            return path.name

    frame["artifact_path"] = frame["artifact_path"].map(relative)
    return frame


def select_validation_view(frame: pd.DataFrame) -> pd.Series:
    """Select one predeclared headline view without inspecting test metrics."""
    label = str(frame["label"].iloc[0])
    primary = "val_macro_f1" if label == "cardinality_bucket" else "val_accuracy"
    valid = pd.to_numeric(frame[primary], errors="coerce").dropna()
    if valid.empty:
        raise ValueError(f"No finite {primary} values for {label}")
    return frame.loc[valid.idxmax()]


def load_layer_permuted_selected(
    runs_root: Path,
    seeds: list[int],
    expected_split_hashes: dict[int, dict[str, str]] | None = None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for seed in seeds:
        seed_root = runs_root / f"seed_{seed}"
        activation_root = seed_root / "layer_permuted_pretrained_activations"
        summary_path = activation_root / "activation_dump_summary.json"
        probe_path = (
            seed_root
            / "layer_permuted_pretrained_probes"
            / "probe_results.csv"
        )
        summary = json.loads(summary_path.read_text())
        if int(summary.get("seed", -1)) != seed:
            raise ValueError(f"{summary_path} records the wrong split seed")
        if summary.get("weight_source") != "layer_permuted_pretrained":
            raise ValueError(f"{summary_path} is not a layer-permuted control")
        if summary.get("model_training") is not False:
            raise ValueError(f"{summary_path} does not record evaluation mode")
        if any(int(value) for value in summary.get("split_overlap_counts", {}).values()):
            raise ValueError(f"{summary_path} records overlapping BOOM splits")
        if (
            expected_split_hashes is not None
            and summary.get("series_id_sha256_by_split")
            != expected_split_hashes[seed]
        ):
            raise ValueError(
                f"{summary_path} does not use the exact taxonomy-control resplit"
            )
        provenance = summary.get("weight_provenance", {})
        permutation = provenance.get("runtime_layer_to_pretrained_layer")
        if (
            provenance.get("control") != "transformer_block_order_permutation"
            or int(provenance.get("permutation_seed", -1)) != seed
            or not isinstance(permutation, list)
            or sorted(int(value) for value in permutation)
            != list(range(len(permutation)))
            or permutation == list(range(len(permutation)))
        ):
            raise ValueError(f"{summary_path} has invalid layer-permutation provenance")
        frame = read_suite_frame(
            probe_path,
            expected_seed=seed,
            expected_method="linear_probe",
            expected_weight_source="layer_permuted_pretrained",
        )
        selected_frame = pd.DataFrame(
            [
                select_validation_view(label_frame)
                for _, label_frame in frame.groupby("label", sort=True)
            ]
        )
        selected_frame["suite_seed"] = seed
        selected_frame["source"] = "layer_permuted_pretrained"
        selected_frame["runtime_layer_to_pretrained_layer"] = json.dumps(permutation)
        frames.append(selected_frame)
    output = pd.concat(frames, ignore_index=True)
    counts = output.groupby("label")["suite_seed"].nunique()
    if not (counts == len(seeds)).all():
        raise ValueError(
            f"Each layer-permuted label must have {len(seeds)} resplits; "
            f"observed {counts.to_dict()}"
        )
    return relativize_artifact_paths(output, runs_root)


def main() -> None:
    args = parse_args()
    validate_requested_seeds(args.seeds)
    split_hashes_by_seed = validate_activation_provenance(
        args.runs_root, args.seeds
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        for method in ("fno", "cnn", "transformer", "gbdt"):
            path = args.runs_root / f"seed_{seed}" / "raw_controls" / method / "probe_results.csv"
            frame = read_suite_frame(
                path,
                expected_seed=seed,
                expected_method=method,
                expected_weight_source="pretrained",
            )
            frame["suite_seed"] = seed
            raw_frames.append(frame)
    raw = relativize_artifact_paths(pd.concat(raw_frames, ignore_index=True), args.runs_root)
    raw.to_csv(args.output_dir / "raw_control_all_seeds.csv", index=False)

    raw_summary_rows: list[dict[str, object]] = []
    for (label, method), frame in raw.groupby(["label", "method"], sort=True):
        row: dict[str, object] = {"label": label, "method": method}
        for column in (
            "test_accuracy",
            "test_macro_f1",
            "parameter_count",
            "selected_epoch",
            "training_seconds",
            "peak_vram_bytes",
        ):
            if column in frame.columns:
                mean, ci, count = mean_ci(frame[column])
                row[f"{column}_mean"] = mean
                row[f"{column}_ci95"] = ci
                row[f"{column}_n"] = count
        raw_summary_rows.append(row)
    raw_summary = pd.DataFrame(raw_summary_rows)
    raw_summary.to_csv(args.output_dir / "raw_control_summary.csv", index=False)

    confounding_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        path = args.runs_root / f"seed_{seed}" / "structural_confounding" / "pairwise_cramers_v.csv"
        frame = pd.read_csv(path)
        required = {"left_label", "right_label", "split", "cramers_v", "series_count", "unit"}
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"Structural confounding output {path} is missing columns: {sorted(missing)}")
        if frame.empty or set(frame["unit"].dropna().astype(str)) != {"series"}:
            raise ValueError(f"Structural confounding output {path} must contain nonempty series-level results")
        frame["suite_seed"] = seed
        confounding_frames.append(frame)
    confounding = pd.concat(confounding_frames, ignore_index=True)
    confounding.to_csv(args.output_dir / "pairwise_cramers_v_all_seeds.csv", index=False)
    confounding_summary_rows: list[dict[str, object]] = []
    for (left, right, split), frame in confounding.groupby(["left_label", "right_label", "split"], sort=True):
        mean, ci, count = mean_ci(frame["cramers_v"])
        series_mean, _, _ = mean_ci(frame["series_count"])
        confounding_summary_rows.append(
            {
                "left_label": left,
                "right_label": right,
                "split": split,
                "cramers_v_mean": mean,
                "cramers_v_ci95": ci,
                "cramers_v_n": count,
                "series_count_mean": series_mean,
            }
        )
    confounding_summary = pd.DataFrame(confounding_summary_rows)
    confounding_summary.to_csv(args.output_dir / "pairwise_cramers_v_summary.csv", index=False)

    unconditional_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        for source in ("pretrained", "random_init"):
            path = args.runs_root / f"seed_{seed}" / "unconditional" / source / "probe_results.csv"
            frame = read_suite_frame(
                path,
                expected_seed=seed,
                expected_method="linear_probe",
                expected_weight_source=source,
            )
            selected = []
            for _, label_frame in frame.groupby("label", sort=True):
                selected.append(select_validation_view(label_frame))
            selected_frame = pd.DataFrame(selected)
            selected_frame["suite_seed"] = seed
            selected_frame["source"] = source
            unconditional_frames.append(selected_frame)
    unconditional = relativize_artifact_paths(pd.concat(unconditional_frames, ignore_index=True), args.runs_root)
    unconditional.to_csv(args.output_dir / "unconditional_selected_all_seeds.csv", index=False)

    unconditional_rows: list[dict[str, object]] = []
    for (label, source), frame in unconditional.groupby(["label", "source"], sort=True):
        row: dict[str, object] = {"label": label, "source": source}
        for column in (
            "test_accuracy",
            "test_macro_f1",
            "baseline_test_accuracy",
            "baseline_test_macro_f1",
            "shuffled_test_accuracy",
            "shuffled_test_macro_f1",
        ):
            mean, ci, count = mean_ci(frame[column])
            row[f"{column}_mean"] = mean
            row[f"{column}_ci95"] = ci
            row[f"{column}_n"] = count
        unconditional_rows.append(row)
    unconditional_summary = pd.DataFrame(unconditional_rows)
    unconditional_summary.to_csv(args.output_dir / "unconditional_summary.csv", index=False)

    layer_permuted = None
    layer_permuted_summary = pd.DataFrame()
    if args.layer_permuted_runs_root is not None:
        layer_permuted = load_layer_permuted_selected(
            args.layer_permuted_runs_root,
            args.seeds,
            expected_split_hashes=split_hashes_by_seed,
        )
        layer_permuted.to_csv(
            args.output_dir / "layer_permuted_selected_all_seeds.csv", index=False
        )
        layer_rows: list[dict[str, object]] = []
        for label, frame in layer_permuted.groupby("label", sort=True):
            row: dict[str, object] = {"label": label}
            for column in ("test_accuracy", "test_macro_f1"):
                mean, ci, count = mean_ci(frame[column])
                row[f"{column}_mean"] = mean
                row[f"{column}_ci95"] = ci
                row[f"{column}_n"] = count
            layer_rows.append(row)
        layer_permuted_summary = pd.DataFrame(layer_rows)
        layer_permuted_summary.to_csv(
            args.output_dir / "layer_permuted_summary.csv", index=False
        )

    control_gap_rows: list[dict[str, object]] = []
    pretrained = unconditional[unconditional["source"] == "pretrained"]
    random_init = unconditional[unconditional["source"] == "random_init"]
    for label, pretrained_frame in pretrained.groupby("label", sort=True):
        alternatives: list[tuple[str, pd.DataFrame]] = [
            ("random_init", random_init[random_init["label"] == label]),
        ]
        if layer_permuted is not None:
            alternatives.append(
                (
                    "layer_permuted_pretrained",
                    layer_permuted[layer_permuted["label"] == label],
                )
            )
        for method in ("cnn", "fno", "transformer", "gbdt"):
            alternatives.append((method, raw[(raw["label"] == label) & (raw["method"] == method)]))
        for alternative, alternative_frame in alternatives:
            paired = pretrained_frame.merge(
                alternative_frame,
                on=["label", "suite_seed"],
                suffixes=("_pretrained", "_alternative"),
                validate="one_to_one",
            )
            row: dict[str, object] = {"label": label, "alternative": alternative}
            for metric in ("test_accuracy", "test_macro_f1"):
                delta = paired[f"{metric}_pretrained"] - paired[f"{metric}_alternative"]
                mean, ci, count = mean_ci(delta)
                row[f"{metric}_gap_mean"] = mean
                row[f"{metric}_gap_ci95"] = ci
                row[f"{metric}_gap_n"] = count
                row[f"{metric}_wins"] = int((delta > 0).sum())
            control_gap_rows.append(row)
    control_gap_summary = pd.DataFrame(control_gap_rows)
    control_gap_summary.to_csv(args.output_dir / "control_gap_summary.csv", index=False)

    conditional_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        for source in ("pretrained", "random_init"):
            path = args.runs_root / f"seed_{seed}" / "conditional" / source / "conditional_probe_selected.csv"
            frame = read_suite_frame(
                path,
                expected_seed=seed,
                expected_method="linear_probe",
                expected_weight_source=source,
            )
            frame["suite_seed"] = seed
            frame["source"] = source
            conditional_frames.append(frame)
    conditional = relativize_artifact_paths(pd.concat(conditional_frames, ignore_index=True), args.runs_root)
    conditional.to_csv(args.output_dir / "conditional_all_seeds.csv", index=False)

    conditional_rows: list[dict[str, object]] = []
    for (key, source), frame in conditional.groupby(["conditional_key", "source"], sort=True):
        row: dict[str, object] = {
            "conditional_key": key,
            "source": source,
            "target": frame["target"].iloc[0],
            "conditions": frame["conditions"].iloc[0],
            "description": frame["description"].iloc[0],
            "balanced_series_counts": frame["balanced_series_counts"].iloc[0],
        }
        for column in ("test_macro_f1", "test_accuracy", "baseline_test_macro_f1", "shuffled_test_macro_f1"):
            mean, ci, count = mean_ci(frame[column])
            row[f"{column}_mean"] = mean
            row[f"{column}_ci95"] = ci
            row[f"{column}_n"] = count
        conditional_rows.append(row)
    conditional_summary = pd.DataFrame(conditional_rows)
    conditional_summary.to_csv(args.output_dir / "conditional_summary.csv", index=False)

    conditional_gap_rows: list[dict[str, object]] = []
    for key, frame in conditional.groupby("conditional_key", sort=True):
        pretrained_frame = frame[frame["source"] == "pretrained"].sort_values("suite_seed")
        random_frame = frame[frame["source"] == "random_init"].sort_values("suite_seed")
        paired = pretrained_frame.merge(
            random_frame[["suite_seed", "test_macro_f1"]],
            on="suite_seed",
            suffixes=("_pretrained", "_random_init"),
            validate="one_to_one",
        )
        alternatives = {
            "random_init": paired["test_macro_f1_random_init"],
            "raw_six_stat": paired["baseline_test_macro_f1"],
            "shuffled": paired["shuffled_test_macro_f1"],
        }
        for alternative, scores in alternatives.items():
            delta = paired["test_macro_f1_pretrained"] - scores
            mean, ci, count = mean_ci(delta)
            conditional_gap_rows.append(
                {
                    "conditional_key": key,
                    "target": paired["target"].iloc[0],
                    "conditions": paired["conditions"].iloc[0],
                    "alternative": alternative,
                    "test_macro_f1_gap_mean": mean,
                    "test_macro_f1_gap_ci95": ci,
                    "test_macro_f1_gap_n": count,
                    "test_macro_f1_wins": int((delta > 0).sum()),
                }
            )
    conditional_gap_summary = pd.DataFrame(conditional_gap_rows)
    conditional_gap_summary.to_csv(args.output_dir / "conditional_gap_summary.csv", index=False)

    holdout_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        for source in ("pretrained", "random_init"):
            paths = sorted(
                (args.runs_root / f"seed_{seed}" / "structural_holdout").glob(
                    f"*/*/rotation_*/{source}/structural_holdout_selected.csv"
                )
            )
            if not paths:
                raise FileNotFoundError(
                    f"No rotated structural holdout outputs for seed {seed}/{source}"
                )
            seen_cells: set[tuple[str, str]] = set()
            for path in paths:
                frame = read_suite_frame(
                    path,
                    expected_seed=seed,
                    expected_method="linear_probe",
                    expected_weight_source=source,
                )
                required = {
                    "target",
                    "holdout_mode",
                    "holdout_axes",
                    "holdout_values",
                    "rotation",
                    "no_heldout_value_in_train_or_val",
                    "source_test_only_evaluation",
                }
                missing = required.difference(frame.columns)
                if missing:
                    raise ValueError(
                        f"Structural holdout output {path} is missing columns: "
                        f"{sorted(missing)}"
                    )
                require_true_flags(
                    frame, "no_heldout_value_in_train_or_val", path
                )
                require_true_flags(
                    frame, "source_test_only_evaluation", path
                )
                cell = (
                    str(frame["target"].iloc[0]),
                    str(frame["holdout_mode"].iloc[0]),
                )
                if cell in seen_cells:
                    raise ValueError(
                        f"Duplicate structural holdout cell {cell} for seed {seed}/{source}"
                    )
                seen_cells.add(cell)
                frame["suite_seed"] = seed
                frame["source"] = source
                holdout_frames.append(frame)
    holdout = relativize_artifact_paths(pd.concat(holdout_frames, ignore_index=True), args.runs_root)
    holdout.to_csv(args.output_dir / "structural_holdout_all_seeds.csv", index=False)
    holdout_rows: list[dict[str, object]] = []
    for (target, mode, source), frame in holdout.groupby(["target", "holdout_mode", "source"], sort=True):
        if frame["suite_seed"].nunique() != len(args.seeds):
            raise ValueError(
                f"Held-out {target}/{mode}/{source} does not cover all requested seeds"
            )
        row: dict[str, object] = {
            "target": target,
            "holdout_mode": mode,
            "source": source,
            "holdout_axes": " | ".join(sorted(set(frame["holdout_axes"].astype(str)))),
            "rotated_holdout_value_count": int(frame["holdout_values"].nunique()),
        }
        for column in ("test_accuracy", "test_macro_f1", "baseline_test_accuracy", "shuffled_test_accuracy"):
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            row[f"{column}_mean"] = (
                float(values.mean()) if len(values) else float("nan")
            )
            row[f"{column}_n"] = int(len(values))
            row[f"{column}_min"] = (
                float(values.min()) if len(values) else float("nan")
            )
            row[f"{column}_max"] = (
                float(values.max()) if len(values) else float("nan")
            )
        holdout_rows.append(row)
    holdout_summary = pd.DataFrame(holdout_rows)
    holdout_summary.to_csv(args.output_dir / "structural_holdout_summary.csv", index=False)

    lines = [
        "# Taxonomy-Control Stress Tests",
        "",
        "Compute resources for the reported runs are listed in the repository README.",
        "",
        "Five seeded resplits, with train/validation/test series disjoint within each run. Values are means +/- Student-t(4) 95% half-widths across resplits; the half-widths measure variability across resplits of one corpus and are descriptive rather than significance tests. Raw controls receive the same masked context window and coverage channel; all neural raw controls use validation-selected checkpoints. The GBDT control receives explicit last-patch, spectral, and autocorrelation summaries. Pairwise Cramér's V uses one label record per series. The predeclared Toto view rule selects frequency, metric type, and domain by validation accuracy and cardinality by validation macro-F1; every reported macro-F1 remains held-out test performance. Conditional probes use only terminal-context or first-decode views, first average repeated windows within series, then balance target classes by series within each split and hold the two remaining structural labels fixed.",
        "",
        "## Raw-window controls",
        "",
        "| Target | Representation / raw family | Test accuracy | Test macro-F1 | Parameters | Selected epoch |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, row in unconditional_summary.sort_values(["label", "source"]).iterrows():
        lines.append(
            "| {label} | Toto {source} | {acc} | {f1} | -- | -- |".format(
                label=row["label"],
                source=row["source"],
                acc=value(row["test_accuracy_mean"], row["test_accuracy_ci95"]),
                f1=value(row["test_macro_f1_mean"], row["test_macro_f1_ci95"]),
            )
        )
    for _, row in raw_summary.sort_values(["label", "method"]).iterrows():
        lines.append(
            "| {label} | {method} | {acc} | {f1} | {params} | {epoch} |".format(
                label=row["label"],
                method=row["method"],
                acc=value(row.get("test_accuracy_mean", float("nan")), row.get("test_accuracy_ci95", float("nan"))),
                f1=value(row.get("test_macro_f1_mean", float("nan")), row.get("test_macro_f1_ci95", float("nan"))),
                params=value(
                    row.get("parameter_count_mean", float("nan")), row.get("parameter_count_ci95", float("nan"))
                ),
                epoch=value(row.get("selected_epoch_mean", float("nan")), row.get("selected_epoch_ci95", float("nan"))),
            )
        )

    lines.extend(
        [
            "",
            "## Structural-label association",
            "",
            "| Pair | Split | Cramér's V | Mean series count |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in confounding_summary.sort_values(["left_label", "right_label", "split"]).iterrows():
        lines.append(
            "| {left} × {right} | {split} | {score} | {count:.0f} |".format(
                left=row["left_label"],
                right=row["right_label"],
                split=row["split"],
                score=value(row["cramers_v_mean"], row["cramers_v_ci95"]),
                count=row["series_count_mean"],
            )
        )

    lines.extend(
        [
            "",
            "## Paired pretrained-control gaps",
            "",
            "| Target | Alternative | Test accuracy gap | Test macro-F1 gap | Macro-F1 wins |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in control_gap_summary.sort_values(["label", "alternative"]).iterrows():
        lines.append(
            "| {label} | {alternative} | {acc} | {f1} | {wins}/{n} |".format(
                label=row["label"],
                alternative=row["alternative"],
                acc=value(row["test_accuracy_gap_mean"], row["test_accuracy_gap_ci95"]),
                f1=value(row["test_macro_f1_gap_mean"], row["test_macro_f1_gap_ci95"]),
                wins=int(row["test_macro_f1_wins"]),
                n=int(row["test_macro_f1_gap_n"]),
            )
        )

    if not layer_permuted_summary.empty:
        lines.extend(
            [
                "",
                "## Pretrained block-order permutation",
                "",
                "Every seed reconstructs a fresh pretrained Toto copy, applies a "
                "deterministic non-identity permutation of its 12 transformer "
                "blocks, and records the exact runtime-to-pretrained map.",
                "",
                "| Target | Layer-permuted test accuracy | Layer-permuted test macro-F1 |",
                "|---|---:|---:|",
            ]
        )
        for _, row in layer_permuted_summary.sort_values("label").iterrows():
            lines.append(
                "| {label} | {acc} | {f1} |".format(
                    label=row["label"],
                    acc=value(row["test_accuracy_mean"], row["test_accuracy_ci95"]),
                    f1=value(row["test_macro_f1_mean"], row["test_macro_f1_ci95"]),
                )
            )

    lines.extend(
        [
            "",
            "## Conditional structural probes",
            "",
            "| Target and fixed stratum | Source | Test macro-F1 | Six-statistic raw macro-F1 | Shuffled-label macro-F1 |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in conditional_summary.sort_values(["conditional_key", "source"]).iterrows():
        label = f"{row['target']} conditioned on {row['conditions']}"
        lines.append(
            "| {label} | {source} | {probe} | {raw} | {shuffled} |".format(
                label=label,
                source=row["source"],
                probe=value(row["test_macro_f1_mean"], row["test_macro_f1_ci95"]),
                raw=value(row["baseline_test_macro_f1_mean"], row["baseline_test_macro_f1_ci95"]),
                shuffled=value(row["shuffled_test_macro_f1_mean"], row["shuffled_test_macro_f1_ci95"]),
            )
        )
    lines.extend(
        [
            "",
            "## Paired conditional gaps",
            "",
            "| Target and fixed stratum | Alternative | Test macro-F1 gap | Wins |",
            "|---|---|---:|---:|",
        ]
    )
    for _, row in conditional_gap_summary.sort_values(["conditional_key", "alternative"]).iterrows():
        label = f"{row['target']} conditioned on {row['conditions']}"
        lines.append(
            "| {label} | {alternative} | {gap} | {wins}/{n} |".format(
                label=label,
                alternative=row["alternative"],
                gap=value(row["test_macro_f1_gap_mean"], row["test_macro_f1_gap_ci95"]),
                wins=int(row["test_macro_f1_wins"]),
                n=int(row["test_macro_f1_gap_n"]),
            )
        )
    lines.extend(
        [
            "",
            "## Within-BOOM held-out-combination stress tests",
            "",
            "Each run removes a support-valid rotated tuple/domain from source "
            "train and validation, then evaluates only matching source-test series. "
            "Because rotations can select different held-out values across seeds, "
            "these are coverage ranges, not same-condition confidence intervals.",
            "",
            "| Target | Holdout | Source | Test accuracy | Test macro-F1 |",
            "|---|---|---|---:|---:|",
        ]
    )
    for _, row in holdout_summary.sort_values(["target", "holdout_mode", "source"]).iterrows():
        lines.append(
            "| {target} | {mode}: {axes} | {source} | {accuracy} | {f1} |".format(
                target=row["target"],
                mode=row["holdout_mode"],
                axes=row["holdout_axes"],
                source=row["source"],
                accuracy=(
                    f"{row['test_accuracy_mean']:.3f} "
                    f"[{row['test_accuracy_min']:.3f}, {row['test_accuracy_max']:.3f}]"
                ),
                f1=(
                    f"{row['test_macro_f1_mean']:.3f} "
                    f"[{row['test_macro_f1_min']:.3f}, {row['test_macro_f1_max']:.3f}]"
                ),
            )
        )
    lines.append("")
    (args.output_dir / "TAXONOMY_RESULTS.md").write_text("\n".join(lines))
    with open(args.output_dir / "taxonomy_summary.json", "w") as handle:
        json.dump(
            {
                "seeds": args.seeds,
                "raw_control_summary": raw_summary.to_dict(orient="records"),
                "pairwise_cramers_v_summary": confounding_summary.to_dict(orient="records"),
                "unconditional_summary": unconditional_summary.to_dict(orient="records"),
                "layer_permuted_summary": layer_permuted_summary.to_dict(
                    orient="records"
                ),
                "control_gap_summary": control_gap_summary.to_dict(orient="records"),
                "conditional_summary": conditional_summary.to_dict(orient="records"),
                "conditional_gap_summary": conditional_gap_summary.to_dict(orient="records"),
                "structural_holdout_summary": holdout_summary.to_dict(orient="records"),
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
