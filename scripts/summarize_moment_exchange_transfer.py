from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t


STRUCTURAL_LABELS = ("frequency_bucket", "metric_type", "domain", "cardinality_bucket")
DYNAMIC_LABELS = (
    "current_sparsity",
    "future_sparsity",
    "current_burstiness",
    "future_burstiness",
    "shift_risk",
    "coordination",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the five-resplit Toto donor-exchange, MOMENT, and external-transfer runs."
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument(
        "--patch-runs-root",
        type=Path,
        default=None,
        help="Optional paired-patch root when it differs from --runs-root.",
    )
    parser.add_argument(
        "--moment-runs-root",
        type=Path,
        default=None,
        help="Optional MOMENT root when it differs from --runs-root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    return parser.parse_args()


def interval_half_width(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) <= 1:
        return float("nan")
    return float(t.ppf(0.975, df=len(clean) - 1) * clean.std(ddof=1) / np.sqrt(len(clean)))


def aggregate_columns(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(group_columns, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys))
        row["n_resplits"] = int(group["split_seed"].nunique())
        for column in value_columns:
            values = pd.to_numeric(group[column], errors="coerce").dropna()
            row[f"{column}_n_resplits"] = int(len(values))
            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_ci95_half_width"] = interval_half_width(values)
        rows.append(row)
    return pd.DataFrame(rows)


def load_paired_patch(runs_root: Path, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for split_seed in seeds:
        seed_root = runs_root / f"seed_{split_seed}"
        probe_result_path = seed_root / "future_burstiness_probe" / "probe_results.csv"
        if not probe_result_path.exists():
            raise FileNotFoundError(f"Missing fixed-view probe results: {probe_result_path}")
        probe_frame = pd.read_csv(probe_result_path)
        probe_rows = probe_frame[
            (probe_frame["label"] == "future_burstiness")
            & (probe_frame["layer"].astype(int) == 11)
            & (probe_frame["token_position"] == "all_context")
            & (probe_frame["pooling_mode"] == "series_mean")
        ]
        if len(probe_rows) != 1:
            raise ValueError(f"Expected one fixed-view probe row in {probe_result_path}, found {len(probe_rows)}")
        probe_row = probe_rows.iloc[0]
        blend_dirs = sorted(seed_root.glob("paired_patch_blend_*"))
        if not blend_dirs:
            raise FileNotFoundError(f"No paired-patch blend directories under {seed_root}")
        seen_blends: set[float] = set()
        for blend_dir in blend_dirs:
            result_path = blend_dir / "paired_patch_results.csv"
            meta_path = blend_dir / "paired_patch_meta.json"
            if not result_path.exists() or not meta_path.exists():
                raise FileNotFoundError(f"Incomplete paired-patch directory: {blend_dir}")
            meta = json.loads(meta_path.read_text())
            if int(meta.get("split_seed", -1)) != split_seed:
                raise ValueError(f"{meta_path} records split_seed={meta.get('split_seed')}, expected {split_seed}")
            if int(meta.get("probe_seed", -1)) != split_seed:
                raise ValueError(f"{meta_path} records probe_seed={meta.get('probe_seed')}, expected {split_seed}")
            if meta.get("model_training") is not False:
                raise ValueError(f"{meta_path} does not record evaluation-mode TOTO")
            if any(int(value) for value in meta.get("split_overlap_counts", {}).values()):
                raise ValueError(f"{meta_path} records overlapping series splits")
            if not meta.get("common_forecast_random_numbers"):
                raise ValueError(f"{meta_path} does not record common forecast random numbers")
            if not meta.get("cross_series_sources"):
                raise ValueError(f"{meta_path} does not record cross-series source matching")
            if meta.get("split") != "test":
                raise ValueError(f"{meta_path} must use the held-out test split")
            blend = float(meta["blend"])
            if blend in seen_blends:
                raise ValueError(f"Duplicate blend {blend} for split seed {split_seed}")
            seen_blends.add(blend)

            frame = pd.read_csv(result_path)
            if set(frame["split_seed"].astype(int)) != {split_seed}:
                raise ValueError(f"{result_path} contains inconsistent split_seed values")
            if set(frame["seed"].astype(int)) != {split_seed}:
                raise ValueError(f"{result_path} contains a sampling seed that differs from its split seed")
            key_columns = ["split_seed", "seed", "pair_id", "condition"]
            if frame.duplicated(key_columns).any():
                raise ValueError(f"{result_path} contains duplicate pair-condition rows")
            expected_conditions = {"clean", "real_patch", "null_patch"}
            pair_conditions = frame.groupby(["split_seed", "seed", "pair_id"])["condition"].agg(set)
            if not pair_conditions.map(lambda value: value == expected_conditions).all():
                raise ValueError(f"{result_path} contains incomplete clean/real/null triples")
            expected_pairs = int(meta.get("num_pairs_per_seed", 40))
            if len(pair_conditions) != expected_pairs:
                raise ValueError(
                    f"{result_path} contains {len(pair_conditions)} complete pairs, expected {expected_pairs}"
                )
            sources = frame.pivot(
                index=["split_seed", "seed", "pair_id"],
                columns="condition",
                values=["target_series_id", "source_series_id"],
            )
            target = sources[("target_series_id", "clean")]
            real_source = sources[("source_series_id", "real_patch")]
            null_source = sources[("source_series_id", "null_patch")]
            if (
                (target == real_source).any()
                or (target == null_source).any()
                or (real_source == null_source).any()
            ):
                raise ValueError(f"{result_path} violates cross-series source matching")

            signature_raw = meta.get("probe_activation_source_signature")
            try:
                signature = json.loads(signature_raw)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{meta_path} has no parseable probe activation provenance") from exc
            if int(signature.get("seed", -1)) != split_seed:
                raise ValueError(f"{meta_path} probe activation seed does not match split seed")
            if signature.get("model_training") is not False:
                raise ValueError(f"{meta_path} probe activations were not extracted in evaluation mode")
            signature_hashes = signature.get("series_id_sha256_by_split", {})
            meta_manifest = meta.get("split_manifest", {})
            for split_name in ("train", "val", "test"):
                if signature_hashes.get(split_name) != meta_manifest.get(split_name, {}).get(
                    "series_id_sha256"
                ):
                    raise ValueError(
                        f"{meta_path} probe and intervention disagree on {split_name} series split"
                    )

            pivot = frame.pivot(
                index=["split_seed", "seed", "pair_id"],
                columns="condition",
                values=["forecast_burstiness", "probe_score", "wape"],
            )
            required = {
                ("forecast_burstiness", "real_patch"),
                ("forecast_burstiness", "null_patch"),
                ("probe_score", "real_patch"),
                ("probe_score", "null_patch"),
                ("wape", "real_patch"),
                ("wape", "null_patch"),
            }
            if not required.issubset(set(pivot.columns)):
                raise ValueError(f"{result_path} is missing paired conditions")
            burst_diff = (
                pivot[("forecast_burstiness", "real_patch")]
                - pivot[("forecast_burstiness", "null_patch")]
            )
            score_diff = pivot[("probe_score", "real_patch")] - pivot[("probe_score", "null_patch")]
            wape_ratio = pivot[("wape", "real_patch")] / pivot[("wape", "null_patch")].clip(lower=1e-6)
            rows.append(
                {
                    "split_seed": split_seed,
                    "sampling_seeds": json.dumps(sorted(frame["seed"].astype(int).unique().tolist())),
                    "blend": blend,
                    "n_pairs": int(len(pivot)),
                    "probe_test_r2": float(probe_row["test_r2"]),
                    "probe_raw_test_r2": float(probe_row["baseline_test_r2"]),
                    "probe_shuffled_test_r2": float(probe_row["shuffled_test_r2"]),
                    "probe_minus_raw_r2": float(
                        probe_row["test_r2"] - probe_row["baseline_test_r2"]
                    ),
                    "probe_minus_shuffled_r2": float(
                        probe_row["test_r2"] - probe_row["shuffled_test_r2"]
                    ),
                    "burst_wins": int((burst_diff > 0).sum()),
                    "burst_losses": int((burst_diff < 0).sum()),
                    "burst_ties": int((burst_diff == 0).sum()),
                    "probe_wins": int((score_diff > 0).sum()),
                    "probe_losses": int((score_diff < 0).sum()),
                    "probe_ties": int((score_diff == 0).sum()),
                    "burst_win_fraction": float((burst_diff > 0).mean()),
                    "probe_win_fraction": float((score_diff > 0).mean()),
                    "burst_real_minus_null_median": float(burst_diff.median()),
                    "probe_real_minus_null_median": float(score_diff.median()),
                    "wape_real_over_null_median": float(wape_ratio.median()),
                }
            )
            manifests.append(
                {
                    "split_seed": split_seed,
                    "blend": blend,
                    "series_count": int(meta["series_count"]),
                    "series_id_sha256": str(meta["series_id_sha256"]),
                    "context_length": int(meta["context_length"]),
                    "num_samples": int(meta["num_samples"]),
                    "n_windows": int(meta["n_windows"]),
                    "probe_sha256": str(meta["probe_sha256"]),
                }
            )
    per_seed = pd.DataFrame(rows).sort_values(["blend", "split_seed"]).reset_index(drop=True)
    expected = len(seeds)
    counts = per_seed.groupby("blend")["split_seed"].nunique()
    if not (counts == expected).all():
        raise ValueError(f"Each blend must have {expected} resplits; observed {counts.to_dict()}")
    return per_seed, pd.DataFrame(manifests).sort_values(["blend", "split_seed"]).reset_index(drop=True)


def load_moment(runs_root: Path, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    for split_seed in seeds:
        seed_root = runs_root / f"seed_{split_seed}" / "moment"
        result_path = seed_root / "probes" / "probe_results.csv"
        summary_path = seed_root / "activations" / "activation_dump_summary.json"
        if not result_path.exists() or not summary_path.exists():
            raise FileNotFoundError(f"Incomplete MOMENT run for seed {split_seed}: {seed_root}")
        summary = json.loads(summary_path.read_text())
        if int(summary.get("seed", -1)) != split_seed:
            raise ValueError(f"{summary_path} records seed={summary.get('seed')}, expected {split_seed}")
        if summary.get("weight_source") != "pretrained":
            raise ValueError(f"{summary_path} is not a pretrained MOMENT run")
        if summary.get("model_training") is not False:
            raise ValueError(f"{summary_path} does not record evaluation mode")
        enrichment = summary.get("provenance_enrichment", {})
        if (
            enrichment.get("measurement_values_unchanged") is not True
            or enrichment.get("weight_source") not in (None, "pretrained")
        ):
            raise ValueError(f"{summary_path} lacks audited provenance enrichment")
        if any(int(value) for value in summary.get("split_overlap_counts", {}).values()):
            raise ValueError(f"{summary_path} records overlapping series splits")
        frame = pd.read_csv(result_path)
        if set(frame["seed"].astype(int)) != {split_seed}:
            raise ValueError(f"{result_path} contains inconsistent seed values")
        for label in STRUCTURAL_LABELS:
            label_frame = frame[frame["label"] == label].dropna(subset=["val_macro_f1"])
            if label_frame.empty:
                raise ValueError(f"{result_path} has no validation-scored row for {label}")
            selected = label_frame.sort_values(
                ["val_macro_f1", "layer", "token_position", "pooling_mode"],
                ascending=[False, True, True, True],
            ).iloc[0]
            rows.append(
                {
                    "split_seed": split_seed,
                    "label": label,
                    "layer": int(selected["layer"]),
                    "token_position": str(selected["token_position"]),
                    "pooling_mode": str(selected["pooling_mode"]),
                    "val_macro_f1": float(selected["val_macro_f1"]),
                    "test_macro_f1": float(selected["test_macro_f1"]),
                    "raw_test_macro_f1": float(selected["baseline_test_macro_f1"]),
                    "shuffled_test_macro_f1": float(selected["shuffled_test_macro_f1"]),
                    "moment_minus_raw": float(
                        selected["test_macro_f1"] - selected["baseline_test_macro_f1"]
                    ),
                    "moment_minus_shuffled": float(
                        selected["test_macro_f1"] - selected["shuffled_test_macro_f1"]
                    ),
                }
            )
        split_manifest = {"split_seed": split_seed}
        for split_name in ("train", "val", "test"):
            split_summary = summary["splits"][split_name]
            split_manifest[f"{split_name}_series_count"] = int(split_summary["series_count"])
            split_manifest[f"{split_name}_series_id_sha256"] = str(split_summary["series_id_sha256"])
            split_manifest[f"{split_name}_window_count"] = int(split_summary["window_count"])
        manifests.append(split_manifest)
    per_seed = pd.DataFrame(rows).sort_values(["label", "split_seed"]).reset_index(drop=True)
    counts = per_seed.groupby("label")["split_seed"].nunique()
    if not (counts == len(seeds)).all():
        raise ValueError(f"Each label must have {len(seeds)} resplits; observed {counts.to_dict()}")
    return per_seed, pd.DataFrame(manifests).sort_values("split_seed").reset_index(drop=True)


def load_moment_dynamic(runs_root: Path, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split_seed in seeds:
        result_path = (
            runs_root
            / f"seed_{split_seed}"
            / "moment"
            / "dynamic_probes"
            / "probe_results.csv"
        )
        if not result_path.exists():
            raise FileNotFoundError(f"Missing MOMENT dynamic probes for seed {split_seed}: {result_path}")
        frame = pd.read_csv(result_path)
        if set(frame["seed"].astype(int)) != {split_seed}:
            raise ValueError(f"{result_path} contains inconsistent seed values")
        for label in DYNAMIC_LABELS:
            label_frame = frame[frame["label"] == label].dropna(subset=["val_r2"])
            if label_frame.empty:
                raise ValueError(f"{result_path} has no validation-scored row for {label}")
            selected = label_frame.sort_values(
                ["val_r2", "layer", "token_position", "pooling_mode"],
                ascending=[False, True, True, True],
            ).iloc[0]
            rows.append(
                {
                    "split_seed": split_seed,
                    "label": label,
                    "layer": int(selected["layer"]),
                    "token_position": str(selected["token_position"]),
                    "pooling_mode": str(selected["pooling_mode"]),
                    "val_r2": float(selected["val_r2"]),
                    "test_r2": float(selected["test_r2"]),
                    "raw_test_r2": float(selected["baseline_test_r2"]),
                    "shuffled_test_r2": float(selected["shuffled_test_r2"]),
                    "moment_minus_raw": float(
                        selected["test_r2"] - selected["baseline_test_r2"]
                    ),
                    "moment_minus_shuffled": float(
                        selected["test_r2"] - selected["shuffled_test_r2"]
                    ),
                }
            )
    per_seed = pd.DataFrame(rows).sort_values(["label", "split_seed"]).reset_index(drop=True)
    counts = per_seed.groupby("label")["split_seed"].nunique()
    if not (counts == len(seeds)).all():
        raise ValueError(f"Each dynamic label must have {len(seeds)} resplits; observed {counts.to_dict()}")
    return per_seed


def load_moment_random(
    runs_root: Path,
    seeds: list[int],
    *,
    label_group: str,
) -> pd.DataFrame:
    if label_group == "taxonomy":
        labels = STRUCTURAL_LABELS
        probe_dir = "taxonomy_probes"
        validation_column = "val_macro_f1"
        test_column = "test_macro_f1"
    elif label_group == "dynamic":
        labels = DYNAMIC_LABELS
        probe_dir = "dynamic_probes"
        validation_column = "val_r2"
        test_column = "test_r2"
    else:
        raise ValueError(f"Unsupported MOMENT random-control label group: {label_group}")

    rows: list[dict[str, object]] = []
    for split_seed in seeds:
        seed_root = runs_root / f"seed_{split_seed}" / "moment_random"
        result_path = (
            seed_root
            / probe_dir
            / "probe_results.csv"
        )
        summary_path = seed_root / "activations" / "activation_dump_summary.json"
        if not result_path.exists():
            raise FileNotFoundError(
                f"Missing MOMENT random-init {label_group} probes for seed "
                f"{split_seed}: {result_path}"
            )
        summary = json.loads(summary_path.read_text())
        if int(summary.get("seed", -1)) != split_seed:
            raise ValueError(f"{summary_path} records the wrong random-init seed")
        if summary.get("weight_source") != "random_init":
            raise ValueError(f"{summary_path} is not a random-init MOMENT run")
        if summary.get("model_training") is not False:
            raise ValueError(f"{summary_path} does not record evaluation mode")
        if any(int(value) for value in summary.get("split_overlap_counts", {}).values()):
            raise ValueError(f"{summary_path} records overlapping BOOM series")
        enrichment = summary.get("provenance_enrichment", {})
        if (
            enrichment.get("measurement_values_unchanged") is not True
            or enrichment.get("weight_source") != "random_init"
        ):
            raise ValueError(f"{summary_path} lacks audited random-init provenance")
        frame = pd.read_csv(result_path)
        if set(frame["seed"].astype(int)) != {split_seed}:
            raise ValueError(f"{result_path} contains inconsistent seed values")
        if set(frame["weight_source"].astype(str)) != {"random_init"}:
            raise ValueError(f"{result_path} is not a random-initialization control")
        for label in labels:
            label_frame = frame[frame["label"] == label].dropna(
                subset=[validation_column]
            )
            if label_frame.empty:
                raise ValueError(
                    f"{result_path} has no validation-scored row for {label}"
                )
            selected = label_frame.sort_values(
                [validation_column, "layer", "token_position", "pooling_mode"],
                ascending=[False, True, True, True],
            ).iloc[0]
            rows.append(
                {
                    "split_seed": split_seed,
                    "label": label,
                    "random_layer": int(selected["layer"]),
                    "random_token_position": str(selected["token_position"]),
                    "random_pooling_mode": str(selected["pooling_mode"]),
                    "random_validation_score": float(selected[validation_column]),
                    "random_test_score": float(selected[test_column]),
                }
            )
    per_seed = pd.DataFrame(rows).sort_values(["label", "split_seed"]).reset_index(
        drop=True
    )
    counts = per_seed.groupby("label")["split_seed"].nunique()
    if not (counts == len(seeds)).all():
        raise ValueError(
            f"Each MOMENT random-init label must have {len(seeds)} resplits; "
            f"observed {counts.to_dict()}"
        )
    return per_seed


def load_transfer(runs_root: Path, seeds: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_dataset_rows: list[pd.DataFrame] = []
    per_resplit_rows: list[dict[str, object]] = []
    expected_datasets: set[tuple[str, str]] | None = None
    for split_seed in seeds:
        transfer_root = runs_root / f"seed_{split_seed}" / "transfer"
        result_path = transfer_root / "transfer_probe_metrics.csv"
        meta_path = transfer_root / "transfer_meta.json"
        if not result_path.exists() or not meta_path.exists():
            raise FileNotFoundError(
                f"Incomplete transfer run for seed {split_seed}: {transfer_root}"
            )
        meta = json.loads(meta_path.read_text())
        if meta.get("model_training") is not False or not meta.get(
            "require_eval_provenance"
        ):
            raise ValueError(f"{meta_path} does not enforce eval-mode provenance")
        if (
            meta.get("probe_selection_rule")
            != "max_boom_validation_r2_per_label_with_deterministic_tie_break"
        ):
            raise ValueError(f"{meta_path} does not record BOOM-validation-only probe selection")
        if (
            meta.get("external_window_eligibility_rule")
            != "finite_context_and_next_patch_complete_case"
        ):
            raise ValueError(f"{meta_path} does not record the external complete-case rule")
        frame = pd.read_csv(result_path)
        if set(frame["probe_seed"].astype(int)) != {split_seed}:
            raise ValueError(f"{result_path} contains a probe from the wrong BOOM resplit")
        dataset_set = set(
            frame[["benchmark", "dataset_name"]].itertuples(index=False, name=None)
        )
        if expected_datasets is None:
            expected_datasets = dataset_set
        elif dataset_set != expected_datasets:
            raise ValueError(
                f"{result_path} evaluates a different external dataset set from the other resplits"
            )
        seed_frame = frame.copy()
        seed_frame.insert(0, "split_seed", split_seed)
        per_dataset_rows.append(seed_frame)
        for (benchmark, label), group in frame.groupby(
            ["benchmark", "probe_label"], dropna=False
        ):
            per_resplit_rows.append(
                {
                    "split_seed": split_seed,
                    "benchmark": benchmark,
                    "label": label,
                    "dataset_count": int(group["dataset_name"].nunique()),
                    "transfer_r2_macro": float(group["transfer_r2"].mean()),
                    "raw_transfer_r2_macro": float(
                        group["baseline_transfer_r2"].mean()
                    ),
                    "transfer_minus_raw_r2_macro": float(
                        (group["transfer_r2"] - group["baseline_transfer_r2"]).mean()
                    ),
                    "positive_dataset_count": int((group["transfer_r2"] > 0).sum()),
                }
            )
    per_dataset = pd.concat(per_dataset_rows, ignore_index=True).sort_values(
        ["benchmark", "probe_label", "dataset_name", "split_seed"]
    )
    per_resplit = pd.DataFrame(per_resplit_rows).sort_values(
        ["benchmark", "label", "split_seed"]
    )
    counts = per_resplit.groupby(["benchmark", "label"])["split_seed"].nunique()
    if not (counts == len(seeds)).all():
        raise ValueError(
            f"Each transfer cell must have {len(seeds)} resplits; observed {counts.to_dict()}"
        )
    return per_resplit.reset_index(drop=True), per_dataset.reset_index(drop=True)


def load_moment_interchange(
    runs_root: Path, seeds: list[int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    manifests: list[dict[str, object]] = []
    expected_secondary_availability: bool | None = None
    for split_seed in seeds:
        root = runs_root / f"seed_{split_seed}" / "moment" / "interchange"
        result_path = root / "moment_interchange_results.csv"
        meta_path = root / "moment_interchange_meta.json"
        pair_path = root / "moment_interchange_pairs.csv"
        if not result_path.exists() or not meta_path.exists() or not pair_path.exists():
            raise FileNotFoundError(
                f"Incomplete MOMENT interchange for seed {split_seed}: {root}"
            )
        meta = json.loads(meta_path.read_text())
        expected = {
            "split_seed": split_seed,
            "sampling_seed": split_seed,
            "probe_seed": split_seed,
            "model_training": False,
            "split": "test",
            "weight_source": "pretrained",
            "primary_endpoint": "probe_score_real_minus_null_within_target",
            "probe_selection_rule": "validation_selected",
            "unique_exact_pair_triples": True,
            "same_pair_triples_across_blends": True,
            "common_targets_across_conditions": True,
            "cross_series_sources": True,
            "exact_channel_count_matching": True,
            "implicit_source_broadcasting": False,
        }
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(
                    f"{meta_path} records {key}={meta.get(key)!r}, expected {value!r}"
                )
        if any(int(value) for value in meta.get("split_overlap_counts", {}).values()):
            raise ValueError(f"{meta_path} records overlapping BOOM series splits")
        signature = meta.get("probe_activation_source")
        if not isinstance(signature, dict) or signature.get("model_training") is not False:
            raise ValueError(f"{meta_path} has no eval-mode probe activation signature")
        if int(signature.get("seed", -1)) != split_seed:
            raise ValueError(f"{meta_path} probe activation seed does not match its resplit")
        signature_hashes = signature.get("series_id_sha256_by_split", {})
        for split_name in ("train", "val", "test"):
            if signature_hashes.get(split_name) != meta.get("split_manifest", {}).get(
                split_name, {}
            ).get("series_id_sha256"):
                raise ValueError(
                    f"{meta_path} probe and intervention disagree on {split_name}"
                )

        secondary_available = bool(meta.get("secondary_endpoint_available"))
        if expected_secondary_availability is None:
            expected_secondary_availability = secondary_available
        elif secondary_available != expected_secondary_availability:
            raise ValueError("MOMENT secondary endpoint availability differs across resplits")

        frame = pd.read_csv(result_path)
        key = ["split_seed", "sampling_seed", "pair_id", "blend", "condition"]
        if frame.duplicated(key).any():
            raise ValueError(f"{result_path} contains duplicate pair-condition rows")
        if set(frame["split_seed"].astype(int)) != {split_seed} or set(
            frame["sampling_seed"].astype(int)
        ) != {split_seed}:
            raise ValueError(f"{result_path} contains inconsistent seed values")
        blends = tuple(float(value) for value in meta["blends"])
        if set(frame["blend"].astype(float)) != set(blends):
            raise ValueError(f"{result_path} does not contain the predeclared blend set")
        grouped = frame.groupby(["pair_id", "blend"])["condition"].agg(set)
        if not grouped.map(
            lambda value: value == {"clean", "real_patch", "null_patch"}
        ).all():
            raise ValueError(f"{result_path} contains incomplete clean/real/null triples")
        pair_count = int(meta["num_pairs_per_seed"])
        if frame["pair_id"].nunique() != pair_count:
            raise ValueError(
                f"{result_path} has {frame['pair_id'].nunique()} pairs, expected {pair_count}"
            )
        if not frame.groupby("pair_id")["target_window_id"].nunique().eq(1).all():
            raise ValueError(f"{result_path} changes targets across conditions or blends")
        sources = (
            frame[frame["condition"] != "clean"]
            .groupby(["pair_id", "condition"])["source_window_id"]
            .nunique()
        )
        if not sources.eq(1).all():
            raise ValueError(f"{result_path} changes donors across blends")

        pairs = pd.read_csv(pair_path)
        if len(pairs) != pair_count or pairs["pair_id"].nunique() != pair_count:
            raise ValueError(f"{pair_path} does not contain one row per exact triple")
        if pairs.duplicated(
            [
                "target_window_id",
                "real_source_window_id",
                "null_source_window_id",
            ]
        ).any():
            raise ValueError(f"{pair_path} contains duplicate exact triples")
        if (
            (pairs["target_series_id"] == pairs["real_source_series_id"]).any()
            or (pairs["target_series_id"] == pairs["null_source_series_id"]).any()
            or (pairs["real_source_series_id"] == pairs["null_source_series_id"]).any()
        ):
            raise ValueError(f"{pair_path} violates cross-series donor matching")

        pivot = frame.pivot(
            index=["split_seed", "sampling_seed", "pair_id", "blend"],
            columns="condition",
            values=["probe_score", "future_mae"],
        )
        for blend, group in pivot.groupby(level="blend", sort=True):
            probe_difference = (
                group[("probe_score", "real_patch")]
                - group[("probe_score", "null_patch")]
            )
            future_difference = (
                group[("future_mae", "real_patch")]
                - group[("future_mae", "null_patch")]
            )
            rows.append(
                {
                    "split_seed": split_seed,
                    "blend": float(blend),
                    "n_pairs": int(len(group)),
                    "probe_wins": int((probe_difference > 0).sum()),
                    "probe_losses": int((probe_difference < 0).sum()),
                    "probe_ties": int((probe_difference == 0).sum()),
                    "probe_win_fraction": float((probe_difference > 0).mean()),
                    "probe_real_minus_null_median": float(probe_difference.median()),
                    "secondary_endpoint_available": secondary_available,
                    "future_mae_real_minus_null_median": (
                        float(future_difference.median())
                        if secondary_available
                        else float("nan")
                    ),
                    "future_mae_real_better_fraction": (
                        float((future_difference < 0).mean())
                        if secondary_available
                        else float("nan")
                    ),
                }
            )
        manifests.append(
            {
                "split_seed": split_seed,
                "sampling_seed": int(meta["sampling_seed"]),
                "probe_layer": int(meta["probe_layer"]),
                "probe_token_position": str(meta["probe_token_position"]),
                "probe_pooling_mode": str(meta["probe_pooling_mode"]),
                "probe_selection_score": float(meta["probe_selection_score"]),
                "pair_triples_sha256": str(meta["pair_triples_sha256"]),
                "test_series_id_sha256": str(meta["series_id_sha256"]),
                "secondary_endpoint_available": secondary_available,
                "secondary_endpoint_omission_reason": meta.get(
                    "secondary_endpoint_omission_reason"
                ),
            }
        )
    per_seed = pd.DataFrame(rows).sort_values(["blend", "split_seed"]).reset_index(
        drop=True
    )
    counts = per_seed.groupby("blend")["split_seed"].nunique()
    if not (counts == len(seeds)).all():
        raise ValueError(
            f"Each MOMENT interchange blend must have {len(seeds)} resplits; "
            f"observed {counts.to_dict()}"
        )
    return per_seed, pd.DataFrame(manifests).sort_values("split_seed").reset_index(
        drop=True
    )


def render_markdown(
    patch_per_seed: pd.DataFrame,
    patch_summary: pd.DataFrame,
    moment_per_seed: pd.DataFrame,
    moment_summary: pd.DataFrame,
    moment_dynamic_per_seed: pd.DataFrame,
    moment_dynamic_summary: pd.DataFrame,
    moment_random_structural_summary: pd.DataFrame,
    moment_random_dynamic_summary: pd.DataFrame,
    moment_interchange_per_seed: pd.DataFrame,
    moment_interchange_summary: pd.DataFrame,
    transfer_per_resplit: pd.DataFrame,
    transfer_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Five-Resplit Replications\n\n",
        "All ± values are Student-t 95% half-widths across five seeded, "
        "series-disjoint BOOM resplits (seeds 42-46); they measure variability across "
        "resplits of one corpus and are descriptive rather than significance tests.\n\n",
        "## Toto donor exchange (future burstiness)\n\n",
        "Each resplit fits the fixed L11/all-context/series-mean probe on its training "
        "series and evaluates 40 paired high-burst/randomized-donor source patches "
        "on held-out test series. The randomized donor is from another series but "
        "is not covariate- or taxonomy-matched. Probe wins give the probe check; "
        "forecast wins (forecast burstier under the high-burst than under the "
        "randomized donor) give the forecast endpoint; the median WAPE ratio "
        "(high-burst/randomized) is the secondary endpoint. Pair observations are summarized "
        "within split before aggregation; the five data resplits are the "
        "replication units.\n\n",
        "| Blend | Forecast wins/losses/ties | Probe wins/losses/ties | Median forecast difference | "
        "Median probe difference | Median WAPE ratio | Forecast wins > .5 | Probe wins > .5 |\n",
        "|---:|---:|---:|---:|---:|---:|---:|---:|\n",
    ]
    for _, row in patch_summary.sort_values("blend").iterrows():
        blend_frame = patch_per_seed[patch_per_seed["blend"] == row["blend"]]
        lines.append(
            f"| {row['blend']:.2f} | "
            f"{int(blend_frame['burst_wins'].sum())}/"
            f"{int(blend_frame['burst_losses'].sum())}/"
            f"{int(blend_frame['burst_ties'].sum())} | "
            f"{int(blend_frame['probe_wins'].sum())}/"
            f"{int(blend_frame['probe_losses'].sum())}/"
            f"{int(blend_frame['probe_ties'].sum())} | "
            f"{row['burst_real_minus_null_median_mean']:.3f} ± "
            f"{row['burst_real_minus_null_median_ci95_half_width']:.3f} | "
            f"{row['probe_real_minus_null_median_mean']:.3f} ± "
            f"{row['probe_real_minus_null_median_ci95_half_width']:.3f} | "
            f"{row['wape_real_over_null_median_mean']:.3f} ± "
            f"{row['wape_real_over_null_median_ci95_half_width']:.3f} | "
            f"{int((blend_frame['burst_win_fraction'] > 0.5).sum())}/5 | "
            f"{int((blend_frame['probe_win_fraction'] > 0.5).sum())}/5 |\n"
        )
    probe_once = patch_per_seed.sort_values(["split_seed", "blend"]).drop_duplicates("split_seed")
    lines.extend(
        [
            "\nFixed-view future-burstiness probe quality across the same resplits: "
            f"test R2 {probe_once['probe_test_r2'].mean():.3f} ± "
            f"{interval_half_width(probe_once['probe_test_r2']):.3f}; raw six-statistic "
            f"{probe_once['probe_raw_test_r2'].mean():.3f} ± "
            f"{interval_half_width(probe_once['probe_raw_test_r2']):.3f}; shuffled "
            f"{probe_once['probe_shuffled_test_r2'].mean():.3f} ± "
            f"{interval_half_width(probe_once['probe_shuffled_test_r2']):.3f}.\n",
            f"The fixed view beats the raw six-statistic baseline in "
            f"{int((probe_once['probe_minus_raw_r2'] > 0).sum())}/5 resplits and "
            f"the shuffled-label control in "
            f"{int((probe_once['probe_minus_shuffled_r2'] > 0).sum())}/5.\n",
        ]
    )
    lines.extend(
        [
            "\n## MOMENT-base structural recurrence\n\n",
            "Within each resplit, the view is selected only by validation macro-F1 and "
            "evaluated once on held-out test series.\n\n",
            "| Label | MOMENT test macro-F1 | Random-init | Raw six-stat. | Shuffled | "
            "MOMENT-raw gap / wins | MOMENT-shuffled gap / wins |\n",
            "|---|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for _, row in moment_summary.sort_values("label").iterrows():
        label_frame = moment_per_seed[moment_per_seed["label"] == row["label"]]
        random_row = moment_random_structural_summary[
            moment_random_structural_summary["label"] == row["label"]
        ].iloc[0]
        lines.append(
            f"| {row['label']} | "
            f"{row['test_macro_f1_mean']:.3f} ± {row['test_macro_f1_ci95_half_width']:.3f} | "
            f"{random_row['random_test_score_mean']:.3f} ± "
            f"{random_row['random_test_score_ci95_half_width']:.3f} | "
            f"{row['raw_test_macro_f1_mean']:.3f} ± {row['raw_test_macro_f1_ci95_half_width']:.3f} | "
            f"{row['shuffled_test_macro_f1_mean']:.3f} ± {row['shuffled_test_macro_f1_ci95_half_width']:.3f} | "
            f"{row['moment_minus_raw_mean']:+.3f} ± {row['moment_minus_raw_ci95_half_width']:.3f}; "
            f"{int((label_frame['moment_minus_raw'] > 0).sum())}/5 | "
            f"{row['moment_minus_shuffled_mean']:+.3f} ± "
            f"{row['moment_minus_shuffled_ci95_half_width']:.3f}; "
            f"{int((label_frame['moment_minus_shuffled'] > 0).sum())}/5 |\n"
        )
    lines.extend(
        [
            "\n## MOMENT-base dynamic-axis readouts\n\n",
            "Within each seeded resplit, the representation view is selected by validation "
            "R2 and read once on held-out test series. This tests cross-backbone dynamic "
            "decodability; it is not itself a forecast-behavior intervention.\n\n",
            "| Label | MOMENT test R2 | Random-init | Raw six-stat. | Shuffled | "
            "MOMENT-raw gap / wins | MOMENT-shuffled gap / wins |\n",
            "|---|---:|---:|---:|---:|---:|---:|\n",
        ]
    )
    for _, row in moment_dynamic_summary.sort_values("label").iterrows():
        label_frame = moment_dynamic_per_seed[
            moment_dynamic_per_seed["label"] == row["label"]
        ]
        random_row = moment_random_dynamic_summary[
            moment_random_dynamic_summary["label"] == row["label"]
        ].iloc[0]
        lines.append(
            f"| {row['label']} | "
            f"{row['test_r2_mean']:.3f} ± {row['test_r2_ci95_half_width']:.3f} | "
            f"{random_row['random_test_score_mean']:.3f} ± "
            f"{random_row['random_test_score_ci95_half_width']:.3f} | "
            f"{row['raw_test_r2_mean']:.3f} ± {row['raw_test_r2_ci95_half_width']:.3f} | "
            f"{row['shuffled_test_r2_mean']:.3f} ± {row['shuffled_test_r2_ci95_half_width']:.3f} | "
            f"{row['moment_minus_raw_mean']:+.3f} ± {row['moment_minus_raw_ci95_half_width']:.3f}; "
            f"{int((label_frame['moment_minus_raw'] > 0).sum())}/5 | "
            f"{row['moment_minus_shuffled_mean']:+.3f} ± "
            f"{row['moment_minus_shuffled_ci95_half_width']:.3f}; "
            f"{int((label_frame['moment_minus_shuffled'] > 0).sum())}/5 |\n"
        )
    lines.extend(
        [
            "\n## MOMENT-base matched interchange\n\n",
            "The future-burstiness view is selected by BOOM validation R2 only. "
            "Each resplit uses 40 exact target/real/null triples on held-out test "
            "series, with the same triples at all blends. Probe wins give the probe "
            "check; the lower future-MAE fraction is the forecast endpoint.\n\n",
            "| Blend | Probe wins/losses/ties | Median probe difference | "
            "Resplits with win fraction > .5 | Lower future-MAE fraction |\n",
            "|---:|---:|---:|---:|---:|\n",
        ]
    )
    for _, row in moment_interchange_summary.sort_values("blend").iterrows():
        blend_frame = moment_interchange_per_seed[
            moment_interchange_per_seed["blend"] == row["blend"]
        ]
        secondary = (
            f"{row['future_mae_real_better_fraction_mean']:.3f} ± "
            f"{row['future_mae_real_better_fraction_ci95_half_width']:.3f}"
            if blend_frame["secondary_endpoint_available"].all()
            else "unavailable in installed pretrained API"
        )
        lines.append(
            f"| {row['blend']:.2f} | "
            f"{int(blend_frame['probe_wins'].sum())}/"
            f"{int(blend_frame['probe_losses'].sum())}/"
            f"{int(blend_frame['probe_ties'].sum())} | "
            f"{row['probe_real_minus_null_median_mean']:+.3f} ± "
            f"{row['probe_real_minus_null_median_ci95_half_width']:.3f} | "
            f"{int((blend_frame['probe_win_fraction'] > 0.5).sum())}/5 | "
            f"{secondary} |\n"
        )
    lines.extend(
        [
            "\n## Five-resplit external transfer\n\n",
            "Each row macro-averages the fixed external datasets within a BOOM resplit, "
            "then summarizes the five BOOM-trained probe sets. No target dataset is selected "
            "post hoc. Several external labels are nearly constant, which makes their R2 "
            "unstable; the paper reports coordination, and the remaining labels are "
            "listed for completeness.\n\n",
            "| Benchmark | Label | Transfer R2 | Raw six-stat. R2 | Transfer-raw gap / wins |\n",
            "|---|---|---:|---:|---:|\n",
        ]
    )
    for _, row in transfer_summary.sort_values(["benchmark", "label"]).iterrows():
        cell = transfer_per_resplit[
            (transfer_per_resplit["benchmark"] == row["benchmark"])
            & (transfer_per_resplit["label"] == row["label"])
        ]
        lines.append(
            f"| {row['benchmark']} | {row['label']} | "
            f"{row['transfer_r2_macro_mean']:.3f} ± "
            f"{row['transfer_r2_macro_ci95_half_width']:.3f} | "
            f"{row['raw_transfer_r2_macro_mean']:.3f} ± "
            f"{row['raw_transfer_r2_macro_ci95_half_width']:.3f} | "
            f"{row['transfer_minus_raw_r2_macro_mean']:+.3f} ± "
            f"{row['transfer_minus_raw_r2_macro_ci95_half_width']:.3f}; "
            f"{int((cell['transfer_minus_raw_r2_macro'] > 0).sum())}/5 |\n"
        )
    return "".join(lines)


def main() -> None:
    args = parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError(f"Duplicate seeds are not allowed: {args.seeds}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    patch_runs_root = args.patch_runs_root or args.runs_root
    moment_runs_root = args.moment_runs_root or args.runs_root

    patch_per_seed, patch_manifest = load_paired_patch(patch_runs_root, args.seeds)
    patch_values = [
        "probe_test_r2",
        "probe_raw_test_r2",
        "probe_shuffled_test_r2",
        "probe_minus_raw_r2",
        "probe_minus_shuffled_r2",
        "burst_win_fraction",
        "probe_win_fraction",
        "burst_real_minus_null_median",
        "probe_real_minus_null_median",
        "wape_real_over_null_median",
    ]
    patch_summary = aggregate_columns(patch_per_seed, ["blend"], patch_values)

    moment_per_seed, moment_manifest = load_moment(moment_runs_root, args.seeds)
    moment_values = [
        "test_macro_f1",
        "raw_test_macro_f1",
        "shuffled_test_macro_f1",
        "moment_minus_raw",
        "moment_minus_shuffled",
    ]
    moment_summary = aggregate_columns(moment_per_seed, ["label"], moment_values)

    moment_dynamic_per_seed = load_moment_dynamic(moment_runs_root, args.seeds)
    moment_dynamic_summary = aggregate_columns(
        moment_dynamic_per_seed,
        ["label"],
        [
            "test_r2",
            "raw_test_r2",
            "shuffled_test_r2",
            "moment_minus_raw",
            "moment_minus_shuffled",
        ],
    )
    moment_random_structural = load_moment_random(
        moment_runs_root, args.seeds, label_group="taxonomy"
    )
    moment_random_dynamic = load_moment_random(
        moment_runs_root, args.seeds, label_group="dynamic"
    )
    moment_random_structural_summary = aggregate_columns(
        moment_random_structural, ["label"], ["random_test_score"]
    )
    moment_random_dynamic_summary = aggregate_columns(
        moment_random_dynamic, ["label"], ["random_test_score"]
    )
    moment_interchange_per_seed, moment_interchange_manifest = (
        load_moment_interchange(moment_runs_root, args.seeds)
    )
    moment_interchange_summary = aggregate_columns(
        moment_interchange_per_seed,
        ["blend"],
        [
            "probe_win_fraction",
            "probe_real_minus_null_median",
            "future_mae_real_minus_null_median",
            "future_mae_real_better_fraction",
        ],
    )
    transfer_per_resplit, transfer_per_dataset = load_transfer(
        patch_runs_root, args.seeds
    )
    transfer_summary = aggregate_columns(
        transfer_per_resplit,
        ["benchmark", "label"],
        [
            "transfer_r2_macro",
            "raw_transfer_r2_macro",
            "transfer_minus_raw_r2_macro",
        ],
    )

    patch_per_seed.to_csv(args.output_dir / "paired_patch_per_resplit.csv", index=False)
    patch_summary.to_csv(args.output_dir / "paired_patch_summary.csv", index=False)
    patch_manifest.to_csv(args.output_dir / "paired_patch_manifest.csv", index=False)
    moment_per_seed.to_csv(args.output_dir / "moment_per_resplit.csv", index=False)
    moment_summary.to_csv(args.output_dir / "moment_summary.csv", index=False)
    moment_manifest.to_csv(args.output_dir / "moment_manifest.csv", index=False)
    moment_dynamic_per_seed.to_csv(
        args.output_dir / "moment_dynamic_per_resplit.csv", index=False
    )
    moment_dynamic_summary.to_csv(
        args.output_dir / "moment_dynamic_summary.csv", index=False
    )
    moment_random_structural.to_csv(
        args.output_dir / "moment_random_structural_per_resplit.csv", index=False
    )
    moment_random_structural_summary.to_csv(
        args.output_dir / "moment_random_structural_summary.csv", index=False
    )
    moment_random_dynamic.to_csv(
        args.output_dir / "moment_random_dynamic_per_resplit.csv", index=False
    )
    moment_random_dynamic_summary.to_csv(
        args.output_dir / "moment_random_dynamic_summary.csv", index=False
    )
    moment_interchange_per_seed.to_csv(
        args.output_dir / "moment_interchange_per_resplit.csv", index=False
    )
    moment_interchange_summary.to_csv(
        args.output_dir / "moment_interchange_summary.csv", index=False
    )
    moment_interchange_manifest.to_csv(
        args.output_dir / "moment_interchange_manifest.csv", index=False
    )
    transfer_per_resplit.to_csv(
        args.output_dir / "transfer_per_resplit.csv", index=False
    )
    transfer_per_dataset.to_csv(
        args.output_dir / "transfer_per_dataset.csv", index=False
    )
    transfer_summary.to_csv(args.output_dir / "transfer_summary.csv", index=False)
    (args.output_dir / "MOMENT_EXCHANGE_TRANSFER_RESULTS.md").write_text(
        render_markdown(
            patch_per_seed,
            patch_summary,
            moment_per_seed,
            moment_summary,
            moment_dynamic_per_seed,
            moment_dynamic_summary,
            moment_random_structural_summary,
            moment_random_dynamic_summary,
            moment_interchange_per_seed,
            moment_interchange_summary,
            transfer_per_resplit,
            transfer_summary,
        )
    )


if __name__ == "__main__":
    main()
