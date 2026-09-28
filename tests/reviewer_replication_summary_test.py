from __future__ import annotations

import json

import pandas as pd

from scripts.summarize_reviewer_replications import (
    DYNAMIC_LABELS,
    STRUCTURAL_LABELS,
    aggregate_columns,
    load_moment,
    load_moment_dynamic,
    load_moment_interchange,
    load_moment_random,
    load_paired_patch,
    load_transfer,
)


def _write_patch_seed(root, split_seed: int) -> None:
    seed_root = root / f"seed_{split_seed}"
    probe_output = seed_root / "future_burstiness_probe"
    probe_output.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "label": "future_burstiness",
                "layer": 11,
                "token_position": "all_context",
                "pooling_mode": "series_mean",
                "test_r2": 0.2,
                "baseline_test_r2": 0.1,
                "shuffled_test_r2": -0.1,
            }
        ]
    ).to_csv(probe_output / "probe_results.csv", index=False)

    output = seed_root / "paired_patch_blend_0p25"
    output.mkdir(parents=True)
    rows = []
    split_hashes = {
        split: f"{split}-{split_seed}" for split in ("train", "val", "test")
    }
    for pair_id in range(2):
        for condition, burst, score, wape in (
            ("clean", 1.0, 1.0, 1.0),
            ("real_patch", 4.0 + pair_id, 5.0 + pair_id, 2.0),
            ("null_patch", 2.0 + pair_id, 3.0 + pair_id, 2.0),
        ):
            rows.append(
                {
                    "split_seed": split_seed,
                    "seed": split_seed,
                    "pair_id": pair_id,
                    "condition": condition,
                    "target_series_id": f"target-{pair_id}",
                    "source_series_id": (
                        "(none)"
                        if condition == "clean"
                        else f"real-{pair_id}"
                        if condition == "real_patch"
                        else f"null-{pair_id}"
                    ),
                    "forecast_burstiness": burst,
                    "probe_score": score,
                    "wape": wape,
                }
            )
    pd.DataFrame(rows).to_csv(output / "paired_patch_results.csv", index=False)
    (output / "paired_patch_meta.json").write_text(
        json.dumps(
            {
                "split_seed": split_seed,
                "probe_seed": split_seed,
                "split": "test",
                "blend": 0.25,
                "model_training": False,
                "common_forecast_random_numbers": True,
                "cross_series_sources": True,
                "split_overlap_counts": {"train_val": 0, "train_test": 0, "val_test": 0},
                "split_manifest": {
                    split: {"series_id_sha256": value}
                    for split, value in split_hashes.items()
                },
                "probe_activation_source_signature": json.dumps(
                    {
                        "seed": split_seed,
                        "model_training": False,
                        "series_id_sha256_by_split": split_hashes,
                    }
                ),
                "probe_sha256": f"probe-{split_seed}",
                "series_count": 100,
                "series_id_sha256": split_hashes["test"],
                "context_length": 1024,
                "num_samples": 16,
                "num_pairs_per_seed": 2,
                "n_windows": 200,
            }
        )
    )


def _write_moment_seed(root, split_seed: int) -> None:
    output = root / f"seed_{split_seed}" / "moment"
    (output / "probes").mkdir(parents=True)
    (output / "activations").mkdir(parents=True)
    rows = []
    for label_index, label in enumerate(STRUCTURAL_LABELS):
        rows.extend(
            [
                {
                    "seed": split_seed,
                    "label": label,
                    "layer": 3,
                    "token_position": "all_context",
                    "pooling_mode": "series_mean",
                    "val_macro_f1": 0.5 + label_index * 0.01,
                    "test_macro_f1": 0.4 + label_index * 0.01,
                    "baseline_test_macro_f1": 0.3,
                    "shuffled_test_macro_f1": 0.2,
                },
                {
                    "seed": split_seed,
                    "label": label,
                    "layer": 9,
                    "token_position": "all_context",
                    "pooling_mode": "series_mean",
                    "val_macro_f1": 0.4,
                    "test_macro_f1": 0.9,
                    "baseline_test_macro_f1": 0.3,
                    "shuffled_test_macro_f1": 0.2,
                },
            ]
        )
    pd.DataFrame(rows).to_csv(output / "probes" / "probe_results.csv", index=False)
    dynamic_rows = []
    for label_index, label in enumerate(DYNAMIC_LABELS):
        dynamic_rows.extend(
            [
                {
                    "seed": split_seed,
                    "label": label,
                    "layer": 3,
                    "token_position": "all_context",
                    "pooling_mode": "series_mean",
                    "val_r2": 0.5 + label_index * 0.01,
                    "test_r2": 0.4 + label_index * 0.01,
                    "baseline_test_r2": 0.3,
                    "shuffled_test_r2": -0.1,
                },
                {
                    "seed": split_seed,
                    "label": label,
                    "layer": 9,
                    "token_position": "all_context",
                    "pooling_mode": "series_mean",
                    "val_r2": 0.4,
                    "test_r2": 0.9,
                    "baseline_test_r2": 0.3,
                    "shuffled_test_r2": -0.1,
                },
            ]
        )
    (output / "dynamic_probes").mkdir()
    pd.DataFrame(dynamic_rows).to_csv(
        output / "dynamic_probes" / "probe_results.csv", index=False
    )
    random_output = root / f"seed_{split_seed}" / "moment_random"
    (random_output / "taxonomy_probes").mkdir(parents=True)
    (random_output / "dynamic_probes").mkdir()
    (random_output / "activations").mkdir()
    random_structural = pd.DataFrame(rows).assign(weight_source="random_init")
    random_structural.to_csv(
        random_output / "taxonomy_probes" / "probe_results.csv", index=False
    )
    random_dynamic = pd.DataFrame(dynamic_rows).assign(weight_source="random_init")
    random_dynamic.to_csv(
        random_output / "dynamic_probes" / "probe_results.csv", index=False
    )
    split_summary = {
        split: {
            "series_count": 100,
            "series_id_sha256": f"{split}-{split_seed}",
            "window_count": 200,
        }
        for split in ("train", "val", "test")
    }
    (output / "activations" / "activation_dump_summary.json").write_text(
        json.dumps(
            {
                "seed": split_seed,
                "weight_source": "pretrained",
                "model_training": False,
                "split_overlap_counts": {
                    "train_val": 0,
                    "train_test": 0,
                    "val_test": 0,
                },
                "provenance_enrichment": {
                    "measurement_values_unchanged": True,
                    "weight_source": "pretrained",
                },
                "splits": split_summary,
            }
        )
    )
    (random_output / "activations" / "activation_dump_summary.json").write_text(
        json.dumps(
            {
                "seed": split_seed,
                "weight_source": "random_init",
                "model_training": False,
                "split_overlap_counts": {
                    "train_val": 0,
                    "train_test": 0,
                    "val_test": 0,
                },
                "provenance_enrichment": {
                    "measurement_values_unchanged": True,
                    "weight_source": "random_init",
                },
                "splits": split_summary,
            }
        )
    )


def _write_transfer_seed(root, split_seed: int) -> None:
    output = root / f"seed_{split_seed}" / "transfer"
    output.mkdir(parents=True)
    pd.DataFrame(
        [
            {
                "benchmark": benchmark,
                "dataset_name": dataset,
                "probe_label": "future_burstiness",
                "probe_seed": split_seed,
                "transfer_r2": score,
                "baseline_transfer_r2": score - 0.1,
            }
            for benchmark, dataset, score in (
                ("fev", "external_a", 0.2),
                ("fev", "external_b", 0.4),
                ("lsf", "external_c", -0.2),
            )
        ]
    ).to_csv(output / "transfer_probe_metrics.csv", index=False)
    (output / "transfer_meta.json").write_text(
        json.dumps(
            {
                "model_training": False,
                "require_eval_provenance": True,
                "probe_selection_rule": "max_boom_validation_r2_per_label_with_deterministic_tie_break",
                "external_window_eligibility_rule": "finite_context_and_next_patch_complete_case",
            }
        )
    )


def _write_moment_interchange_seed(root, split_seed: int) -> None:
    output = root / f"seed_{split_seed}" / "moment" / "interchange"
    output.mkdir(parents=True)
    rows = []
    for pair_id in range(2):
        for blend in (0.25, 0.5, 1.0):
            for condition, score in (
                ("clean", 0.0),
                ("real_patch", 2.0 + pair_id),
                ("null_patch", 1.0 + pair_id),
            ):
                rows.append(
                    {
                        "split_seed": split_seed,
                        "sampling_seed": split_seed,
                        "pair_id": pair_id,
                        "blend": blend,
                        "condition": condition,
                        "target_window_id": f"target-window-{pair_id}",
                        "source_window_id": (
                            ""
                            if condition == "clean"
                            else f"{condition}-window-{pair_id}"
                        ),
                        "probe_score": score,
                        "future_mae": float("nan"),
                    }
                )
    pd.DataFrame(rows).to_csv(
        output / "moment_interchange_results.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "pair_id": pair_id,
                "target_series_id": f"target-{pair_id}",
                "target_window_id": f"target-window-{pair_id}",
                "real_source_series_id": f"real-{pair_id}",
                "real_source_window_id": f"real_patch-window-{pair_id}",
                "null_source_series_id": f"null-{pair_id}",
                "null_source_window_id": f"null_patch-window-{pair_id}",
            }
            for pair_id in range(2)
        ]
    ).to_csv(output / "moment_interchange_pairs.csv", index=False)
    split_hashes = {
        split: f"{split}-{split_seed}" for split in ("train", "val", "test")
    }
    (output / "moment_interchange_meta.json").write_text(
        json.dumps(
            {
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
                "split_overlap_counts": {
                    "train_val": 0,
                    "train_test": 0,
                    "val_test": 0,
                },
                "split_manifest": {
                    split: {"series_id_sha256": value}
                    for split, value in split_hashes.items()
                },
                "probe_activation_source": {
                    "seed": split_seed,
                    "model_training": False,
                    "series_id_sha256_by_split": split_hashes,
                },
                "secondary_endpoint_available": False,
                "secondary_endpoint_omission_reason": "not_available",
                "blends": [0.25, 0.5, 1.0],
                "num_pairs_per_seed": 2,
                "probe_layer": 3,
                "probe_token_position": "all_context",
                "probe_pooling_mode": "series_mean",
                "probe_selection_score": 0.2,
                "pair_triples_sha256": f"pairs-{split_seed}",
                "series_id_sha256": split_hashes["test"],
            }
        )
    )


def test_five_resplit_aggregator_uses_split_level_units_and_validation_selection(tmp_path):
    seeds = [42, 43, 44, 45, 46]
    for seed in seeds:
        _write_patch_seed(tmp_path, seed)
        _write_moment_seed(tmp_path, seed)
        _write_moment_interchange_seed(tmp_path, seed)
        _write_transfer_seed(tmp_path, seed)

    patch_per_seed, patch_manifest = load_paired_patch(tmp_path, seeds)
    moment_per_seed, moment_manifest = load_moment(tmp_path, seeds)
    moment_dynamic_per_seed = load_moment_dynamic(tmp_path, seeds)
    moment_random_structural = load_moment_random(
        tmp_path, seeds, label_group="taxonomy"
    )
    moment_random_dynamic = load_moment_random(
        tmp_path, seeds, label_group="dynamic"
    )
    moment_interchange_per_seed, moment_interchange_manifest = (
        load_moment_interchange(tmp_path, seeds)
    )
    transfer_per_resplit, transfer_per_dataset = load_transfer(tmp_path, seeds)

    assert len(patch_per_seed) == 5
    assert patch_per_seed["burst_win_fraction"].eq(1.0).all()
    assert patch_per_seed["n_pairs"].eq(2).all()
    assert len(patch_manifest) == 5

    assert len(moment_per_seed) == 5 * len(STRUCTURAL_LABELS)
    assert moment_per_seed["layer"].eq(3).all()
    assert len(moment_manifest) == 5
    assert len(moment_dynamic_per_seed) == 5 * len(DYNAMIC_LABELS)
    assert moment_dynamic_per_seed["layer"].eq(3).all()
    assert len(moment_random_structural) == 5 * len(STRUCTURAL_LABELS)
    assert len(moment_random_dynamic) == 5 * len(DYNAMIC_LABELS)
    assert len(moment_interchange_per_seed) == 15
    assert moment_interchange_per_seed["probe_win_fraction"].eq(1.0).all()
    assert len(moment_interchange_manifest) == 5
    assert len(transfer_per_resplit) == 10
    assert len(transfer_per_dataset) == 15

    summary = aggregate_columns(patch_per_seed, ["blend"], ["burst_win_fraction"])
    assert summary.loc[0, "n_resplits"] == 5
    assert summary.loc[0, "burst_win_fraction_mean"] == 1.0
