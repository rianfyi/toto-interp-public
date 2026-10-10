from __future__ import annotations

import importlib.util
from pathlib import Path

import json
import pandas as pd
import pytest


def _load_summary_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "summarize_taxonomy_suite.py"
    spec = importlib.util.spec_from_file_location("summarize_taxonomy_suite", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_mean_ci_uses_small_sample_student_t_interval():
    module = _load_summary_module()
    mean, half_width, count = module.mean_ci(pd.Series([1.0, 2.0, 3.0, 4.0, 5.0]))

    assert mean == 3.0
    assert count == 5
    assert abs(half_width - 1.963) < 0.001


def test_validation_view_selection_never_reads_test_score():
    module = _load_summary_module()
    frame = pd.DataFrame(
        [
            {"label": "frequency_bucket", "val_accuracy": 0.8, "test_accuracy": 0.1},
            {"label": "frequency_bucket", "val_accuracy": 0.7, "test_accuracy": 0.9},
        ]
    )

    selected = module.select_validation_view(frame)

    assert selected["val_accuracy"] == 0.8
    assert selected["test_accuracy"] == 0.1


def test_artifact_paths_are_relativized_for_portable_result_tables():
    module = _load_summary_module()
    root = Path("/cluster/runs")
    frame = pd.DataFrame([{"artifact_path": "/cluster/runs/seed_42/probe.pt"}])

    portable = module.relativize_artifact_paths(frame, root)

    assert portable["artifact_path"].iloc[0] == "seed_42/probe.pt"


def test_requested_taxonomy_seeds_must_be_the_canonical_five():
    module = _load_summary_module()

    module.validate_requested_seeds([42, 43, 44, 45, 46])
    with pytest.raises(ValueError, match="canonical five"):
        module.validate_requested_seeds([42, 43, 44, 45])
    with pytest.raises(ValueError, match="canonical five"):
        module.validate_requested_seeds([42, 43, 44, 46, 45])


def test_true_flag_parser_rejects_false_strings():
    module = _load_summary_module()
    with pytest.raises(ValueError, match="leakage_checked=true"):
        module.require_true_flags(
            pd.DataFrame({"leakage_checked": ["True", "False"]}),
            "leakage_checked",
            Path("heldout.csv"),
        )


@pytest.mark.parametrize(
    ("column", "bad_value", "message"),
    [
        ("seed", 43, "seed 42"),
        ("method", "fno", "methods"),
        ("weight_source", "random_init", "weight sources"),
    ],
)
def test_suite_frame_rejects_mismatched_internal_metadata(column, bad_value, message):
    module = _load_summary_module()
    frame = pd.DataFrame([{"seed": 42, "method": "linear_probe", "weight_source": "pretrained"}])
    frame.loc[0, column] = bad_value

    with pytest.raises(ValueError, match=message):
        module.validate_suite_frame(
            frame,
            path=Path("seed_42/probe_results.csv"),
            expected_seed=42,
            expected_method="linear_probe",
            expected_weight_source="pretrained",
        )


def test_layer_permuted_loader_requires_and_preserves_exact_map(tmp_path: Path):
    module = _load_summary_module()
    seed_root = tmp_path / "seed_42"
    activation_root = seed_root / "layer_permuted_pretrained_activations"
    probe_root = seed_root / "layer_permuted_pretrained_probes"
    activation_root.mkdir(parents=True)
    probe_root.mkdir()
    permutation = [1, 0, 2]
    (activation_root / "activation_dump_summary.json").write_text(
        json.dumps(
            {
                "seed": 42,
                "weight_source": "layer_permuted_pretrained",
                "model_training": False,
                "split_overlap_counts": {
                    "train_val": 0,
                    "train_test": 0,
                    "val_test": 0,
                },
                "series_id_sha256_by_split": {
                    "train": "a" * 64,
                    "val": "b" * 64,
                    "test": "c" * 64,
                },
                "weight_provenance": {
                    "control": "transformer_block_order_permutation",
                    "permutation_seed": 42,
                    "runtime_layer_to_pretrained_layer": permutation,
                },
            }
        )
    )
    pd.DataFrame(
        [
            {
                "seed": 42,
                "method": "linear_probe",
                "weight_source": "layer_permuted_pretrained",
                "label": "frequency_bucket",
                "val_accuracy": 0.6,
                "test_accuracy": 0.5,
                "test_macro_f1": 0.4,
            }
        ]
    ).to_csv(probe_root / "probe_results.csv", index=False)

    selected = module.load_layer_permuted_selected(tmp_path, [42])

    assert selected["suite_seed"].tolist() == [42]
    assert json.loads(
        selected["runtime_layer_to_pretrained_layer"].iloc[0]
    ) == permutation
