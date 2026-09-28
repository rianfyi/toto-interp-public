from __future__ import annotations

import sys

import torch

from scripts import fit_toto_probes
from toto_interp import (
    ActivationBatch,
    GBDTConfig,
    LabelSpec,
    WindowDataset,
    fit_fno_probe,
    fit_gbdt_raw_window_probe,
    fit_probe,
    fit_raw_window_probe,
)
from toto_interp import gbdt as gbdt_module
from toto_interp import score_probe
from toto_interp.fno import FNOConfig
from toto_interp.gbdt import GBDT_FEATURE_NAMES
from toto_interp.labels import RAW_FEATURE_NAMES
from toto_interp.types import WindowExample


def _make_synthetic_batch() -> ActivationBatch:
    total = 30
    splits = ["train"] * 20 + ["val"] * 5 + ["test"] * 5
    activations = []
    raw_features = []
    continuous_labels = []
    categorical_labels = []

    for idx in range(total):
        value = float(idx)
        activations.append(torch.tensor([value, value * 0.5, -value], dtype=torch.float32))
        raw_features.append(torch.zeros(len(RAW_FEATURE_NAMES), dtype=torch.float32))
        continuous_labels.append(value)
        categorical_labels.append("high" if value >= 15 else "low")

    return ActivationBatch(
        activations=torch.stack(activations),
        raw_features=torch.stack(raw_features),
        raw_feature_names=RAW_FEATURE_NAMES,
        layer_indices=torch.full((total,), 10, dtype=torch.long),
        patch_indices=torch.zeros(total, dtype=torch.long),
        variate_indices=torch.full((total,), -1, dtype=torch.long),
        token_positions=["final_context"] * total,
        pooling_modes=["series_mean"] * total,
        series_ids=[f"series-{idx}" for idx in range(total)],
        window_ids=[f"window-{idx}" for idx in range(total)],
        splits=splits,
        labels={
            "shift_risk": continuous_labels,
            "band": categorical_labels,
        },
    )


def test_fit_probe_supports_continuous_and_categorical_labels():
    batch = _make_synthetic_batch()

    continuous_probe = fit_probe(batch, LabelSpec(name="shift_risk", task_type="continuous"))
    categorical_probe = fit_probe(
        batch,
        LabelSpec(name="band", task_type="categorical", classes=("low", "high")),
    )

    assert continuous_probe.metrics["test_r2"] > 0.9
    assert continuous_probe.mean_difference_vector is not None
    assert continuous_probe.positive_threshold is not None
    assert continuous_probe.negative_threshold is not None

    assert categorical_probe.metrics["test_accuracy"] > 0.9
    assert categorical_probe.class_names == ("high", "low") or categorical_probe.class_names == ("low", "high")


def test_score_probe_reuses_frozen_probe_and_raw_baseline():
    batch = _make_synthetic_batch()
    probe = fit_probe(batch, LabelSpec(name="shift_risk", task_type="continuous"))

    metrics = score_probe(batch, probe, prefix="transfer")

    assert metrics["transfer_r2"] > 0.9
    assert "baseline_transfer_r2" in metrics
    assert metrics["transfer_count"] == float(len(batch))


def test_interrupted_probe_sweep_reuses_completed_view_artifact(tmp_path, monkeypatch):
    batch = _make_synthetic_batch()
    label_spec = LabelSpec(name="band", task_type="categorical", classes=("low", "high"))
    metadata = {
        "model_id": "test-model",
        "weight_source": "pretrained",
        "checkpoint_path": None,
        "randomize_scope": None,
        "randomize_layers": (),
        "seed": 7,
        "activation_source_signature": '{"context_length": 1024}',
    }

    first = fit_toto_probes._fit_one(
        batch,
        label_spec,
        10,
        "final_context",
        "series_mean",
        metadata,
        tmp_path,
        False,
    )
    assert first is not None

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("completed artifact should be loaded instead of refit")

    monkeypatch.setattr(fit_toto_probes, "fit_probe", fail_if_refit)
    resumed = fit_toto_probes._fit_one(
        batch,
        label_spec,
        10,
        "final_context",
        "series_mean",
        metadata,
        tmp_path,
        True,
    )

    assert resumed is not None
    assert resumed[0]["test_accuracy"] == first[0]["test_accuracy"]


def test_reused_probe_artifact_requires_matching_request_metadata(tmp_path, monkeypatch):
    batch = _make_synthetic_batch()
    label_spec = LabelSpec(name="band", task_type="categorical", classes=("low", "high"))
    metadata = {
        "model_id": "test-model",
        "weight_source": "pretrained",
        "checkpoint_path": None,
        "randomize_scope": None,
        "randomize_layers": (),
        "seed": 7,
        "activation_source_signature": '{"context_length": 1024}',
    }
    result = fit_toto_probes._fit_one(
        batch,
        label_spec,
        10,
        "final_context",
        "series_mean",
        metadata,
        tmp_path,
        False,
    )
    assert result is not None
    artifact = fit_toto_probes.ProbeArtifact.load(tmp_path / "band__layer_10__final_context__series_mean.pt")
    assert fit_toto_probes._artifact_matches_request(
        artifact,
        label_spec=label_spec,
        layer=10,
        token_position="final_context",
        pooling_mode="series_mean",
        metadata=metadata,
    )

    mismatched_metadata = {**metadata, "seed": 8, "weight_source": "random_init"}
    assert not fit_toto_probes._artifact_matches_request(
        artifact,
        label_spec=label_spec,
        layer=10,
        token_position="final_context",
        pooling_mode="series_mean",
        metadata=mismatched_metadata,
    )
    assert not fit_toto_probes._artifact_matches_request(
        artifact,
        label_spec=LabelSpec(name="different_label", task_type="categorical", classes=("low", "high")),
        layer=10,
        token_position="final_context",
        pooling_mode="series_mean",
        metadata=metadata,
    )
    assert not fit_toto_probes._artifact_matches_request(
        artifact,
        label_spec=label_spec,
        layer=10,
        token_position="final_context",
        pooling_mode="series_mean",
        metadata={**metadata, "model_id": "different-model"},
    )
    assert not fit_toto_probes._artifact_matches_request(
        artifact,
        label_spec=label_spec,
        layer=10,
        token_position="final_context",
        pooling_mode="series_mean",
        metadata={**metadata, "activation_source_signature": '{"context_length": 512}'},
    )
    assert not fit_toto_probes._artifact_matches_request(
        artifact,
        label_spec=label_spec,
        layer=9,
        token_position="final_context",
        pooling_mode="series_mean",
        metadata=metadata,
    )
    assert not fit_toto_probes._artifact_matches_request(
        artifact,
        label_spec=label_spec,
        layer=10,
        token_position="first_decode",
        pooling_mode="series_mean",
        metadata=metadata,
    )
    assert not fit_toto_probes._artifact_matches_request(
        artifact,
        label_spec=label_spec,
        layer=10,
        token_position="final_context",
        pooling_mode="per_variate",
        metadata=metadata,
    )

    original_fit_probe = fit_toto_probes.fit_probe
    refit_calls = []

    def record_refit(*args, **kwargs):
        refit_calls.append((args, kwargs))
        return original_fit_probe(*args, **kwargs)

    monkeypatch.setattr(fit_toto_probes, "fit_probe", record_refit)
    resumed = fit_toto_probes._fit_one(
        batch,
        label_spec,
        10,
        "final_context",
        "series_mean",
        mismatched_metadata,
        tmp_path,
        True,
    )
    assert resumed is not None
    assert refit_calls
    assert resumed[0]["seed"] == 8
    assert resumed[0]["weight_source"] == "random_init"


def test_window_dataset_from_windows_pads_variable_variate_counts():
    base_window = WindowExample(
        series_id="series-a",
        window_id="series-a:0",
        split="train",
        context=torch.ones(2, 8),
        next_patch=torch.ones(2, 4),
        patch_size=4,
        freq="1min",
        item_id="item-a",
        num_target_variates=2,
        labels={"shift_risk": 0.1, "band": "low"},
    )
    second_window = WindowExample(
        series_id="series-b",
        window_id="series-b:0",
        split="val",
        context=torch.ones(3, 8),
        next_patch=torch.ones(3, 4),
        patch_size=4,
        freq="1min",
        item_id="item-b",
        num_target_variates=3,
        labels={"shift_risk": 0.2, "band": "high"},
    )

    dataset = WindowDataset.from_windows([base_window, second_window])

    assert dataset.contexts.shape == (2, 3, 8)
    assert dataset.next_patches.shape == (2, 3, 4)
    assert dataset.variate_mask.tolist() == [[True, True, False], [True, True, True]]


def test_fit_fno_probe_returns_method_tagged_artifact():
    windows = []
    for idx in range(24):
        split = "train" if idx < 16 else "val" if idx < 20 else "test"
        value = float(idx)
        windows.append(
            WindowExample(
                series_id=f"series-{idx}",
                window_id=f"series-{idx}:0",
                split=split,
                context=torch.full((2, 8), value, dtype=torch.float32),
                next_patch=torch.full((2, 4), value + 1.0, dtype=torch.float32),
                patch_size=4,
                freq="1min",
                item_id=f"item-{idx}",
                num_target_variates=2,
                labels={"shift_risk": value, "band": "high" if idx >= 12 else "low"},
            )
        )
    dataset = WindowDataset.from_windows(windows)

    artifact = fit_fno_probe(
        dataset,
        LabelSpec(name="shift_risk", task_type="continuous"),
        config=FNOConfig(epochs=2, batch_size=8, width=8, modes=4, layers=2, seed=0),
    )

    assert artifact.method == "fno"
    assert artifact.layer == -2
    assert "test_r2" in artifact.metrics
    assert artifact.raw_baseline_feature_names == RAW_FEATURE_NAMES


def test_other_raw_window_controls_fit_and_record_validation_selected_metadata():
    windows = []
    for idx in range(30):
        split = "train" if idx < 18 else "val" if idx < 24 else "test"
        value = float(idx)
        windows.append(
            WindowExample(
                series_id=f"series-{idx}",
                window_id=f"series-{idx}:0",
                split=split,
                context=torch.full((2, 8), value, dtype=torch.float32),
                next_patch=torch.full((2, 4), value + 1.0, dtype=torch.float32),
                patch_size=4,
                freq="1min",
                item_id=f"item-{idx}",
                num_target_variates=2,
                labels={"shift_risk": value},
            )
        )
    dataset = WindowDataset.from_windows(windows)
    config = FNOConfig(epochs=2, batch_size=8, width=8, modes=4, layers=2, seed=0)

    for method in ("cnn", "transformer"):
        artifact = fit_raw_window_probe(
            dataset,
            LabelSpec(name="shift_risk", task_type="continuous"),
            config=config,
            model_type=method,
        )
        assert artifact.method == method
        assert artifact.artifact_metadata["raw_window_family"] == method
        assert 1 <= artifact.artifact_metadata["selected_epoch"] <= config.epochs
        assert artifact.artifact_metadata["parameter_count"] > 0


def test_gbdt_raw_window_control_uses_last_patch_spectral_and_autocorrelation_features():
    windows = []
    for idx in range(45):
        split = "train" if idx < 27 else "val" if idx < 36 else "test"
        label = "high" if idx % 2 else "low"
        amplitude = 3.0 if label == "high" else 0.5
        phase = torch.arange(16, dtype=torch.float32)
        context = torch.stack(
            [amplitude * torch.sin(phase * (idx % 3 + 1)), amplitude * torch.cos(phase * (idx % 3 + 1))]
        )
        windows.append(
            WindowExample(
                series_id=f"series-{idx}",
                window_id=f"series-{idx}:0",
                split=split,
                context=context,
                next_patch=context[:, -4:],
                patch_size=4,
                freq="1min",
                item_id=f"item-{idx}",
                num_target_variates=2,
                labels={"band": label},
            )
        )
    dataset = WindowDataset.from_windows(windows)

    artifact = fit_gbdt_raw_window_probe(
        dataset,
        LabelSpec(name="band", task_type="categorical", classes=("low", "high")),
        config=GBDTConfig(
            backend="hist_gradient_boosting",
            max_iter=20,
            min_samples_leaf=2,
            max_leaf_nodes=7,
            seed=3,
        ),
    )

    assert artifact.method == "gbdt"
    assert artifact.layer == -3
    assert artifact.artifact_metadata["gbdt_backend"] == "hist_gradient_boosting"
    assert artifact.artifact_metadata["feature_names"] == list(GBDT_FEATURE_NAMES)
    assert set(artifact.artifact_metadata["feature_family"]) == {"last_patch", "spectral", "autocorrelation"}
    assert artifact.artifact_metadata["feature_count"] == len(GBDT_FEATURE_NAMES)
    assert "test_macro_f1" in artifact.metrics


def test_gbdt_auto_backend_falls_back_to_sklearn_when_xgboost_is_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "xgboost", None)

    assert gbdt_module._resolve_backend("auto") == "hist_gradient_boosting"
