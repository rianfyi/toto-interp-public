from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import torch

from scripts import run_moment_paired_interchange as runner
from toto_interp.moment_interchange import (
    MomentInterchangeConfig,
    apply_moment_interchange,
    capture_moment_residual,
    score_moment_probe,
)
from toto_interp import moment_loader
from toto_interp.moment_loader import MOMENT_BASE_ID
from toto_interp.types import LabelSpec, ProbeArtifact, WindowExample


class _TupleBlock(torch.nn.Module):
    def __init__(self, bias: float):
        super().__init__()
        self.bias = bias

    def forward(self, hidden: torch.Tensor):
        return hidden + self.bias, {"bias": self.bias}


class _TinyMoment(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Module()
        self.encoder.block = torch.nn.ModuleList([_TupleBlock(1.0), _TupleBlock(2.0)])

    def forward(self, hidden: torch.Tensor):
        metadata = None
        for block in self.encoder.block:
            hidden, metadata = block(hidden)
        return hidden, metadata


def _probe(*, signature: dict | None = None) -> ProbeArtifact:
    return ProbeArtifact(
        label_spec=LabelSpec(name="future_burstiness", task_type="continuous"),
        layer=0,
        token_position="all_context",
        pooling_mode="series_mean",
        coef=torch.tensor([[1.0, 0.0]]),
        intercept=torch.tensor([0.0]),
        metrics={},
        baseline_metrics={},
        shuffled_metrics={},
        model_id=MOMENT_BASE_ID,
        weight_source="pretrained",
        seed=42,
        artifact_metadata=(
            {}
            if signature is None
            else {"activation_source_signature": json.dumps(signature, sort_keys=True)}
        ),
        feature_mean=torch.zeros(2),
        feature_std=torch.ones(2),
    )


def test_moment_interchange_preserves_tuple_and_patches_only_selected_token():
    pipeline = _TinyMoment().eval()
    source_input = torch.full((2, 3, 4), 5.0)
    target_input = torch.zeros((2, 3, 4))

    with capture_moment_residual(pipeline, layer=0) as source_capture:
        pipeline(source_input)
    with capture_moment_residual(pipeline, layer=0) as natural_capture:
        pipeline(target_input)

    config = MomentInterchangeConfig(
        layer=0,
        token_position="final_context",
        source_residual=source_capture["residual"],
        blend=1.0,
    )
    with apply_moment_interchange(pipeline, config):
        with capture_moment_residual(pipeline, layer=0) as patched_capture:
            output, metadata = pipeline(target_input)

    natural = natural_capture["residual"]
    patched = patched_capture["residual"]
    assert torch.allclose(patched[:, :-1], natural[:, :-1])
    assert torch.allclose(
        patched[:, -1:],
        source_capture["residual"][:, -1:],
    )
    assert metadata == {"bias": 2.0}
    assert output.shape == target_input.shape


def test_moment_interchange_refuses_channel_broadcasting():
    pipeline = _TinyMoment().eval()
    config = MomentInterchangeConfig(
        layer=0,
        token_position="all_context",
        source_residual=torch.zeros(3, 3, 4),
        blend=0.5,
    )
    with pytest.raises(ValueError, match="must match exactly"):
        with apply_moment_interchange(pipeline, config):
            pipeline(torch.zeros(2, 3, 4))


def test_moment_probe_pooling_matches_activation_dump_reduction():
    residual = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
        ]
    )

    mean, std, count = score_moment_probe(
        residual,
        num_variates=2,
        probe=_probe(),
    )

    assert mean == pytest.approx(4.0)
    assert std == pytest.approx(1.0)
    assert count == 2


def _window(index: int, label: float) -> WindowExample:
    context = torch.linspace(0.0, 1.0 + index / 100.0, 8).unsqueeze(0)
    next_patch = torch.tensor([[label, label + 0.1]], dtype=torch.float32)
    return WindowExample(
        series_id=f"series-{index}",
        window_id=f"series-{index}:0",
        split="test",
        context=context,
        next_patch=next_patch,
        patch_size=2,
        freq="1h",
        item_id=f"item-{index}",
        num_target_variates=1,
        labels={
            "future_burstiness": label,
            "current_burstiness": float(index % 5),
            "current_sparsity": 0.0,
            "frequency_bucket": "hourly" if index % 2 else "daily",
            "metric_type": "gauge",
            "domain": "Infrastructure",
        },
    )


def test_matched_pair_builder_is_deterministic_unique_and_cross_series():
    windows = (
        [_window(index, 0.1) for index in range(12)]
        + [_window(index, 2.0) for index in range(12, 24)]
        + [_window(index, 10.0) for index in range(24, 36)]
    )

    triples, high_threshold, low_threshold = runner.build_matched_pair_triples(
        windows,
        num_pairs=5,
        seed=7,
        high_quantile=0.75,
        low_quantile=0.25,
        null_match_k=3,
    )
    repeated, _, _ = runner.build_matched_pair_triples(
        windows,
        num_pairs=5,
        seed=7,
        high_quantile=0.75,
        low_quantile=0.25,
        null_match_k=3,
    )

    exact = [
        (
            triple.target.window_id,
            triple.real_source.window_id,
            triple.null_source.window_id,
        )
        for triple in triples
    ]
    repeated_exact = [
        (
            triple.target.window_id,
            triple.real_source.window_id,
            triple.null_source.window_id,
        )
        for triple in repeated
    ]
    assert exact == repeated_exact
    assert len(exact) == len(set(exact)) == 5
    assert all(
        len(
            {
                triple.target.series_id,
                triple.real_source.series_id,
                triple.null_source.series_id,
            }
        )
        == 3
        for triple in triples
    )
    assert all(triple.target.context.shape == triple.real_source.context.shape for triple in triples)
    assert all(runner._future_burstiness(triple.target) <= low_threshold for triple in triples)
    assert all(runner._future_burstiness(triple.real_source) >= high_threshold for triple in triples)
    assert all(runner._future_burstiness(triple.null_source) < high_threshold for triple in triples)


def test_validation_probe_selection_never_uses_test_score():
    frame = pd.DataFrame(
        [
            {
                "label": "future_burstiness",
                "method": "linear_probe",
                "layer": 3,
                "token_position": "all_context",
                "pooling_mode": "series_mean",
                "artifact_path": "validation-winner.pt",
                "val_r2": 0.8,
                "test_r2": -10.0,
            },
            {
                "label": "future_burstiness",
                "method": "linear_probe",
                "layer": 9,
                "token_position": "all_context",
                "pooling_mode": "series_mean",
                "artifact_path": "test-winner.pt",
                "val_r2": 0.7,
                "test_r2": 100.0,
            },
        ]
    )

    selected = runner.select_validation_probe_row(frame)

    assert selected["artifact_path"] == "validation-winner.pt"


def test_probe_provenance_requires_exact_resplit_hashes():
    split_manifest = {
        split: {
            "series_count": 10,
            "series_id_sha256": f"{split}-hash",
        }
        for split in ("train", "val", "test")
    }
    signature = {
        "seed": 42,
        "model_id": MOMENT_BASE_ID,
        "weight_source": "pretrained",
        "context_length": 512,
        "patch_size": 8,
        "model_training": False,
        "series_id_sha256_by_split": {
            split: values["series_id_sha256"]
            for split, values in split_manifest.items()
        },
    }
    summary = {
        "context_length": 512,
        "patch_size": 8,
    }
    selection = runner.ProbeSelection(
        path=Path("predeclared.pt"),
        rule="predeclared_artifact",
        validation_metric=None,
        validation_score=None,
        result_row=None,
    )
    probe = _probe(signature=signature)

    _, parsed = runner.validate_probe_provenance(
        probe,
        selection=selection,
        summary=summary,
        split_seed=42,
        model_id=MOMENT_BASE_ID,
        split_manifest=split_manifest,
    )
    assert parsed["seed"] == 42

    bad_manifest = {
        **split_manifest,
        "test": {"series_count": 10, "series_id_sha256": "different"},
    }
    with pytest.raises(ValueError, match="test series split"):
        runner.validate_probe_provenance(
            probe,
            selection=selection,
            summary=summary,
            split_seed=42,
            model_id=MOMENT_BASE_ID,
            split_manifest=bad_manifest,
        )


def test_result_validator_requires_common_complete_triples():
    rows = []
    for blend in (0.25, 0.5, 1.0):
        for condition, source in (
            ("clean", "(none)"),
            ("real_patch", "real"),
            ("null_patch", "null"),
        ):
            rows.append(
                {
                    "split_seed": 42,
                    "sampling_seed": 42,
                    "pair_id": 0,
                    "blend": blend,
                    "condition": condition,
                    "target_window_id": "target:0",
                    "target_series_id": "target",
                    "source_window_id": source,
                    "source_series_id": source,
                }
            )
    frame = pd.DataFrame(rows)
    runner.validate_result_frame(frame, num_pairs=1, blends=(0.25, 0.5, 1.0))

    frame.loc[
        (frame["condition"] == "null_patch") & (frame["blend"] == 0.5),
        "target_window_id",
    ] = "different-target"
    with pytest.raises(ValueError, match="target"):
        runner.validate_result_frame(frame, num_pairs=1, blends=(0.25, 0.5, 1.0))


def test_moment_loader_can_retain_reconstruction_head_and_enforces_eval(monkeypatch):
    calls: list[dict] = []

    class FakeMoment(torch.nn.Module):
        @classmethod
        def from_pretrained(cls, model_id, *, model_kwargs):
            calls.append({"model_id": model_id, "model_kwargs": dict(model_kwargs)})
            return cls()

        def init(self):
            self.initialized = True

    monkeypatch.setattr(moment_loader, "_import_moment", lambda: FakeMoment)

    pipeline = moment_loader.load_moment_with_fallback(
        MOMENT_BASE_ID,
        device="cpu",
        task_name="reconstruction",
        seq_len=512,
        dtype=torch.float32,
    )

    assert calls[0]["model_kwargs"]["task_name"] == "reconstruction"
    assert pipeline.initialized
    assert not pipeline.training
