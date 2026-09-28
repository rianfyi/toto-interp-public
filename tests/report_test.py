from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from toto_interp.report import write_report


def test_write_report_renders_markdown_and_summary(tmp_path: Path):
    probe_results = pd.DataFrame(
        [
            {
                "label": "metric_type",
                "task_type": "categorical",
                "method": "linear_probe",
                "weight_source": "pretrained",
                "layer": 8,
                "token_position": "final_context",
                "pooling_mode": "series_mean",
                "test_accuracy": 0.82,
                "baseline_test_accuracy": 0.70,
            },
            {
                "label": "shift_risk",
                "task_type": "continuous",
                "method": "linear_probe",
                "weight_source": "pretrained",
                "layer": 10,
                "token_position": "first_decode",
                "pooling_mode": "series_mean",
                "test_r2": 0.56,
                "baseline_test_r2": 0.18,
            },
            {
                "label": "coordination",
                "task_type": "continuous",
                "method": "linear_probe",
                "weight_source": "pretrained",
                "layer": 11,
                "token_position": "first_decode",
                "pooling_mode": "series_mean",
                "test_r2": 0.61,
                "baseline_test_r2": 0.21,
            },
            {
                "label": "cardinality_bucket",
                "task_type": "categorical",
                "method": "linear_probe",
                "weight_source": "pretrained",
                "layer": 11,
                "token_position": "final_context",
                "pooling_mode": "series_mean",
                "test_accuracy": 0.76,
                "baseline_test_accuracy": 0.63,
            },
            {
                "label": "shift_risk",
                "task_type": "continuous",
                "method": "fno",
                "weight_source": "pretrained",
                "layer": -2,
                "token_position": "window",
                "pooling_mode": "window",
                "test_r2": 0.72,
                "baseline_test_r2": 0.18,
            },
            {
                "label": "shift_risk",
                "task_type": "continuous",
                "method": "linear_probe",
                "weight_source": "random_init",
                "layer": 9,
                "token_position": "first_decode",
                "pooling_mode": "series_mean",
                "test_r2": 0.22,
                "baseline_test_r2": 0.18,
            },
        ]
    )
    probe_results_path = tmp_path / "probe_results.csv"
    probe_results.to_csv(probe_results_path, index=False)

    intervention_dir = tmp_path / "interventions" / "shift_risk"
    intervention_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {"mode": "baseline", "strength": 0.0, "subset": "all", "probe_score": 0.0, "median_forecast_change": 0.0},
            {"mode": "steer", "strength": -0.05, "subset": "all", "probe_score": -0.2, "median_forecast_change": 0.12},
            {"mode": "steer", "strength": -0.02, "subset": "all", "probe_score": -0.1, "median_forecast_change": 0.05},
            {"mode": "steer", "strength": 0.02, "subset": "all", "probe_score": 0.1, "median_forecast_change": 0.07},
            {"mode": "steer", "strength": 0.05, "subset": "all", "probe_score": 0.2, "median_forecast_change": 0.15},
        ]
    ).to_csv(intervention_dir / "intervention_summary.csv", index=False)
    (intervention_dir / "intervention_meta.json").write_text(json.dumps({"label": "shift_risk", "mode": "steer"}))

    transfer_dir = tmp_path / "transfer"
    transfer_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "benchmark": "fev",
                "probe_label": "shift_risk",
                "dataset_count": 2,
                "transfer_r2": 0.22,
                "baseline_transfer_r2": 0.10,
            },
            {
                "benchmark": "lsf",
                "probe_label": "shift_risk",
                "dataset_count": 1,
                "transfer_r2": 0.19,
                "baseline_transfer_r2": 0.05,
            },
        ]
    ).to_csv(transfer_dir / "transfer_summary.csv", index=False)

    report_path = tmp_path / "report.md"
    summary_path = tmp_path / "report_summary.json"
    summary = write_report(
        probe_results_path=probe_results_path,
        intervention_dirs=[intervention_dir],
        transfer_dir=transfer_dir,
        output_markdown_path=report_path,
        output_summary_path=summary_path,
    )

    assert report_path.exists()
    assert summary_path.exists()
    report_text = report_path.read_text()
    assert "Toto Interp Research Report" in report_text
    assert "`shift_risk`" in report_text
    assert "Control Comparison" in report_text
    assert summary["acceptance"]["transfer_probe_count"] == 1


def test_write_report_operational_focus_uses_pretrained_slice(tmp_path: Path):
    probe_results = pd.DataFrame(
        [
            {
                "label": "coordination",
                "task_type": "continuous",
                "method": "linear_probe",
                "weight_source": "pretrained",
                "layer": 5,
                "token_position": "final_context",
                "pooling_mode": "series_mean",
                "test_r2": 0.31,
                "baseline_test_r2": 0.05,
                "shuffled_test_r2": -0.12,
            },
            {
                "label": "coordination",
                "task_type": "continuous",
                "method": "linear_probe",
                "weight_source": "random_init",
                "layer": 8,
                "token_position": "first_decode",
                "pooling_mode": "series_mean",
                "test_r2": 0.02,
                "baseline_test_r2": 0.05,
                "shuffled_test_r2": -0.20,
            },
            {
                "label": "coordination",
                "task_type": "continuous",
                "method": "fno",
                "weight_source": "pretrained",
                "layer": -2,
                "token_position": "window",
                "pooling_mode": "window",
                "test_r2": -0.15,
                "baseline_test_r2": 0.05,
                "shuffled_test_r2": -0.25,
            },
        ]
    )
    probe_results_path = tmp_path / "probe_results.csv"
    probe_results.to_csv(probe_results_path, index=False)

    report_path = tmp_path / "report.md"
    summary = write_report(
        probe_results_path=[probe_results_path],
        report_focus="operational",
        primary_method="linear_probe",
        primary_weight_source="pretrained",
        output_markdown_path=report_path,
    )

    report_text = report_path.read_text()
    assert "Main-Track Operational Results" in report_text
    assert "linear_probe/pretrained=0.310" in report_text
    assert summary["acceptance"]["operational_control_wins"] == 1
