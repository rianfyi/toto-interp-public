from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

STRUCTURAL_LABELS = ("metric_type", "domain", "frequency_bucket", "cardinality_bucket")
DYNAMIC_LABELS = (
    "current_sparsity",
    "future_sparsity",
    "current_burstiness",
    "future_burstiness",
    "shift_risk",
    "coordination",
)
OPERATIONAL_LABELS = ("current_burstiness", "future_burstiness", "shift_risk", "coordination")
TRIVIAL_DYNAMIC_LABELS = ("current_sparsity", "future_sparsity")


@dataclass(frozen=True)
class ResearchAcceptance:
    structural_above_baseline: int
    dynamic_above_baseline: int
    operational_above_baseline: int
    operational_control_wins: int
    layer11_beats_layer10: bool
    monotonic_intervention_families: int
    transfer_probe_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "structural_above_baseline": self.structural_above_baseline,
            "dynamic_above_baseline": self.dynamic_above_baseline,
            "operational_above_baseline": self.operational_above_baseline,
            "operational_control_wins": self.operational_control_wins,
            "layer11_beats_layer10": self.layer11_beats_layer10,
            "monotonic_intervention_families": self.monotonic_intervention_families,
            "transfer_probe_count": self.transfer_probe_count,
        }


def _metric_column(frame: pd.DataFrame, prefix: str) -> str:
    task_types = set(frame.get("task_type", pd.Series(dtype=object)).dropna().astype(str))
    if "continuous" in task_types and f"{prefix}_r2" in frame.columns:
        return f"{prefix}_r2"
    if task_types and task_types.issubset({"categorical"}) and f"{prefix}_accuracy" in frame.columns:
        return f"{prefix}_accuracy"
    for candidate in (f"{prefix}_r2", f"{prefix}_accuracy"):
        if candidate in frame.columns and frame[candidate].notna().any():
            return candidate
    raise ValueError(f"Could not determine metric column for prefix {prefix!r}.")


def select_best_probe_rows(results_df: pd.DataFrame) -> pd.DataFrame:
    best_rows: list[pd.Series] = []
    for label, frame in results_df.groupby("label", dropna=False):
        metric_column = _metric_column(frame, "test")
        valid = frame[metric_column].dropna()
        if valid.empty:
            continue
        best_rows.append(frame.loc[valid.idxmax()])
    if not best_rows:
        return pd.DataFrame()
    return pd.DataFrame(best_rows).sort_values("label").reset_index(drop=True)


def summarize_method_comparison(probe_results: pd.DataFrame) -> pd.DataFrame:
    if "method" not in probe_results.columns:
        return pd.DataFrame()
    summaries: list[pd.Series] = []
    group_columns = ["label", "method"]
    if "weight_source" in probe_results.columns:
        group_columns.append("weight_source")
    for _, frame in probe_results.groupby(group_columns, dropna=False):
        metric_column = _metric_column(frame, "test")
        valid = frame[metric_column].dropna()
        if valid.empty:
            continue
        summaries.append(frame.loc[valid.idxmax()])
    if not summaries:
        return pd.DataFrame()
    sort_columns = ["label", "method"]
    if "weight_source" in probe_results.columns:
        sort_columns.append("weight_source")
    return pd.DataFrame(summaries).sort_values(sort_columns).reset_index(drop=True)


def count_probe_wins(best_rows: pd.DataFrame, labels: Iterable[str]) -> int:
    count = 0
    label_set = set(labels)
    for _, row in best_rows.iterrows():
        label = str(row["label"])
        if label not in label_set:
            continue
        metric_column = "test_accuracy" if "test_accuracy" in row.index and pd.notna(row["test_accuracy"]) else "test_r2"
        baseline_column = f"baseline_{metric_column}"
        metric = float(row.get(metric_column, np.nan))
        baseline = float(row.get(baseline_column, np.nan))
        if np.isfinite(metric) and np.isfinite(baseline) and metric - baseline >= 0.05:
            count += 1
    return count


def count_control_supported_operational_wins(method_summary: pd.DataFrame) -> int:
    if method_summary.empty:
        return 0

    count = 0
    for label in OPERATIONAL_LABELS:
        frame = method_summary[method_summary["label"] == label]
        if frame.empty:
            continue

        pretrained_linear = frame[
            (frame["method"] == "linear_probe")
            & (frame.get("weight_source", pd.Series(index=frame.index, dtype=object)) == "pretrained")
        ]
        random_linear = frame[
            (frame["method"] == "linear_probe")
            & (frame.get("weight_source", pd.Series(index=frame.index, dtype=object)) == "random_init")
        ]
        pretrained_fno = frame[
            (frame["method"] == "fno")
            & (frame.get("weight_source", pd.Series(index=frame.index, dtype=object)) == "pretrained")
        ]
        if pretrained_linear.empty or random_linear.empty or pretrained_fno.empty:
            continue

        metric_column = _metric_column(pretrained_linear, "test")
        p = float(pretrained_linear.iloc[0][metric_column])
        r = float(random_linear.iloc[0][metric_column])
        f = float(pretrained_fno.iloc[0][metric_column])
        if np.isfinite(p) and np.isfinite(r) and np.isfinite(f) and p > r and p > f:
            count += 1
    return count


def layer11_multivariate_win(best_rows: pd.DataFrame) -> bool:
    label_targets = {"coordination", "cardinality_bucket"}
    for label in label_targets:
        rows = best_rows[best_rows["label"] == label]
        if rows.empty:
            return False
        layer = int(rows.iloc[0]["layer"])
        if layer != 11:
            return False
    return True


def summarize_intervention_dir(path: Path) -> dict[str, object]:
    summary_path = path / "intervention_summary.csv"
    meta_path = path / "intervention_meta.json"
    if not summary_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Expected intervention outputs under {path}")

    frame = pd.read_csv(summary_path)
    with open(meta_path, "r") as handle:
        meta = json.load(handle)

    frame = frame[frame["mode"] != "baseline"].copy()
    if frame.empty:
        return {
            "label": meta["label"],
            "mode": meta["mode"],
            "is_monotonic": False,
            "probe_score_correlation": float("nan"),
            "forecast_change_correlation": float("nan"),
        }

    if "window_count" in frame.columns:
        grouped_rows: list[dict[str, float]] = []
        for strength, subframe in frame.groupby("strength", dropna=False):
            weights = subframe["window_count"].to_numpy(dtype=float)
            if not np.isfinite(weights).all() or float(weights.sum()) <= 0:
                weights = np.ones(len(subframe), dtype=float)
            grouped_rows.append(
                {
                    "strength": float(strength),
                    "probe_score": float(np.average(subframe["probe_score"].to_numpy(dtype=float), weights=weights)),
                    "median_forecast_change": float(
                        np.average(subframe["median_forecast_change"].to_numpy(dtype=float), weights=weights)
                    ),
                }
            )
        ordered = pd.DataFrame(grouped_rows).sort_values("strength")
    else:
        ordered = (
            frame.groupby("strength")
            .agg(probe_score=("probe_score", "mean"), median_forecast_change=("median_forecast_change", "mean"))
            .reset_index()
            .sort_values("strength")
        )

    strengths = ordered["strength"].to_numpy(dtype=float)
    probe_scores = ordered["probe_score"].to_numpy(dtype=float)
    forecast_change = ordered["median_forecast_change"].to_numpy(dtype=float)

    if strengths.shape[0] < 2:
        probe_corr = float("nan")
        forecast_corr = float("nan")
        is_monotonic = False
    else:
        probe_corr = float(np.corrcoef(strengths, probe_scores)[0, 1])
        forecast_corr = float(np.corrcoef(np.abs(strengths), forecast_change)[0, 1])
        probe_deltas = np.diff(probe_scores)
        is_monotonic = bool(np.all(probe_deltas >= -1e-6) or np.all(probe_deltas <= 1e-6))
        is_monotonic = is_monotonic and bool(np.isfinite(forecast_corr) and forecast_corr >= 0.5)

    return {
        "label": meta["label"],
        "mode": meta["mode"],
        "is_monotonic": is_monotonic,
        "probe_score_correlation": probe_corr,
        "forecast_change_correlation": forecast_corr,
    }


def summarize_transfer_dir(path: Path) -> pd.DataFrame:
    summary_path = path / "transfer_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Expected transfer summary under {path}")
    return pd.read_csv(summary_path)


def count_transfer_wins(transfer_summary: pd.DataFrame) -> int:
    if transfer_summary.empty:
        return 0
    win_labels: set[str] = set()
    for label, frame in transfer_summary.groupby("probe_label", dropna=False):
        okay = True
        for benchmark_name, subframe in frame.groupby("benchmark", dropna=False):
            metric = float(subframe["transfer_r2"].mean()) if "transfer_r2" in subframe.columns else float("nan")
            baseline = (
                float(subframe["baseline_transfer_r2"].mean())
                if "baseline_transfer_r2" in subframe.columns
                else float("nan")
            )
            if not (np.isfinite(metric) and np.isfinite(baseline) and metric - baseline >= 0.02):
                okay = False
                break
        if okay:
            win_labels.add(str(label))
    return len(win_labels)


def build_acceptance_summary(
    *,
    best_probe_rows: pd.DataFrame,
    method_summary: pd.DataFrame,
    intervention_summaries: list[dict[str, object]],
    transfer_summary: pd.DataFrame | None,
) -> ResearchAcceptance:
    return ResearchAcceptance(
        structural_above_baseline=count_probe_wins(best_probe_rows, STRUCTURAL_LABELS),
        dynamic_above_baseline=count_probe_wins(best_probe_rows, DYNAMIC_LABELS),
        operational_above_baseline=count_probe_wins(best_probe_rows, OPERATIONAL_LABELS),
        operational_control_wins=count_control_supported_operational_wins(method_summary),
        layer11_beats_layer10=layer11_multivariate_win(best_probe_rows),
        monotonic_intervention_families=sum(bool(item["is_monotonic"]) for item in intervention_summaries),
        transfer_probe_count=0 if transfer_summary is None else count_transfer_wins(transfer_summary),
    )


def render_markdown_report(
    *,
    probe_results: pd.DataFrame,
    best_probe_rows: pd.DataFrame,
    method_summary: pd.DataFrame,
    intervention_summaries: list[dict[str, object]],
    transfer_summary: pd.DataFrame | None,
    acceptance: ResearchAcceptance,
    report_focus: str,
) -> str:
    lines: list[str] = []
    lines.append("# Toto Interp Research Report")
    lines.append("")
    lines.append("## Executive Summary")
    lines.append("")
    if report_focus == "operational":
        lines.append(
            "This run evaluates whether operational regime labels are linearly recoverable from a frozen pretrained"
            " Toto checkpoint, relative to raw-feature, random-initialization, and raw-window FNO controls."
        )
    else:
        lines.append(
            "This run evaluates whether observability-native concepts are linearly decodable from Toto's residual"
            " stream, whether those readouts localize to specific layers and token positions, and whether steering or"
            " ablating selected regime directions changes forecasts."
        )
    lines.append("")
    lines.append("## Acceptance Snapshot")
    lines.append("")
    lines.append(f"- Operational concepts above raw-feature baseline: {acceptance.operational_above_baseline}/4")
    lines.append(
        f"- Operational concepts where pretrained linear beats random-init and FNO: {acceptance.operational_control_wins}/4"
    )
    if report_focus != "operational":
        lines.append(f"- Structural concepts above raw-feature baseline: {acceptance.structural_above_baseline}/4")
        lines.append(f"- Dynamic concepts above raw-feature baseline: {acceptance.dynamic_above_baseline}/6")
        lines.append(f"- Layer 11 wins on coordination and cardinality: {acceptance.layer11_beats_layer10}")
        lines.append(f"- Cross-benchmark transfer wins (FEV and LSF): {acceptance.transfer_probe_count}")
    lines.append(f"- Monotonic intervention families: {acceptance.monotonic_intervention_families}")
    lines.append("")

    if report_focus == "operational":
        lines.append("## Operational Results")
    else:
        lines.append("## Best BOOM Probe Views")
    lines.append("")
    if best_probe_rows.empty:
        lines.append("No probe fits were available.")
    else:
        for _, row in best_probe_rows.iterrows():
            if report_focus == "operational" and str(row["label"]) not in OPERATIONAL_LABELS:
                continue
            if pd.notna(row.get("test_accuracy")):
                metric_text = (
                    f"test_accuracy={row.get('test_accuracy', float('nan')):.3f},"
                    f" baseline_test_accuracy={row.get('baseline_test_accuracy', float('nan')):.3f}"
                )
            else:
                metric_text = (
                    f"test_r2={row.get('test_r2', float('nan')):.3f},"
                    f" baseline_test_r2={row.get('baseline_test_r2', float('nan')):.3f},"
                    f" shuffled_test_r2={row.get('shuffled_test_r2', float('nan')):.3f}"
                )
            lines.append(
                f"- `{row['label']}`: layer {int(row['layer'])}, `{row['token_position']}`,"
                f" `{row['pooling_mode']}`; {metric_text}"
            )
    lines.append("")

    if report_focus != "operational":
        lines.append("## Localization And Geometry")
        lines.append("")
        if best_probe_rows.empty:
            lines.append("Localization summary unavailable.")
        else:
            layer_counts = best_probe_rows.groupby("layer").size().sort_index()
            layer_summary = ", ".join(
                f"layer {int(layer)}: {int(count)} best probes" for layer, count in layer_counts.items()
            )
            lines.append(f"- Best probes concentrate as follows: {layer_summary}.")
            late_dynamic = best_probe_rows[
                best_probe_rows["label"].isin(["future_sparsity", "future_burstiness", "shift_risk", "coordination"])
            ]
            if not late_dynamic.empty:
                mean_layer = float(late_dynamic["layer"].mean())
                lines.append(f"- Future-facing dynamic concepts average at layer {mean_layer:.2f}.")
        lines.append("")

    lines.append("## Direction Interventions")
    lines.append("")
    if not intervention_summaries:
        lines.append("No intervention outputs were found.")
    else:
        for summary in sorted(intervention_summaries, key=lambda item: str(item["label"])):
            lines.append(
                f"- `{summary['label']}` ({summary['mode']}): monotonic={summary['is_monotonic']},"
                f" corr(strength, probe_score)={summary['probe_score_correlation']:.3f},"
                f" corr(|strength|, forecast_change)={summary['forecast_change_correlation']:.3f}"
            )
    lines.append("")

    if report_focus != "operational":
        lines.append("## Zero-Shot Transfer")
        lines.append("")
        if transfer_summary is None or transfer_summary.empty:
            lines.append("No transfer summary was available.")
        else:
            for _, row in transfer_summary.sort_values(["benchmark", "probe_label"]).iterrows():
                lines.append(
                    f"- `{row['benchmark']}` / `{row['probe_label']}`: transfer_r2={row.get('transfer_r2', float('nan')):.3f},"
                    f" baseline_transfer_r2={row.get('baseline_transfer_r2', float('nan')):.3f},"
                    f" datasets={int(row.get('dataset_count', 0))}"
                )
        lines.append("")

    has_multiple_controls = (
        not method_summary.empty
        and (
            method_summary["method"].nunique() > 1
            or ("weight_source" in method_summary.columns and method_summary["weight_source"].nunique() > 1)
        )
    )
    if has_multiple_controls:
        lines.append("## Control Comparison")
        lines.append("")
        for label, frame in method_summary.groupby("label", dropna=False):
            if report_focus == "operational" and label not in OPERATIONAL_LABELS:
                continue
            fragments: list[str] = []
            sort_columns = ["method"] + (["weight_source"] if "weight_source" in frame.columns else [])
            for _, row in frame.sort_values(sort_columns).iterrows():
                metric_column = "test_accuracy" if pd.notna(row.get("test_accuracy", np.nan)) else "test_r2"
                weight_source = row.get("weight_source", "unknown")
                fragments.append(f"{row['method']}/{weight_source}={row.get(metric_column, float('nan')):.3f}")
            lines.append(f"- `{label}`: " + ", ".join(fragments))
        lines.append("")

    lines.append("## What This Run Demonstrates")
    lines.append("")
    if report_focus == "operational":
        lines.append(
            "This run compares linear recovery of operational regime variables (burstiness, shift risk, and"
            " coordination) from a pretrained Toto checkpoint with randomized weights and a nonlinear raw-window FNO"
            " baseline; the Acceptance Snapshot counts the labels on which the pretrained probe exceeds both."
        )
    else:
        lines.append(
            "This codebase can automatically build BOOM research windows, trace residual-stream activations over Toto's"
            " patch-token sequence, fit linear probes for both taxonomy and dynamic regime concepts, identify the best"
            " localization views, intervene on learned concept directions during forecasting, and test whether dynamic"
            " regime probes transfer to public FEV and LSF datasets."
        )
    lines.append("")
    return "\n".join(lines)


def write_report(
    *,
    probe_results_path: Path | Iterable[Path],
    intervention_dirs: Iterable[Path] = (),
    transfer_dir: Path | None = None,
    report_focus: str = "full",
    primary_method: str = "linear_probe",
    primary_weight_source: str = "pretrained",
    output_markdown_path: Path,
    output_summary_path: Path | None = None,
) -> dict[str, object]:
    probe_result_paths = [probe_results_path] if isinstance(probe_results_path, Path) else list(probe_results_path)
    probe_results = pd.concat([pd.read_csv(path) for path in probe_result_paths], ignore_index=True)

    primary_results = probe_results.copy()
    if "method" in primary_results.columns:
        primary_results = primary_results[primary_results["method"] == primary_method].copy()
    if "weight_source" in primary_results.columns:
        primary_results = primary_results[primary_results["weight_source"] == primary_weight_source].copy()
    best_probe_rows = select_best_probe_rows(primary_results)
    method_summary = summarize_method_comparison(probe_results)

    intervention_summaries = [summarize_intervention_dir(path) for path in intervention_dirs]
    transfer_summary = summarize_transfer_dir(transfer_dir) if transfer_dir is not None and transfer_dir.exists() else None

    acceptance = build_acceptance_summary(
        best_probe_rows=best_probe_rows,
        method_summary=method_summary,
        intervention_summaries=intervention_summaries,
        transfer_summary=transfer_summary,
    )

    markdown = render_markdown_report(
        probe_results=probe_results,
        best_probe_rows=best_probe_rows,
        method_summary=method_summary,
        intervention_summaries=intervention_summaries,
        transfer_summary=transfer_summary,
        acceptance=acceptance,
        report_focus=report_focus,
    )
    output_markdown_path.parent.mkdir(parents=True, exist_ok=True)
    output_markdown_path.write_text(markdown)

    summary = {
        "acceptance": acceptance.as_dict(),
        "best_probe_rows": best_probe_rows.to_dict(orient="records"),
        "method_summary": method_summary.to_dict(orient="records"),
        "interventions": intervention_summaries,
        "transfer_summary": [] if transfer_summary is None else transfer_summary.to_dict(orient="records"),
    }
    if output_summary_path is not None:
        output_summary_path.parent.mkdir(parents=True, exist_ok=True)
        output_summary_path.write_text(json.dumps(summary, indent=2))
    return summary
