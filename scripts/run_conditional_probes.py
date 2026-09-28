from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.conditional import aggregate_series_records, conditional_subset, default_conditional_probe_specs
from toto_interp.types import ActivationBatch
from toto_interp.probe import fit_probe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit predeclared common-support conditional probes for BOOM structural labels."
    )
    parser.add_argument("--activation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def _metric_column(frame: pd.DataFrame) -> str:
    if "val_macro_f1" not in frame.columns:
        raise ValueError("Conditional categorical probes require val_macro_f1 for validation-only selection.")
    return "val_macro_f1"


def fit_conditional_probes(
    batch: ActivationBatch,
    *,
    output_dir: Path,
    seed: int,
) -> None:
    """Fit all predeclared conditional cells from an aggregated activation batch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = output_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    metadata = dict(batch.source_metadata)
    rows: list[dict[str, object]] = []

    for spec in default_conditional_probe_specs():
        conditioned, counts = conditional_subset(batch, spec, seed=seed)
        spec_dir = artifact_root / spec.key
        spec_dir.mkdir(parents=True, exist_ok=True)
        for layer in sorted(set(conditioned.layer_indices.tolist())):
            # ``all_context`` has one representation per patch position.  The
            # conditional stress test uses the series as its statistical unit,
            # so keep only the terminal context and first decode views, each
            # of which yields one record per series after aggregation.
            token_positions = sorted(set(conditioned.token_positions) - {"all_context"})
            for token_position in token_positions:
                subset = conditioned.subset(layer=layer, token_position=token_position, pooling_mode="series_mean")
                if not len(subset):
                    continue
                try:
                    artifact = fit_probe(
                        subset,
                        spec.target,
                        model_id=str(metadata.get("model_id", "Datadog/Toto-Open-Base-1.0")),
                        weight_source=str(metadata.get("weight_source", "pretrained")),
                        backbone_train_mode="frozen",
                        checkpoint_path=metadata.get("checkpoint_path"),
                        randomize_scope=metadata.get("randomize_scope"),
                        randomize_layers=tuple(metadata.get("randomize_layers") or ()),
                        seed=seed,
                    )
                except ValueError:
                    continue
                artifact_path = spec_dir / f"{spec.target.name}__layer_{layer}__{token_position}__series_mean.pt"
                artifact.save(artifact_path)
                row: dict[str, object] = {
                    "conditional_key": spec.key,
                    "description": spec.description,
                    "target": spec.target.name,
                    "conditions": json.dumps(spec.conditions, sort_keys=True),
                    "balanced_series_counts": json.dumps(counts, sort_keys=True),
                    "task_type": spec.target.task_type,
                    "method": artifact.method,
                    "model_id": artifact.model_id,
                    "weight_source": artifact.weight_source,
                    "seed": artifact.seed,
                    "layer": layer,
                    "token_position": token_position,
                    "pooling_mode": "series_mean",
                    "artifact_path": str(artifact_path),
                }
                row.update(artifact.metrics)
                row.update({f"baseline_{key}": value for key, value in artifact.baseline_metrics.items()})
                row.update({f"shuffled_{key}": value for key, value in artifact.shuffled_metrics.items()})
                rows.append(row)

    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No conditional probes were fitted; inspect activation labels and stratum support.")
    results["is_val_selected"] = False
    for _, frame in results.groupby("conditional_key", sort=False):
        best_index = frame[_metric_column(frame)].idxmax()
        results.loc[best_index, "is_val_selected"] = True
    results = results.sort_values(["conditional_key", "layer", "token_position"]).reset_index(drop=True)
    results.to_csv(output_dir / "conditional_probe_results.csv", index=False)
    results[results["is_val_selected"]].to_csv(output_dir / "conditional_probe_selected.csv", index=False)
    with open(output_dir / "conditional_probe_metadata.json", "w") as handle:
        json.dump(
            {
                "seed": seed,
                "weight_source": metadata.get("weight_source", "pretrained"),
                "specifications": [
                    {
                        "key": spec.key,
                        "target": spec.target.name,
                        "conditions": spec.conditions,
                        "description": spec.description,
                    }
                    for spec in default_conditional_probe_specs()
                ],
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    batch = aggregate_series_records(
        ActivationBatch.concatenate(
            [ActivationBatch.load(path) for path in args.activation_files]
        )
    )
    fit_conditional_probes(
        batch,
        output_dir=args.output_dir,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
