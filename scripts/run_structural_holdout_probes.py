from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.conditional import aggregate_series_records
from toto_interp.confounds import STRUCTURAL_CONFOUND_LABELS
from toto_interp.probe import fit_probe
from toto_interp.structural_holdout import (
    build_structural_holdout_batch,
    choose_structural_holdout,
)
from toto_interp.types import ActivationBatch, LabelSpec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate structural probes on a tuple or domain withheld from train and validation."
    )
    parser.add_argument("--activation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target-label", choices=STRUCTURAL_CONFOUND_LABELS, default="frequency_bucket")
    parser.add_argument("--holdout-mode", choices=("combination", "domain"), default="combination")
    parser.add_argument("--rotation", type=int, required=True)
    parser.add_argument(
        "--token-positions",
        nargs="+",
        choices=("final_context", "first_decode"),
        default=("final_context", "first_decode"),
    )
    return parser.parse_args()


def fit_structural_holdout(
    batch: ActivationBatch,
    *,
    output_dir: Path,
    seed: int,
    target_label: str,
    holdout_mode: str,
    rotation: int,
    token_positions: tuple[str, ...],
) -> None:
    """Fit one held-out structural cell from an already-loaded activation batch."""
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_root = output_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    holdout_spec = choose_structural_holdout(
        batch,
        target_label=target_label,
        mode=holdout_mode,
        rotation=rotation,
    )
    heldout = build_structural_holdout_batch(batch, holdout_spec)
    train_classes = tuple(
        sorted(
            {str(value) for value, split in zip(heldout.labels[target_label], heldout.splits) if split == "train"}
        )
    )
    label_spec = LabelSpec(name=target_label, task_type="categorical", classes=train_classes)
    metadata = dict(heldout.source_metadata)
    rows: list[dict[str, object]] = []
    for layer in sorted(set(heldout.layer_indices.tolist())):
        for token_position in token_positions:
            subset = heldout.subset(layer=layer, token_position=token_position, pooling_mode="series_mean")
            if not len(subset):
                continue
            try:
                artifact = fit_probe(
                    subset,
                    label_spec,
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
            artifact.artifact_metadata.update(
                {
                    "structural_holdout": holdout_spec.as_dict(),
                    "no_heldout_value_in_train_or_val": True,
                    "source_test_only_evaluation": True,
                }
            )
            artifact_path = artifact_root / f"{target_label}__layer_{layer}__{token_position}__series_mean.pt"
            artifact.save(artifact_path)
            row: dict[str, object] = {
                "target": target_label,
                "method": artifact.method,
                "model_id": artifact.model_id,
                "weight_source": artifact.weight_source,
                "seed": artifact.seed,
                "layer": layer,
                "token_position": token_position,
                "pooling_mode": "series_mean",
                "holdout_mode": holdout_spec.mode,
                "holdout_axes": json.dumps(holdout_spec.holdout_axes),
                "holdout_values": json.dumps(holdout_spec.holdout_values),
                "holdout_candidate_count": holdout_spec.candidate_count,
                "rotation": holdout_spec.rotation,
                "no_heldout_value_in_train_or_val": True,
                "source_test_only_evaluation": True,
                "artifact_path": str(artifact_path),
            }
            row.update(artifact.metrics)
            row.update({f"baseline_{key}": value for key, value in artifact.baseline_metrics.items()})
            row.update({f"shuffled_{key}": value for key, value in artifact.shuffled_metrics.items()})
            rows.append(row)
    results = pd.DataFrame(rows)
    if results.empty:
        raise RuntimeError("No structural holdout probes were fitted; inspect support and selected activation views.")
    valid = pd.to_numeric(results["val_accuracy"], errors="coerce").dropna()
    if valid.empty:
        raise RuntimeError("Structural holdout probes did not produce a validation accuracy for view selection.")
    results["is_val_selected"] = False
    results.loc[valid.idxmax(), "is_val_selected"] = True
    results = results.sort_values(["layer", "token_position"]).reset_index(drop=True)
    results.to_csv(output_dir / "structural_holdout_results.csv", index=False)
    results[results["is_val_selected"]].to_csv(output_dir / "structural_holdout_selected.csv", index=False)
    with open(output_dir / "structural_holdout_metadata.json", "w") as handle:
        json.dump(
            {
                "seed": seed,
                "structural_holdout": holdout_spec.as_dict(),
                "target_train_classes": list(train_classes),
                "no_heldout_value_in_train_or_val": True,
                "source_test_only_evaluation": True,
            },
            handle,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    batch = ActivationBatch.concatenate([ActivationBatch.load(path) for path in args.activation_files])
    batch = aggregate_series_records(batch)
    fit_structural_holdout(
        batch,
        output_dir=args.output_dir,
        seed=args.seed,
        target_label=args.target_label,
        holdout_mode=args.holdout_mode,
        rotation=args.rotation,
        token_positions=tuple(args.token_positions),
    )


if __name__ == "__main__":
    main()
