from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import PCA

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp import (
    ActivationBatch,
    GBDTConfig,
    ProbeArtifact,
    fit_gbdt_raw_window_probe,
    fit_probe,
)
from toto_interp.defaults import (
    default_dynamic_label_specs,
    default_label_specs,
    default_operational_label_specs,
    default_taxonomy_label_specs,
)
from toto_interp.fno import FNOConfig, fit_raw_window_probe
from toto_interp.loader import resolve_device
from toto_interp.types import WindowDataset

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.linear_model\._base")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit linear interpretability probes on Toto activation dumps.")
    parser.add_argument("--activation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--window-files", type=Path, nargs="*", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method",
        choices=("linear_probe", "fno", "cnn", "transformer", "gbdt"),
        default="linear_probe",
        help="Readout family. Nonlinear choices are supervised controls trained directly on raw context windows.",
    )
    parser.add_argument(
        "--label-group",
        choices=("all", "taxonomy", "dynamic", "operational"),
        default="all",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional exact label-name filter, applied after --label-group.",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Optional exact layer filter for linear probes.",
    )
    parser.add_argument(
        "--token-positions",
        nargs="+",
        choices=("all_context", "final_context", "first_decode"),
        default=None,
        help="Optional token-position filter for linear probes.",
    )
    parser.add_argument(
        "--pooling-modes",
        nargs="+",
        choices=("per_variate", "series_mean"),
        default=None,
        help="Optional pooling-mode filter for linear probes.",
    )
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument(
        "--weight-source",
        choices=("pretrained", "layer_permuted_pretrained", "random_init", "checkpoint"),
        default=None,
    )
    parser.add_argument("--checkpoint-path", type=Path, default=None)
    parser.add_argument("--randomize-scope", choices=("full", "selected_layers", "head_only"), default=None)
    parser.add_argument("--randomize-layers", type=int, nargs="*", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--max-train-series",
        type=int,
        default=0,
        help="Subsample training split to at most N unique series before fitting. 0 = no limit.",
    )
    parser.add_argument("--fno-modes", type=int, default=16)
    parser.add_argument("--fno-width", type=int, default=32)
    parser.add_argument("--fno-layers", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--gbdt-backend",
        choices=("auto", "xgboost", "hist_gradient_boosting"),
        default="hist_gradient_boosting",
        help="GBDT backend (paper: sklearn HistGradientBoosting). auto uses XGBoost when available.",
    )
    parser.add_argument("--gbdt-max-iter", type=int, default=300)
    parser.add_argument("--gbdt-learning-rate", type=float, default=0.05)
    parser.add_argument("--gbdt-max-leaf-nodes", type=int, default=31)
    parser.add_argument("--gbdt-max-depth", type=int, default=6)
    parser.add_argument("--gbdt-min-samples-leaf", type=int, default=20)
    parser.add_argument("--gbdt-l2-regularization", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of parallel probe fits (joblib). -1 = all CPUs. Lower to reduce peak memory.",
    )
    parser.add_argument(
        "--reuse-artifacts",
        action="store_true",
        help="Reuse completed per-view probe artifacts when resuming an interrupted linear-probe sweep.",
    )
    return parser.parse_args()


def choose_label_specs(group: str):
    if group == "taxonomy":
        return default_taxonomy_label_specs()
    if group == "dynamic":
        return default_dynamic_label_specs()
    if group == "operational":
        return default_operational_label_specs()
    return default_label_specs()


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    vectors = np.nan_to_num(vectors, nan=0.0, posinf=1e3, neginf=-1e3)
    vectors = np.clip(vectors, -1e3, 1e3)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True).clip(min=1e-6)
    unit = vectors / norms
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return unit @ unit.T


def _primary_metric_for_label(frame: pd.DataFrame) -> str:
    label = str(frame["label"].iloc[0]) if "label" in frame.columns and not frame.empty else ""
    task_type = str(frame["task_type"].iloc[0]) if "task_type" in frame.columns and not frame.empty else ""
    if label == "cardinality_bucket":
        return "macro_f1"
    if task_type == "categorical":
        return "accuracy"
    return "r2"


def select_best_localization_row(frame: pd.DataFrame) -> pd.Series | None:
    metric = _primary_metric_for_label(frame)
    for split_name in ("val", "test"):
        metric_column = f"{split_name}_{metric}"
        if metric_column in frame.columns:
            valid_metric = frame[metric_column].dropna()
            if not valid_metric.empty:
                return frame.loc[valid_metric.idxmax()]

    return None


def resolve_window_files(args: argparse.Namespace) -> list[Path]:
    if args.window_files:
        return list(args.window_files)
    inferred: list[Path] = []
    for activation_path in args.activation_files:
        name = activation_path.name
        if not name.endswith("_activations.pt"):
            raise ValueError(
                "Could not infer window dataset path from activation file "
                f"{activation_path}. Provide --window-files explicitly."
            )
        candidate = activation_path.with_name(name.replace("_activations.pt", "_windows.pt"))
        if not candidate.exists():
            raise FileNotFoundError(f"Expected inferred window dataset file at {candidate}")
        inferred.append(candidate)
    return inferred


def resolve_metadata(args: argparse.Namespace, source_metadata: dict[str, object]) -> dict[str, object]:
    source_seed = int(source_metadata.get("seed", 0))
    if args.seed != 0 and source_seed != 0 and args.seed != source_seed:
        raise ValueError(f"Requested probe seed {args.seed} does not match activation source seed {source_seed}.")
    return {
        "model_id": args.model_id or str(source_metadata.get("model_id", "Datadog/Toto-Open-Base-1.0")),
        "weight_source": args.weight_source or str(source_metadata.get("weight_source", "pretrained")),
        "checkpoint_path": (None if args.checkpoint_path is None else str(args.checkpoint_path))
        or source_metadata.get("checkpoint_path"),
        "randomize_scope": args.randomize_scope or source_metadata.get("randomize_scope"),
        "randomize_layers": tuple(args.randomize_layers or source_metadata.get("randomize_layers") or ()),
        "seed": args.seed if args.seed != 0 else source_seed,
        "weight_provenance": source_metadata.get("weight_provenance"),
        "activation_source_signature": json.dumps(source_metadata, sort_keys=True, default=str),
    }


def _fit_one(
    activation_batch,
    label_spec,
    layer: int,
    token_position: str,
    pooling_mode: str,
    metadata: dict,
    artifact_dir: Path,
    reuse_artifacts: bool,
) -> tuple | None:
    artifact_path = artifact_dir / f"{label_spec.name}__layer_{layer}__{token_position}__{pooling_mode}.pt"
    artifact = None
    if reuse_artifacts and artifact_path.exists():
        try:
            artifact = ProbeArtifact.load(artifact_path)
            if not _artifact_matches_request(
                artifact,
                label_spec=label_spec,
                layer=layer,
                token_position=token_position,
                pooling_mode=pooling_mode,
                metadata=metadata,
            ):
                logger.warning("Recomputing incompatible probe artifact %s", artifact_path)
                artifact = None
        except Exception as exc:
            logger.warning("Recomputing unreadable partial artifact %s: %s", artifact_path, exc)
    if artifact is None:
        subset = activation_batch.subset(layer=layer, token_position=token_position, pooling_mode=pooling_mode)
        if len(subset) == 0:
            return None
        try:
            artifact = fit_probe(
                subset,
                label_spec,
                method="linear_probe",
                model_id=str(metadata["model_id"]),
                weight_source=str(metadata["weight_source"]),
                backbone_train_mode="frozen",
                checkpoint_path=(None if metadata["checkpoint_path"] is None else str(metadata["checkpoint_path"])),
                randomize_scope=(None if metadata["randomize_scope"] is None else str(metadata["randomize_scope"])),
                randomize_layers=tuple(metadata["randomize_layers"]),
                seed=int(metadata["seed"]),
            )
        except ValueError as exc:
            logger.debug("Skipping %s layer %d %s %s: %s", label_spec.name, layer, token_position, pooling_mode, exc)
            return None
        artifact.artifact_metadata["activation_source_signature"] = metadata.get("activation_source_signature")
        artifact.artifact_metadata["weight_provenance"] = metadata.get("weight_provenance")
        artifact.save(artifact_path)

    row = {
        "label": label_spec.name,
        "task_type": label_spec.task_type,
        "method": artifact.method,
        "model_id": artifact.model_id,
        "weight_source": artifact.weight_source,
        "backbone_train_mode": artifact.backbone_train_mode,
        "checkpoint_path": artifact.checkpoint_path,
        "randomize_scope": artifact.randomize_scope,
        "randomize_layers": json.dumps(list(artifact.randomize_layers or ())),
        "seed": artifact.seed,
        "layer": layer,
        "token_position": token_position,
        "pooling_mode": pooling_mode,
        "artifact_path": str(artifact_path),
    }
    row.update(artifact.metrics)
    row.update({f"baseline_{k}": v for k, v in artifact.baseline_metrics.items()})
    row.update({f"shuffled_{k}": v for k, v in artifact.shuffled_metrics.items()})

    vec_result = None
    if artifact.mean_difference_vector is not None:
        vec_name = f"{label_spec.name}__layer_{layer}__{token_position}__{pooling_mode}"
        vec_result = (
            vec_name,
            artifact.mean_difference_vector.cpu().numpy(),
            {
                "name": vec_name,
                "label": label_spec.name,
                "layer": layer,
                "token_position": token_position,
                "pooling_mode": pooling_mode,
            },
        )

    return row, vec_result


def _artifact_matches_request(
    artifact: ProbeArtifact,
    *,
    label_spec: LabelSpec,
    layer: int,
    token_position: str,
    pooling_mode: str,
    metadata: dict,
) -> bool:
    """Return whether a reusable artifact belongs to this exact probe request."""

    return (
        artifact.label_spec.name == label_spec.name
        and int(artifact.seed) == int(metadata["seed"])
        and artifact.weight_source == str(metadata["weight_source"])
        and artifact.model_id == str(metadata["model_id"])
        and int(artifact.layer) == layer
        and artifact.token_position == token_position
        and artifact.pooling_mode == pooling_mode
        and artifact.artifact_metadata.get("activation_source_signature") == metadata.get("activation_source_signature")
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    batches = [ActivationBatch.load(path) for path in args.activation_files]
    activation_batch = ActivationBatch.concatenate(batches)

    if args.max_train_series > 0:
        rng = np.random.default_rng(args.seed)
        train_sids = sorted(
            {sid for sid, sp in zip(activation_batch.series_ids, activation_batch.splits) if sp == "train"}
        )
        n_keep = min(args.max_train_series, len(train_sids))
        kept = set(rng.choice(train_sids, size=n_keep, replace=False).tolist())
        keep_idx = [
            i
            for i, (sid, sp) in enumerate(zip(activation_batch.series_ids, activation_batch.splits))
            if sp != "train" or sid in kept
        ]
        idx_t = torch.tensor(keep_idx, dtype=torch.long)
        activation_batch = ActivationBatch(
            activations=activation_batch.activations[idx_t],
            raw_features=activation_batch.raw_features[idx_t],
            raw_feature_names=activation_batch.raw_feature_names,
            layer_indices=activation_batch.layer_indices[idx_t],
            patch_indices=activation_batch.patch_indices[idx_t],
            variate_indices=activation_batch.variate_indices[idx_t],
            token_positions=[activation_batch.token_positions[i] for i in keep_idx],
            pooling_modes=[activation_batch.pooling_modes[i] for i in keep_idx],
            series_ids=[activation_batch.series_ids[i] for i in keep_idx],
            window_ids=[activation_batch.window_ids[i] for i in keep_idx],
            splits=[activation_batch.splits[i] for i in keep_idx],
            labels={k: [v[i] for i in keep_idx] for k, v in activation_batch.labels.items()},
            source_metadata=dict(activation_batch.source_metadata),
        )
        logger.info("Subsampled training split to %d series (%d rows retained)", n_keep, len(keep_idx))

    label_specs = choose_label_specs(args.label_group)
    if args.labels:
        requested_labels = set(args.labels)
        known_labels = {spec.name for spec in label_specs}
        unknown_labels = sorted(requested_labels - known_labels)
        if unknown_labels:
            raise ValueError(
                f"Labels {unknown_labels} are not in label group {args.label_group!r}; "
                f"available labels are {sorted(known_labels)}"
            )
        label_specs = [spec for spec in label_specs if spec.name in requested_labels]
    metadata = resolve_metadata(args, activation_batch.source_metadata)
    with open(args.output_dir / "probe_input_provenance.json", "w") as handle:
        json.dump(
            {
                "model_id": metadata["model_id"],
                "weight_source": metadata["weight_source"],
                "seed": metadata["seed"],
                "checkpoint_path": metadata["checkpoint_path"],
                "randomize_scope": metadata["randomize_scope"],
                "randomize_layers": list(metadata["randomize_layers"]),
                "weight_provenance": metadata["weight_provenance"],
                "activation_source_signature": metadata["activation_source_signature"],
            },
            handle,
            indent=2,
            sort_keys=True,
            default=str,
        )

    artifact_dir = args.output_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    vector_rows: list[dict[str, object]] = []
    vectors: list[np.ndarray] = []
    vector_names: list[str] = []

    if args.method == "linear_probe":
        unique_layers = sorted(set(activation_batch.layer_indices.tolist()))
        unique_token_positions = sorted(set(activation_batch.token_positions))
        unique_pooling_modes = sorted(set(activation_batch.pooling_modes))
        if args.layers:
            requested_layers = set(args.layers)
            unique_layers = [layer for layer in unique_layers if layer in requested_layers]
        if args.token_positions:
            requested_token_positions = set(args.token_positions)
            unique_token_positions = [
                token_position
                for token_position in unique_token_positions
                if token_position in requested_token_positions
            ]
        if args.pooling_modes:
            requested_pooling_modes = set(args.pooling_modes)
            unique_pooling_modes = [
                pooling_mode for pooling_mode in unique_pooling_modes if pooling_mode in requested_pooling_modes
            ]
        if not unique_layers or not unique_token_positions or not unique_pooling_modes:
            raise ValueError(
                "The requested linear-probe view filters do not match the activation batch: "
                f"layers={unique_layers}, token_positions={unique_token_positions}, "
                f"pooling_modes={unique_pooling_modes}"
            )

        combos = [
            (label_spec, layer, token_position, pooling_mode)
            for label_spec in label_specs
            for layer in unique_layers
            for token_position in unique_token_positions
            for pooling_mode in (
                [m for m in unique_pooling_modes if m == "series_mean"]
                if label_spec.task_type == "categorical"
                else unique_pooling_modes
            )
        ]

        par_results = Parallel(n_jobs=args.n_jobs, backend="threading")(
            delayed(_fit_one)(
                activation_batch,
                label_spec,
                layer,
                token_position,
                pooling_mode,
                metadata,
                artifact_dir,
                args.reuse_artifacts,
            )
            for label_spec, layer, token_position, pooling_mode in combos
        )

        for result in par_results:
            if result is None:
                continue
            row, vec_result = result
            rows.append(row)
            if vec_result is not None:
                vec_name, vec, vec_row = vec_result
                vector_names.append(vec_name)
                vectors.append(vec)
                vector_rows.append(vec_row)
    else:
        window_datasets = [WindowDataset.load(path) for path in resolve_window_files(args)]
        window_dataset = WindowDataset.concatenate(window_datasets)
        raw_window_config = None
        if args.method != "gbdt":
            raw_window_config = FNOConfig(
                modes=args.fno_modes,
                width=args.fno_width,
                layers=args.fno_layers,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                device=resolve_device(args.device),
                seed=int(metadata["seed"]),
            )
        gbdt_config = GBDTConfig(
            backend=args.gbdt_backend,
            max_iter=args.gbdt_max_iter,
            learning_rate=args.gbdt_learning_rate,
            max_leaf_nodes=args.gbdt_max_leaf_nodes,
            max_depth=args.gbdt_max_depth,
            min_samples_leaf=args.gbdt_min_samples_leaf,
            l2_regularization=args.gbdt_l2_regularization,
            n_jobs=max(1, args.n_jobs if args.n_jobs > 0 else 1),
            seed=int(metadata["seed"]),
        )
        for label_spec in label_specs:
            try:
                if args.method == "gbdt":
                    artifact = fit_gbdt_raw_window_probe(
                        window_dataset,
                        label_spec,
                        config=gbdt_config,
                        model_id=str(metadata["model_id"]),
                        weight_source=str(metadata["weight_source"]),
                        backbone_train_mode="frozen",
                        checkpoint_path=(
                            None if metadata["checkpoint_path"] is None else str(metadata["checkpoint_path"])
                        ),
                    )
                else:
                    artifact = fit_raw_window_probe(
                        window_dataset,
                        label_spec,
                        config=raw_window_config,
                        model_type=args.method,
                        model_id=str(metadata["model_id"]),
                        weight_source=str(metadata["weight_source"]),
                        backbone_train_mode="frozen",
                        checkpoint_path=(
                            None if metadata["checkpoint_path"] is None else str(metadata["checkpoint_path"])
                        ),
                    )
            except ValueError:
                continue

            artifact.artifact_metadata["activation_source_signature"] = metadata.get("activation_source_signature")
            artifact.artifact_metadata["weight_provenance"] = metadata.get("weight_provenance")
            artifact_path = artifact_dir / f"{label_spec.name}__{args.method}__window.pt"
            artifact.save(artifact_path)
            row = {
                "label": label_spec.name,
                "task_type": label_spec.task_type,
                "method": artifact.method,
                "model_id": artifact.model_id,
                "weight_source": artifact.weight_source,
                "backbone_train_mode": artifact.backbone_train_mode,
                "checkpoint_path": artifact.checkpoint_path,
                "randomize_scope": artifact.randomize_scope,
                "randomize_layers": json.dumps(list(artifact.randomize_layers or ())),
                "seed": artifact.seed,
                "layer": artifact.layer,
                "token_position": artifact.token_position,
                "pooling_mode": artifact.pooling_mode,
                "artifact_path": str(artifact_path),
            }
            row.update(artifact.metrics)
            row.update({f"baseline_{k}": v for k, v in artifact.baseline_metrics.items()})
            row.update({f"shuffled_{k}": v for k, v in artifact.shuffled_metrics.items()})
            for key in (
                "raw_window_family",
                "parameter_count",
                "selected_epoch",
                "selected_validation_score",
                "training_seconds",
                "peak_vram_bytes",
                "feature_family",
                "feature_names",
                "feature_count",
                "gbdt_backend",
                "gbdt_params",
            ):
                if key in artifact.artifact_metadata:
                    row[key] = artifact.artifact_metadata[key]
            rows.append(row)

    results_df = pd.DataFrame(rows)
    if not results_df.empty:
        results_df = results_df.sort_values(["method", "label", "layer", "token_position", "pooling_mode"])
    results_df.to_csv(args.output_dir / "probe_results.csv", index=False)

    if vectors:
        vector_matrix = np.nan_to_num(np.stack(vectors), nan=0.0, posinf=1e3, neginf=-1e3)
        vector_matrix = np.clip(vector_matrix, -1e3, 1e3)
        cosine_df = pd.DataFrame(cosine_similarity_matrix(vector_matrix), index=vector_names, columns=vector_names)
        cosine_df.to_csv(args.output_dir / "probe_geometry_cosine.csv")

        pca_df = pd.DataFrame(vector_rows)
        meta: dict[str, object] = {"vector_count": len(vectors)}
        has_variance = vector_matrix.shape[0] >= 2 and bool(np.any(np.var(vector_matrix, axis=0) > 0))
        if has_variance:
            pca = PCA(n_components=min(3, len(vectors), vector_matrix.shape[1]))
            projected = pca.fit_transform(vector_matrix)
            for idx in range(projected.shape[1]):
                pca_df[f"pc_{idx + 1}"] = projected[:, idx]
            meta["explained_variance_ratio"] = pca.explained_variance_ratio_.tolist()
        else:
            meta["explained_variance_ratio"] = []
            meta["pca_skipped"] = "insufficient finite variance"
        pca_df.to_csv(args.output_dir / "probe_geometry_pca.csv", index=False)
        with open(args.output_dir / "probe_geometry_meta.json", "w") as handle:
            json.dump(meta, handle, indent=2)

    if not results_df.empty and args.method == "linear_probe":
        localization_rows: list[pd.Series] = []
        for _, frame in results_df.groupby(["label", "token_position", "pooling_mode"], dropna=False):
            selected = select_best_localization_row(frame)
            if selected is not None:
                localization_rows.append(selected)
        if localization_rows:
            localization_summary = pd.DataFrame(localization_rows).reset_index(drop=True)
            localization_summary.to_csv(args.output_dir / "probe_localization_summary.csv", index=False)


if __name__ == "__main__":
    main()
