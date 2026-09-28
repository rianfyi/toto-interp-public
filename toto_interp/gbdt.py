from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import torch
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from .labels import RAW_FEATURE_NAMES, build_raw_baseline_features
from .types import LabelSpec, ProbeArtifact, WindowDataset

GBDT_FEATURE_NAMES = RAW_FEATURE_NAMES + (
    "spectral_low_band_power",
    "spectral_mid_band_power",
    "spectral_high_band_power",
    "spectral_entropy",
    "spectral_dominant_frequency",
    "spectral_flatness",
    "autocorr_lag_1",
    "autocorr_lag_patch",
    "autocorr_lag_2patch",
)

GBDT_FEATURE_FAMILIES = {
    "last_patch": list(RAW_FEATURE_NAMES),
    "spectral": list(GBDT_FEATURE_NAMES[6:12]),
    "autocorrelation": list(GBDT_FEATURE_NAMES[12:]),
}


@dataclass(frozen=True)
class GBDTConfig:
    """Fixed, CPU-friendly parameters for the raw-window tree baseline."""

    backend: Literal["auto", "xgboost", "hist_gradient_boosting"] = "auto"
    max_iter: int = 300
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    max_depth: int = 6
    min_samples_leaf: int = 20
    l2_regularization: float = 1e-4
    n_jobs: int = 1
    seed: int = 0


def _split_mask(splits: list[str], split_name: str) -> np.ndarray:
    return np.asarray([split == split_name for split in splits], dtype=bool)


def _categorical_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    r2 = float("nan") if y_true.shape[0] < 2 else float(r2_score(y_true, y_pred))
    return {
        f"{prefix}_r2": r2,
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _spectral_autocorrelation_features(context: torch.Tensor, patch_size: int) -> np.ndarray:
    """Summarize the full, valid raw context without using its future patch.

    The last-patch statistics are intentionally retained as a separately named
    family.  Spectral values use a demeaned per-variate FFT and autocorrelation
    uses standardized per-variate context, so the summaries remain stable under
    level and scale changes.
    """

    values = torch.nan_to_num(context.detach().cpu().float(), nan=0.0, posinf=1e3, neginf=-1e3)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("GBDT raw-window features require at least one valid variate.")
    length = int(values.shape[-1])
    if length < 2:
        return np.zeros(len(GBDT_FEATURE_NAMES) - len(RAW_FEATURE_NAMES), dtype=np.float32)

    centered = values - values.mean(dim=-1, keepdim=True)
    scale = centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-6)
    standardized = centered / scale

    spectrum = torch.fft.rfft(centered, dim=-1, norm="forward").abs().square()
    non_dc = spectrum[:, 1:]
    bin_count = int(non_dc.shape[-1])
    if bin_count == 0:
        spectral = torch.zeros(6, dtype=torch.float32)
    else:
        normalized_power = non_dc / non_dc.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        edges = np.linspace(0, bin_count, num=4, dtype=int)
        band_values = []
        for start, end in zip(edges[:-1], edges[1:]):
            if end <= start:
                band_values.append(torch.tensor(0.0))
            else:
                band_values.append(normalized_power[:, start:end].sum(dim=-1).mean())
        entropy_denominator = float(np.log(bin_count)) if bin_count > 1 else 1.0
        entropy = (
            -(normalized_power * normalized_power.clamp_min(1e-12).log()).sum(dim=-1) / entropy_denominator
        ).mean()
        dominant_frequency = (normalized_power.argmax(dim=-1).float() + 1.0).div(float(bin_count)).mean()
        flatness = (non_dc.clamp_min(1e-12).log().mean(dim=-1).exp() / non_dc.mean(dim=-1).clamp_min(1e-12)).mean()
        spectral = torch.stack([*band_values, entropy, dominant_frequency, flatness]).to(torch.float32)

    lags = (1, min(max(1, patch_size), length - 1), min(max(1, 2 * patch_size), length - 1))
    autocorrelations = [(standardized[:, :-lag] * standardized[:, lag:]).mean(dim=-1).mean() for lag in lags]
    return torch.cat([spectral, torch.stack(autocorrelations)]).numpy().astype(np.float32, copy=False)


def build_gbdt_raw_window_features(dataset: WindowDataset) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return last-patch, spectral, and autocorrelation features for every window."""

    rows = []
    for context, next_patch, mask in zip(dataset.contexts, dataset.next_patches, dataset.variate_mask):
        valid_context = context[mask]
        valid_next_patch = next_patch[mask]
        # build_raw_baseline_features uses the second argument only to obtain
        # the patch length. Pass an empty tensor of that shape to make the
        # no-future-value guarantee explicit in this raw-context control.
        last_patch = build_raw_baseline_features(valid_context, torch.empty_like(valid_next_patch)).numpy()
        spectral_autocorr = _spectral_autocorrelation_features(valid_context, int(dataset.patch_size))
        rows.append(np.concatenate([last_patch, spectral_autocorr]))
    features = np.asarray(rows, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=1e3, neginf=-1e3)
    return np.clip(features, -1e3, 1e3), GBDT_FEATURE_NAMES


def _resolve_backend(requested: str) -> str:
    if requested not in {"auto", "xgboost", "hist_gradient_boosting"}:
        raise ValueError(f"Unsupported GBDT backend {requested!r}.")
    if requested in {"auto", "xgboost"}:
        try:
            import xgboost  # noqa: F401

            return "xgboost"
        except ImportError:
            if requested == "xgboost":
                raise ValueError("--gbdt-backend xgboost was requested, but xgboost is not installed.") from None
    return "hist_gradient_boosting"


def _model_params(config: GBDTConfig, backend: str) -> dict[str, Any]:
    params = asdict(config)
    params["resolved_backend"] = backend
    return params


def _fit_model(
    X: np.ndarray,
    y: np.ndarray,
    *,
    task_type: str,
    config: GBDTConfig,
    backend: str,
    sample_weight: np.ndarray | None,
):
    if backend == "xgboost":
        from xgboost import XGBClassifier, XGBRegressor

        common = {
            "n_estimators": config.max_iter,
            "learning_rate": config.learning_rate,
            "max_depth": config.max_depth,
            "max_leaves": config.max_leaf_nodes,
            "min_child_weight": float(config.min_samples_leaf),
            "reg_lambda": config.l2_regularization,
            "tree_method": "hist",
            "n_jobs": config.n_jobs,
            "random_state": config.seed,
            "verbosity": 0,
        }
        if task_type == "categorical":
            if len(np.unique(y)) == 2:
                model = XGBClassifier(objective="binary:logistic", eval_metric="logloss", **common)
            else:
                model = XGBClassifier(
                    objective="multi:softprob",
                    num_class=int(len(np.unique(y))),
                    eval_metric="mlogloss",
                    **common,
                )
        else:
            model = XGBRegressor(objective="reg:squarederror", **common)
    elif backend == "hist_gradient_boosting":
        common = {
            "max_iter": config.max_iter,
            "learning_rate": config.learning_rate,
            "max_leaf_nodes": config.max_leaf_nodes,
            "min_samples_leaf": config.min_samples_leaf,
            "l2_regularization": config.l2_regularization,
            "early_stopping": False,
            "random_state": config.seed,
        }
        model = (
            HistGradientBoostingClassifier(**common)
            if task_type == "categorical"
            else HistGradientBoostingRegressor(**common)
        )
    else:  # pragma: no cover - guarded by _resolve_backend
        raise AssertionError(backend)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def _filtered_dataset(dataset: WindowDataset, keep: np.ndarray) -> WindowDataset:
    return dataset.subset_indices(np.flatnonzero(keep).tolist())


def fit_gbdt_raw_window_probe(
    dataset: WindowDataset,
    label_spec: LabelSpec,
    *,
    config: GBDTConfig,
    model_id: str = "Datadog/Toto-Open-Base-1.0",
    weight_source: str = "pretrained",
    backbone_train_mode: str = "frozen",
    checkpoint_path: str | None = None,
) -> ProbeArtifact:
    """Fit a raw-window GBDT control with explicit Fourier and ACF summaries."""

    if len(dataset) == 0:
        raise ValueError("Cannot fit a GBDT baseline on an empty WindowDataset.")

    labels = dataset.label_array(label_spec.name)
    valid = np.ones(len(dataset), dtype=bool)
    if label_spec.task_type == "categorical":
        candidates = label_spec.classes or tuple(sorted({str(value) for value in labels.tolist()}))
        valid &= np.asarray([str(value) in candidates for value in labels], dtype=bool)
    else:
        valid &= np.isfinite(labels.astype(np.float64))
    filtered = _filtered_dataset(dataset, valid)
    labels = filtered.label_array(label_spec.name)
    train_mask = _split_mask(filtered.splits, "train")
    val_mask = _split_mask(filtered.splits, "val")
    test_mask = _split_mask(filtered.splits, "test")
    if not train_mask.any() or not val_mask.any():
        raise ValueError("GBDT raw-window probes require nonempty train and validation splits.")

    class_names: tuple[str, ...] | None = None
    if label_spec.task_type == "categorical":
        class_names = tuple(sorted({str(value) for value in labels[train_mask].tolist()}))
        if len(class_names) < 2:
            raise ValueError("Categorical GBDT probes require at least two training classes.")
        supported = np.asarray([str(value) in class_names for value in labels], dtype=bool)
        filtered = _filtered_dataset(filtered, supported)
        labels = filtered.label_array(label_spec.name)
        train_mask = _split_mask(filtered.splits, "train")
        val_mask = _split_mask(filtered.splits, "val")
        test_mask = _split_mask(filtered.splits, "test")

    features, feature_names = build_gbdt_raw_window_features(filtered)
    backend = _resolve_backend(config.backend)
    started = time.perf_counter()
    rng = np.random.default_rng(config.seed)

    if label_spec.task_type == "categorical":
        class_to_index = {name: index for index, name in enumerate(class_names or ())}
        targets = np.asarray([class_to_index[str(value)] for value in labels], dtype=np.int64)
        weights = compute_sample_weight(class_weight="balanced", y=targets[train_mask])
        model = _fit_model(
            features[train_mask],
            targets[train_mask],
            task_type="categorical",
            config=config,
            backend=backend,
            sample_weight=weights,
        )
        shuffled_targets = targets[train_mask].copy()
        rng.shuffle(shuffled_targets)
        shuffled_model = _fit_model(
            features[train_mask],
            shuffled_targets,
            task_type="categorical",
            config=config,
            backend=backend,
            sample_weight=weights,
        )
        predictions = np.asarray([(class_names or ())[index] for index in model.predict(features)], dtype=object)
        shuffled_predictions = np.asarray(
            [(class_names or ())[index] for index in shuffled_model.predict(features)], dtype=object
        )
    else:
        targets = labels.astype(np.float64)
        model = _fit_model(
            features[train_mask],
            targets[train_mask],
            task_type="continuous",
            config=config,
            backend=backend,
            sample_weight=None,
        )
        shuffled_targets = targets[train_mask].copy()
        rng.shuffle(shuffled_targets)
        shuffled_model = _fit_model(
            features[train_mask],
            shuffled_targets,
            task_type="continuous",
            config=config,
            backend=backend,
            sample_weight=None,
        )
        predictions = np.asarray(model.predict(features), dtype=np.float64)
        shuffled_predictions = np.asarray(shuffled_model.predict(features), dtype=np.float64)

    metrics: dict[str, float] = {}
    shuffled_metrics: dict[str, float] = {}
    for split_name, split_mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
        if not split_mask.any():
            continue
        if label_spec.task_type == "categorical":
            y_true = np.asarray(labels[split_mask], dtype=object)
            metrics.update(_categorical_metrics(y_true, predictions[split_mask], split_name))
            shuffled_metrics.update(_categorical_metrics(y_true, shuffled_predictions[split_mask], split_name))
        else:
            y_true = targets[split_mask]
            metrics.update(_continuous_metrics(y_true, predictions[split_mask], split_name))
            shuffled_metrics.update(_continuous_metrics(y_true, shuffled_predictions[split_mask], split_name))

    metadata = {
        "raw_window_family": "gbdt_spectral_autocorrelation_last_patch",
        "feature_family": GBDT_FEATURE_FAMILIES,
        "feature_names": list(feature_names),
        "feature_count": len(feature_names),
        "gbdt_backend": backend,
        "gbdt_params": _model_params(config, backend),
        "training_seconds": float(time.perf_counter() - started),
        "peak_vram_bytes": 0,
    }
    return ProbeArtifact(
        label_spec=label_spec,
        layer=-3,
        token_position="window",
        pooling_mode="window",
        coef=torch.zeros((1, len(feature_names)), dtype=torch.float32),
        intercept=torch.zeros(1, dtype=torch.float32),
        metrics=metrics,
        baseline_metrics={},
        shuffled_metrics=shuffled_metrics,
        method="gbdt",
        model_id=model_id,
        weight_source=weight_source,
        backbone_train_mode=backbone_train_mode,
        checkpoint_path=checkpoint_path,
        seed=config.seed,
        artifact_metadata=metadata,
        class_names=class_names,
    )
