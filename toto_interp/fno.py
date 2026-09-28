from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

from .labels import RAW_FEATURE_NAMES, build_raw_baseline_features
from .types import LabelSpec, ProbeArtifact, WindowDataset


def _split_mask(splits: list[str], split_name: str) -> np.ndarray:
    return np.asarray([split == split_name for split in splits], dtype=bool)


def _continuous_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    r2 = float("nan") if y_true.shape[0] < 2 else float(r2_score(y_true, y_pred))
    return {
        f"{prefix}_r2": r2,
        f"{prefix}_rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        f"{prefix}_mae": float(mean_absolute_error(y_true, y_pred)),
    }


def _categorical_metrics(y_true: np.ndarray, y_pred: np.ndarray, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}_macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _standardize_train(train_features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def _standardize(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    standardized = (features - mean) / std
    standardized = np.nan_to_num(standardized, nan=0.0, posinf=1e3, neginf=-1e3)
    return np.clip(standardized, -1e3, 1e3)


def _raw_feature_matrix(dataset: WindowDataset) -> np.ndarray:
    rows = [
        build_raw_baseline_features(
            context[mask],
            next_patch[mask],
        ).cpu().numpy()
        for context, next_patch, mask in zip(dataset.contexts, dataset.next_patches, dataset.variate_mask)
    ]
    return np.asarray(rows, dtype=np.float64)


class SpectralConv1d(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1.0 / max(1, in_channels * out_channels)
        self.weight_real = torch.nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))
        self.weight_imag = torch.nn.Parameter(scale * torch.randn(in_channels, out_channels, modes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, _, seq_len = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)
        out_ft = torch.zeros(
            batch,
            self.out_channels,
            x_ft.shape[-1],
            dtype=torch.cfloat,
            device=x.device,
        )
        kept_modes = min(self.modes, x_ft.shape[-1])
        weights = torch.complex(
            self.weight_real[:, :, :kept_modes],
            self.weight_imag[:, :, :kept_modes],
        )
        out_ft[:, :, :kept_modes] = torch.einsum("bim,iom->bom", x_ft[:, :, :kept_modes], weights)
        return torch.fft.irfft(out_ft, n=seq_len, dim=-1)


class FNOBlock(torch.nn.Module):
    def __init__(self, width: int, modes: int):
        super().__init__()
        self.spectral = SpectralConv1d(width, width, modes)
        self.pointwise = torch.nn.Conv1d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.spectral(x) + self.pointwise(x))


class WindowFNO(torch.nn.Module):
    def __init__(
        self,
        *,
        input_channels: int,
        width: int,
        modes: int,
        layers: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.lift = torch.nn.Conv1d(input_channels, width, kernel_size=1)
        self.blocks = torch.nn.ModuleList([FNOBlock(width, modes) for _ in range(layers)])
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.lift(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


class _CNNResidualBlock(torch.nn.Module):
    """A small dilated residual block for a raw-window CNN control."""

    def __init__(self, width: int, dilation: int) -> None:
        super().__init__()
        padding = 2 * dilation
        self.conv = torch.nn.Conv1d(width, width, kernel_size=5, padding=padding, dilation=dilation)
        self.norm = torch.nn.GroupNorm(1, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.norm(self.conv(x)))


class WindowCNN(torch.nn.Module):
    """A supervised raw-context CNN with a receptive field beyond one patch."""

    def __init__(self, *, input_channels: int, width: int, layers: int, output_dim: int) -> None:
        super().__init__()
        self.stem = torch.nn.Sequential(
            torch.nn.Conv1d(input_channels, width, kernel_size=7, padding=3),
            torch.nn.GroupNorm(1, width),
            torch.nn.GELU(),
        )
        self.blocks = torch.nn.ModuleList(
            [_CNNResidualBlock(width, dilation=2**idx) for idx in range(max(1, layers))]
        )
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(x)
        for block in self.blocks:
            hidden = block(hidden)
        return self.head(hidden)


class WindowTransformer(torch.nn.Module):
    """A patchified encoder trained from scratch on the raw context window."""

    def __init__(
        self,
        *,
        input_channels: int,
        width: int,
        layers: int,
        output_dim: int,
        patch_size: int,
        sequence_length: int,
    ) -> None:
        super().__init__()
        if width % 4 != 0:
            raise ValueError("WindowTransformer requires a width divisible by 4.")
        if sequence_length % patch_size != 0:
            raise ValueError("WindowTransformer requires sequence_length divisible by patch_size.")
        self.patch_embed = torch.nn.Conv1d(
            input_channels,
            width,
            kernel_size=patch_size,
            stride=patch_size,
        )
        patch_count = sequence_length // patch_size
        self.position = torch.nn.Parameter(torch.zeros(1, patch_count, width))
        torch.nn.init.normal_(self.position, mean=0.0, std=0.02)
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=4 * width,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(encoder_layer, num_layers=max(1, layers))
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(width),
            torch.nn.Linear(width, width),
            torch.nn.GELU(),
            torch.nn.Linear(width, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(x).transpose(1, 2)
        tokens = self.encoder(tokens + self.position[:, : tokens.shape[1]])
        return self.head(tokens.mean(dim=1))


@dataclass(frozen=True)
class FNOConfig:
    modes: int = 16
    width: int = 32
    layers: int = 3
    epochs: int = 25
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    device: str = "cpu"
    seed: int = 0


def _build_raw_window_model(
    model_type: str,
    *,
    input_channels: int,
    config: FNOConfig,
    output_dim: int,
    sequence_length: int,
    patch_size: int,
) -> torch.nn.Module:
    if model_type == "fno":
        return WindowFNO(
            input_channels=input_channels,
            width=config.width,
            modes=config.modes,
            layers=config.layers,
            output_dim=output_dim,
        )
    if model_type == "cnn":
        return WindowCNN(
            input_channels=input_channels,
            width=config.width,
            layers=config.layers,
            output_dim=output_dim,
        )
    if model_type == "transformer":
        return WindowTransformer(
            input_channels=input_channels,
            width=config.width,
            layers=config.layers,
            output_dim=output_dim,
            patch_size=patch_size,
            sequence_length=sequence_length,
        )
    raise ValueError(f"Unsupported raw-window model {model_type!r}. Expected fno, cnn, or transformer.")


def _build_fno_inputs(dataset: WindowDataset) -> torch.Tensor:
    masked_context = dataset.contexts * dataset.variate_mask.unsqueeze(-1).to(dataset.contexts.dtype)
    coverage = dataset.variate_mask.float().mean(dim=1, keepdim=True)
    coverage_channel = coverage.unsqueeze(-1).expand(-1, -1, masked_context.shape[-1])
    return torch.cat([masked_context, coverage_channel], dim=1)


def _fit_raw_baseline(
    raw_features: np.ndarray,
    labels: np.ndarray,
    splits: list[str],
    label_spec: LabelSpec,
) -> tuple[dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, tuple[str, ...] | None]:
    train_mask = _split_mask(splits, "train")
    val_mask = _split_mask(splits, "val")
    test_mask = _split_mask(splits, "test")
    raw_mean, raw_std = _standardize_train(raw_features[train_mask])
    raw_z = _standardize(raw_features, raw_mean, raw_std)
    baseline_metrics: dict[str, float] = {}
    baseline_coef: torch.Tensor
    baseline_intercept: torch.Tensor
    class_names: tuple[str, ...] | None = None

    if label_spec.task_type == "categorical":
        baseline = LogisticRegression(class_weight="balanced", max_iter=4000, random_state=0)
        train_targets = np.asarray(labels[train_mask], dtype=object)
        baseline.fit(raw_z[train_mask], train_targets)
        class_names = tuple(str(name) for name in baseline.classes_)
        for split_name, split_mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            if split_mask.any():
                baseline_metrics.update(
                    _categorical_metrics(
                        np.asarray(labels[split_mask], dtype=object),
                        baseline.predict(raw_z[split_mask]),
                        split_name,
                    )
                )
        baseline_coef = torch.as_tensor(np.atleast_2d(baseline.coef_), dtype=torch.float32)
        baseline_intercept = torch.as_tensor(np.atleast_1d(baseline.intercept_), dtype=torch.float32)
    else:
        baseline = Ridge(alpha=1.0)
        targets = labels.astype(np.float64)
        baseline.fit(raw_z[train_mask], targets[train_mask])
        for split_name, split_mask in (("train", train_mask), ("val", val_mask), ("test", test_mask)):
            if split_mask.any():
                baseline_metrics.update(
                    _continuous_metrics(targets[split_mask], baseline.predict(raw_z[split_mask]), split_name)
                )
        baseline_coef = torch.as_tensor(np.atleast_2d(baseline.coef_), dtype=torch.float32)
        baseline_intercept = torch.as_tensor(np.atleast_1d(baseline.intercept_), dtype=torch.float32)

    return (
        baseline_metrics,
        torch.as_tensor(raw_mean, dtype=torch.float32),
        torch.as_tensor(raw_std, dtype=torch.float32),
        baseline_coef,
        baseline_intercept,
        class_names,
    )


def fit_fno_probe(
    dataset: WindowDataset,
    label_spec: LabelSpec,
    *,
    config: FNOConfig,
    model_id: str = "Datadog/Toto-Open-Base-1.0",
    weight_source: str = "pretrained",
    backbone_train_mode: str = "frozen",
    checkpoint_path: str | None = None,
    model_type: str = "fno",
) -> ProbeArtifact:
    if len(dataset) == 0:
        raise ValueError("Cannot fit an FNO baseline on an empty WindowDataset.")

    rng = np.random.default_rng(config.seed)
    torch.manual_seed(config.seed)

    labels = dataset.label_array(label_spec.name)
    raw_features = _raw_feature_matrix(dataset)
    valid_mask = np.ones(len(dataset), dtype=bool)
    class_names: tuple[str, ...] | None = None
    if label_spec.task_type == "categorical":
        candidate_classes = label_spec.classes or tuple(sorted({str(value) for value in labels.tolist()}))
        valid_mask &= np.asarray([str(value) in candidate_classes for value in labels], dtype=bool)
        class_names = tuple(candidate_classes)
    else:
        valid_mask &= np.isfinite(labels.astype(np.float64))

    if not valid_mask.any():
        raise ValueError(f"No valid examples found for label {label_spec.name!r}.")

    indices = np.nonzero(valid_mask)[0].tolist()
    filtered = WindowDataset(
        contexts=dataset.contexts[indices],
        next_patches=dataset.next_patches[indices],
        variate_mask=dataset.variate_mask[indices],
        splits=[dataset.splits[i] for i in indices],
        series_ids=[dataset.series_ids[i] for i in indices],
        window_ids=[dataset.window_ids[i] for i in indices],
        labels={name: [values[i] for i in indices] for name, values in dataset.labels.items()},
        freqs=[dataset.freqs[i] for i in indices],
        item_ids=[dataset.item_ids[i] for i in indices],
        num_target_variates=[dataset.num_target_variates[i] for i in indices],
        patch_size=dataset.patch_size,
        source_metadata=dict(dataset.source_metadata),
    )
    labels = filtered.label_array(label_spec.name)
    raw_features = raw_features[valid_mask]
    train_mask = _split_mask(filtered.splits, "train")
    val_mask = _split_mask(filtered.splits, "val")
    test_mask = _split_mask(filtered.splits, "test")
    if not train_mask.any():
        raise ValueError("fit_fno_probe requires at least one training example.")
    if not val_mask.any():
        raise ValueError("fit_fno_probe requires a validation split for checkpoint selection.")

    (
        baseline_metrics,
        raw_mean,
        raw_std,
        raw_baseline_coef,
        raw_baseline_intercept,
        raw_baseline_classes,
    ) = _fit_raw_baseline(raw_features, labels, filtered.splits, label_spec)

    inputs = _build_fno_inputs(filtered)
    train_indices = torch.as_tensor(np.nonzero(train_mask)[0], dtype=torch.long)
    val_indices = torch.as_tensor(np.nonzero(val_mask)[0], dtype=torch.long)
    test_indices = torch.as_tensor(np.nonzero(test_mask)[0], dtype=torch.long)
    output_dim = len(class_names) if label_spec.task_type == "categorical" and class_names is not None else 1
    model = _build_raw_window_model(
        model_type,
        input_channels=int(inputs.shape[1]),
        config=config,
        output_dim=output_dim,
        sequence_length=int(inputs.shape[-1]),
        patch_size=int(filtered.patch_size),
    ).to(config.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    device_obj = torch.device(config.device)
    if device_obj.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device_obj)

    if label_spec.task_type == "categorical":
        class_lookup = {name: idx for idx, name in enumerate(class_names or ())}
        targets = torch.as_tensor([class_lookup[str(value)] for value in labels], dtype=torch.long)
        shuffled_targets = targets.clone()
        shuffled_train = shuffled_targets[train_indices].cpu().numpy()
        rng.shuffle(shuffled_train)
        shuffled_targets[train_indices] = torch.as_tensor(shuffled_train, dtype=torch.long)
        counts = torch.bincount(targets[train_indices], minlength=len(class_lookup)).float().clamp_min(1.0)
        class_weight = (counts.sum() / counts).to(config.device)
        criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
        shuffled_criterion = torch.nn.CrossEntropyLoss(weight=class_weight)
    else:
        targets = torch.as_tensor(labels.astype(np.float32), dtype=torch.float32)
        shuffled_targets = targets.clone()
        shuffled_train = shuffled_targets[train_indices].cpu().numpy()
        rng.shuffle(shuffled_train)
        shuffled_targets[train_indices] = torch.as_tensor(shuffled_train, dtype=torch.float32)
        criterion = torch.nn.MSELoss()
        shuffled_criterion = torch.nn.MSELoss()

    def run_epoch(model_targets: torch.Tensor, loss_fn: torch.nn.Module) -> None:
        model.train()
        permutation = torch.randperm(train_indices.numel())
        for start in range(0, train_indices.numel(), config.batch_size):
            batch_positions = permutation[start : start + config.batch_size]
            batch_indices = train_indices[batch_positions]
            batch_inputs = inputs[batch_indices].to(config.device)
            batch_targets = model_targets[batch_indices].to(config.device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_inputs)
            if label_spec.task_type == "categorical":
                loss = loss_fn(logits, batch_targets)
            else:
                loss = loss_fn(logits.squeeze(-1), batch_targets)
            loss.backward()
            optimizer.step()

    def validation_score(candidate_logits: torch.Tensor) -> float:
        positions = val_indices.numpy()
        if label_spec.task_type == "categorical":
            predicted = candidate_logits.argmax(dim=-1).numpy()
            decoded = np.asarray([(class_names or ())[idx] for idx in predicted], dtype=object)
            return float(
                f1_score(
                    np.asarray(labels[positions], dtype=object),
                    decoded[positions],
                    average="macro",
                    zero_division=0,
                )
            )
        return float(r2_score(labels[positions].astype(np.float64), candidate_logits.squeeze(-1).numpy()[positions]))

    training_started = time.perf_counter()
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_validation_score = float("-inf")
    validation_history: list[float] = []
    for epoch in range(1, config.epochs + 1):
        run_epoch(targets, criterion)
        with torch.no_grad():
            candidate_logits = model(inputs.to(config.device)).cpu()
        score = validation_score(candidate_logits)
        validation_history.append(score)
        if score > best_validation_score:
            best_validation_score = score
            best_epoch = epoch
            best_state = copy.deepcopy({key: value.detach().cpu() for key, value in model.state_dict().items()})

    if best_state is None:
        raise RuntimeError("Raw-window training did not produce a validation checkpoint.")
    model.load_state_dict(best_state)
    training_seconds = time.perf_counter() - training_started
    peak_vram_bytes = int(torch.cuda.max_memory_allocated(device_obj)) if device_obj.type == "cuda" else 0

    with torch.no_grad():
        logits = model(inputs.to(config.device)).cpu()

    shuffled_model = _build_raw_window_model(
        model_type,
        input_channels=int(inputs.shape[1]),
        config=config,
        output_dim=output_dim,
        sequence_length=int(inputs.shape[-1]),
        patch_size=int(filtered.patch_size),
    ).to(config.device)
    shuffled_optimizer = torch.optim.AdamW(
        shuffled_model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    def run_shuffled_epoch() -> None:
        shuffled_model.train()
        permutation = torch.randperm(train_indices.numel())
        for start in range(0, train_indices.numel(), config.batch_size):
            batch_positions = permutation[start : start + config.batch_size]
            batch_indices = train_indices[batch_positions]
            batch_inputs = inputs[batch_indices].to(config.device)
            batch_targets = shuffled_targets[batch_indices].to(config.device)
            shuffled_optimizer.zero_grad(set_to_none=True)
            logits = shuffled_model(batch_inputs)
            if label_spec.task_type == "categorical":
                loss = shuffled_criterion(logits, batch_targets)
            else:
                loss = shuffled_criterion(logits.squeeze(-1), batch_targets)
            loss.backward()
            shuffled_optimizer.step()

    for _ in range(max(5, config.epochs // 2)):
        run_shuffled_epoch()

    with torch.no_grad():
        shuffled_logits = shuffled_model(inputs.to(config.device)).cpu()

    metrics: dict[str, float] = {}
    shuffled_metrics: dict[str, float] = {}
    split_map = {"train": train_indices, "val": val_indices, "test": test_indices}
    if label_spec.task_type == "categorical":
        pred_indices = logits.argmax(dim=-1).numpy()
        shuffled_pred_indices = shuffled_logits.argmax(dim=-1).numpy()
        decoded = np.asarray([(class_names or ())[idx] for idx in pred_indices], dtype=object)
        shuffled_decoded = np.asarray([(class_names or ())[idx] for idx in shuffled_pred_indices], dtype=object)
        for split_name, split_indices in split_map.items():
            if split_indices.numel() == 0:
                continue
            split_positions = split_indices.numpy()
            y_true = np.asarray(labels[split_positions], dtype=object)
            metrics.update(_categorical_metrics(y_true, decoded[split_positions], split_name))
            shuffled_metrics.update(_categorical_metrics(y_true, shuffled_decoded[split_positions], split_name))
    else:
        predictions = logits.squeeze(-1).numpy()
        shuffled_predictions = shuffled_logits.squeeze(-1).numpy()
        targets_np = labels.astype(np.float64)
        for split_name, split_indices in split_map.items():
            if split_indices.numel() == 0:
                continue
            split_positions = split_indices.numpy()
            metrics.update(_continuous_metrics(targets_np[split_positions], predictions[split_positions], split_name))
            shuffled_metrics.update(
                _continuous_metrics(targets_np[split_positions], shuffled_predictions[split_positions], split_name)
            )

    return ProbeArtifact(
        label_spec=label_spec,
        layer=-2,
        token_position="window",
        pooling_mode="window",
        coef=torch.zeros((1, 1), dtype=torch.float32),
        intercept=torch.zeros(1, dtype=torch.float32),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        shuffled_metrics=shuffled_metrics,
        method=model_type,
        model_id=model_id,
        weight_source=weight_source,
        backbone_train_mode=backbone_train_mode,
        checkpoint_path=checkpoint_path,
        seed=config.seed,
        artifact_metadata={
            "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
            "config": {
                "model_type": model_type,
                "modes": config.modes,
                "width": config.width,
                "layers": config.layers,
                "epochs": config.epochs,
                "batch_size": config.batch_size,
                "learning_rate": config.learning_rate,
                "weight_decay": config.weight_decay,
                "input_channels": int(inputs.shape[1]),
                "patch_size": int(filtered.patch_size),
            },
            "raw_window_family": model_type,
            "parameter_count": parameter_count,
            "selected_epoch": best_epoch,
            "selected_validation_score": best_validation_score,
            "validation_history": validation_history,
            "training_seconds": training_seconds,
            "peak_vram_bytes": peak_vram_bytes,
        },
        class_names=raw_baseline_classes if class_names is None else tuple(class_names),
        raw_feature_mean=raw_mean,
        raw_feature_std=raw_std,
        raw_baseline_feature_names=RAW_FEATURE_NAMES,
        raw_baseline_coef=raw_baseline_coef,
        raw_baseline_intercept=raw_baseline_intercept,
    )


def fit_raw_window_probe(
    dataset: WindowDataset,
    label_spec: LabelSpec,
    *,
    config: FNOConfig,
    model_type: str,
    model_id: str = "Datadog/Toto-Open-Base-1.0",
    weight_source: str = "pretrained",
    backbone_train_mode: str = "frozen",
    checkpoint_path: str | None = None,
) -> ProbeArtifact:
    """Fit one of the supervised raw-window controls used in the taxonomy-control suite."""

    return fit_fno_probe(
        dataset,
        label_spec,
        config=config,
        model_id=model_id,
        weight_source=weight_source,
        backbone_train_mode=backbone_train_mode,
        checkpoint_path=checkpoint_path,
        model_type=model_type,
    )
