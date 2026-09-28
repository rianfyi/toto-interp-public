from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

TokenPosition = Literal["all_context", "final_context", "first_decode"]
PoolingMode = Literal["per_variate", "series_mean"]
TaskType = Literal["categorical", "continuous"]
InterventionMode = Literal["ablate", "steer"]


@dataclass(frozen=True)
class TraceConfig:
    """
    Configuration for activation tracing.

    The implementation treats Toto's internal sequence as patch tokens, not raw
    timesteps. `layers` therefore refers to transformer block outputs, and
    `capture_patch_embedding=True` stores the patch-embedding activations with
    layer index -1.
    """

    layers: tuple[int, ...] = tuple(range(12))
    token_positions: tuple[TokenPosition, ...] = ("all_context", "final_context", "first_decode")
    pooling_modes: tuple[PoolingMode, ...] = ("per_variate", "series_mean")
    capture_patch_embedding: bool = True
    use_kv_cache: bool = True
    cache_format: Literal["pt"] = "pt"


@dataclass(frozen=True)
class WindowExample:
    """
    Single forecasting window used for probing and interventions.
    """

    series_id: str
    window_id: str
    split: str
    context: torch.Tensor
    next_patch: torch.Tensor
    patch_size: int
    freq: str
    item_id: str
    num_target_variates: int
    labels: dict[str, Any]


@dataclass
class ActivationBatch:
    """
    Flattened activation records aligned with window-level metadata and labels.
    """

    activations: torch.Tensor
    raw_features: torch.Tensor
    raw_feature_names: tuple[str, ...]
    layer_indices: torch.Tensor
    patch_indices: torch.Tensor
    variate_indices: torch.Tensor
    token_positions: list[str]
    pooling_modes: list[str]
    series_ids: list[str]
    window_ids: list[str]
    splits: list[str]
    labels: dict[str, list[Any]] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.activations.shape[0])

    def save(self, path: str | Path) -> None:
        payload = {
            "activations": self.activations,
            "raw_features": self.raw_features,
            "raw_feature_names": self.raw_feature_names,
            "layer_indices": self.layer_indices,
            "patch_indices": self.patch_indices,
            "variate_indices": self.variate_indices,
            "token_positions": self.token_positions,
            "pooling_modes": self.pooling_modes,
            "series_ids": self.series_ids,
            "window_ids": self.window_ids,
            "splits": self.splits,
            "labels": self.labels,
            "source_metadata": self.source_metadata,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "ActivationBatch":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**payload)

    def subset(
        self,
        *,
        layer: int | None = None,
        token_position: str | None = None,
        pooling_mode: str | None = None,
        split: str | None = None,
    ) -> "ActivationBatch":
        mask = torch.ones(len(self), dtype=torch.bool)
        if layer is not None:
            mask &= self.layer_indices == layer
        if token_position is not None:
            mask &= torch.tensor([value == token_position for value in self.token_positions], dtype=torch.bool)
        if pooling_mode is not None:
            mask &= torch.tensor([value == pooling_mode for value in self.pooling_modes], dtype=torch.bool)
        if split is not None:
            mask &= torch.tensor([value == split for value in self.splits], dtype=torch.bool)

        indices = mask.nonzero(as_tuple=False).squeeze(-1)
        return ActivationBatch(
            activations=self.activations[indices],
            raw_features=self.raw_features[indices],
            raw_feature_names=self.raw_feature_names,
            layer_indices=self.layer_indices[indices],
            patch_indices=self.patch_indices[indices],
            variate_indices=self.variate_indices[indices],
            token_positions=[self.token_positions[i] for i in indices.tolist()],
            pooling_modes=[self.pooling_modes[i] for i in indices.tolist()],
            series_ids=[self.series_ids[i] for i in indices.tolist()],
            window_ids=[self.window_ids[i] for i in indices.tolist()],
            splits=[self.splits[i] for i in indices.tolist()],
            labels={name: [values[i] for i in indices.tolist()] for name, values in self.labels.items()},
            source_metadata=dict(self.source_metadata),
        )

    def label_array(self, label_name: str) -> np.ndarray:
        if label_name not in self.labels:
            raise KeyError(
                f"Label {label_name!r} not found in ActivationBatch. "
                f"Available labels: {sorted(self.labels)}"
            )
        values = self.labels[label_name]
        if len(values) == 0:
            return np.empty((0,))
        sample = next((v for v in values if v is not None), None)
        if sample is None or isinstance(sample, str):
            return np.asarray(values, dtype=object)
        return np.asarray(values, dtype=np.float64)

    def subset_indices(self, indices: list[int] | torch.Tensor) -> "ActivationBatch":
        """Return a record-aligned subset without changing the original split labels."""

        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        index_list = index_tensor.tolist()
        return ActivationBatch(
            activations=self.activations[index_tensor],
            raw_features=self.raw_features[index_tensor],
            raw_feature_names=self.raw_feature_names,
            layer_indices=self.layer_indices[index_tensor],
            patch_indices=self.patch_indices[index_tensor],
            variate_indices=self.variate_indices[index_tensor],
            token_positions=[self.token_positions[i] for i in index_list],
            pooling_modes=[self.pooling_modes[i] for i in index_list],
            series_ids=[self.series_ids[i] for i in index_list],
            window_ids=[self.window_ids[i] for i in index_list],
            splits=[self.splits[i] for i in index_list],
            labels={name: [values[i] for i in index_list] for name, values in self.labels.items()},
            source_metadata=dict(self.source_metadata),
        )

    @staticmethod
    def concatenate(batches: list["ActivationBatch"]) -> "ActivationBatch":
        if not batches:
            raise ValueError("Cannot concatenate an empty list of ActivationBatch objects.")

        label_names = tuple(batches[0].labels.keys())
        raw_feature_names = batches[0].raw_feature_names
        for i, batch in enumerate(batches[1:], start=1):
            if tuple(batch.labels.keys()) != label_names:
                raise ValueError(
                    f"ActivationBatch.concatenate: batch 0 has labels {list(label_names)}, "
                    f"but batch {i} has labels {list(batch.labels.keys())}."
                )
            if batch.raw_feature_names != raw_feature_names:
                raise ValueError(
                    f"ActivationBatch.concatenate: batch 0 has raw_feature_names {raw_feature_names}, "
                    f"but batch {i} has {batch.raw_feature_names}."
                )
        return ActivationBatch(
            activations=torch.cat([batch.activations for batch in batches], dim=0),
            raw_features=torch.cat([batch.raw_features for batch in batches], dim=0),
            raw_feature_names=raw_feature_names,
            layer_indices=torch.cat([batch.layer_indices for batch in batches], dim=0),
            patch_indices=torch.cat([batch.patch_indices for batch in batches], dim=0),
            variate_indices=torch.cat([batch.variate_indices for batch in batches], dim=0),
            token_positions=[item for batch in batches for item in batch.token_positions],
            pooling_modes=[item for batch in batches for item in batch.pooling_modes],
            series_ids=[item for batch in batches for item in batch.series_ids],
            window_ids=[item for batch in batches for item in batch.window_ids],
            splits=[item for batch in batches for item in batch.splits],
            labels={name: [item for batch in batches for item in batch.labels[name]] for name in label_names},
            source_metadata=dict(batches[0].source_metadata),
        )


@dataclass(frozen=True)
class LabelSpec:
    name: str
    task_type: TaskType
    classes: tuple[str, ...] | None = None
    positive_quantile: float = 0.9
    negative_quantile: float = 0.1


@dataclass
class WindowDataset:
    """
    Serialized window-level dataset for non-activation methods.
    """

    contexts: torch.Tensor
    next_patches: torch.Tensor
    variate_mask: torch.Tensor
    splits: list[str]
    series_ids: list[str]
    window_ids: list[str]
    labels: dict[str, list[Any]] = field(default_factory=dict)
    freqs: list[str] = field(default_factory=list)
    item_ids: list[str] = field(default_factory=list)
    num_target_variates: list[int] = field(default_factory=list)
    patch_size: int = 0
    source_metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return int(self.contexts.shape[0])

    def save(self, path: str | Path) -> None:
        payload = {
            "contexts": self.contexts,
            "next_patches": self.next_patches,
            "variate_mask": self.variate_mask,
            "splits": self.splits,
            "series_ids": self.series_ids,
            "window_ids": self.window_ids,
            "labels": self.labels,
            "freqs": self.freqs,
            "item_ids": self.item_ids,
            "num_target_variates": self.num_target_variates,
            "patch_size": self.patch_size,
            "source_metadata": self.source_metadata,
        }
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "WindowDataset":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        return cls(**payload)

    @classmethod
    def from_windows(
        cls,
        windows: list[WindowExample],
        *,
        source_metadata: dict[str, Any] | None = None,
    ) -> "WindowDataset":
        if not windows:
            raise ValueError("Cannot build a WindowDataset from an empty window list.")
        max_variates = max(window.context.shape[0] for window in windows)
        padded_contexts: list[torch.Tensor] = []
        padded_next_patches: list[torch.Tensor] = []
        variate_masks: list[torch.Tensor] = []
        for window in windows:
            pad_rows = max_variates - window.context.shape[0]
            padded_contexts.append(torch.nn.functional.pad(window.context, (0, 0, 0, pad_rows)))
            padded_next_patches.append(torch.nn.functional.pad(window.next_patch, (0, 0, 0, pad_rows)))
            variate_masks.append(
                torch.cat(
                    [
                        torch.ones(window.context.shape[0], dtype=torch.bool),
                        torch.zeros(pad_rows, dtype=torch.bool),
                    ]
                )
            )
        return cls(
            contexts=torch.stack(padded_contexts).to(torch.float32),
            next_patches=torch.stack(padded_next_patches).to(torch.float32),
            variate_mask=torch.stack(variate_masks),
            splits=[window.split for window in windows],
            series_ids=[window.series_id for window in windows],
            window_ids=[window.window_id for window in windows],
            labels={name: [window.labels[name] for window in windows] for name in windows[0].labels.keys()},
            freqs=[window.freq for window in windows],
            item_ids=[window.item_id for window in windows],
            num_target_variates=[window.num_target_variates for window in windows],
            patch_size=int(windows[0].patch_size),
            source_metadata=dict(source_metadata or {}),
        )

    def label_array(self, label_name: str) -> np.ndarray:
        values = self.labels[label_name]
        if len(values) == 0:
            return np.empty((0,))
        sample = values[0]
        if isinstance(sample, str):
            return np.asarray(values, dtype=object)
        return np.asarray(values, dtype=np.float64)

    def subset_indices(self, indices: list[int] | torch.Tensor) -> "WindowDataset":
        """Return a window-aligned subset while preserving metadata and split assignment."""

        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        index_list = index_tensor.tolist()
        return WindowDataset(
            contexts=self.contexts[index_tensor],
            next_patches=self.next_patches[index_tensor],
            variate_mask=self.variate_mask[index_tensor],
            splits=[self.splits[i] for i in index_list],
            series_ids=[self.series_ids[i] for i in index_list],
            window_ids=[self.window_ids[i] for i in index_list],
            labels={name: [values[i] for i in index_list] for name, values in self.labels.items()},
            freqs=[self.freqs[i] for i in index_list],
            item_ids=[self.item_ids[i] for i in index_list],
            num_target_variates=[self.num_target_variates[i] for i in index_list],
            patch_size=self.patch_size,
            source_metadata=dict(self.source_metadata),
        )

    def subset(self, *, split: str | None = None) -> "WindowDataset":
        if split is None:
            return WindowDataset(
                contexts=self.contexts.clone(),
                next_patches=self.next_patches.clone(),
                variate_mask=self.variate_mask.clone(),
                splits=list(self.splits),
                series_ids=list(self.series_ids),
                window_ids=list(self.window_ids),
                labels={name: list(values) for name, values in self.labels.items()},
                freqs=list(self.freqs),
                item_ids=list(self.item_ids),
                num_target_variates=list(self.num_target_variates),
                patch_size=self.patch_size,
                source_metadata=dict(self.source_metadata),
            )

        mask = torch.tensor([value == split for value in self.splits], dtype=torch.bool)
        indices = mask.nonzero(as_tuple=False).squeeze(-1).tolist()
        return WindowDataset(
            contexts=self.contexts[indices],
            next_patches=self.next_patches[indices],
            variate_mask=self.variate_mask[indices],
            splits=[self.splits[i] for i in indices],
            series_ids=[self.series_ids[i] for i in indices],
            window_ids=[self.window_ids[i] for i in indices],
            labels={name: [values[i] for i in indices] for name, values in self.labels.items()},
            freqs=[self.freqs[i] for i in indices],
            item_ids=[self.item_ids[i] for i in indices],
            num_target_variates=[self.num_target_variates[i] for i in indices],
            patch_size=self.patch_size,
            source_metadata=dict(self.source_metadata),
        )

    @staticmethod
    def concatenate(datasets: list["WindowDataset"]) -> "WindowDataset":
        if not datasets:
            raise ValueError("Cannot concatenate an empty list of WindowDataset objects.")
        label_names = tuple(datasets[0].labels.keys())
        max_variates = max(int(dataset.contexts.shape[1]) for dataset in datasets)

        def pad_tensor_rows(tensor: torch.Tensor) -> torch.Tensor:
            pad_rows = max_variates - int(tensor.shape[1])
            if pad_rows <= 0:
                return tensor
            return torch.nn.functional.pad(tensor, (0, 0, 0, pad_rows))

        def pad_mask(mask: torch.Tensor) -> torch.Tensor:
            pad_rows = max_variates - int(mask.shape[1])
            if pad_rows <= 0:
                return mask
            return torch.nn.functional.pad(mask, (0, pad_rows))

        return WindowDataset(
            contexts=torch.cat([pad_tensor_rows(dataset.contexts) for dataset in datasets], dim=0),
            next_patches=torch.cat([pad_tensor_rows(dataset.next_patches) for dataset in datasets], dim=0),
            variate_mask=torch.cat([pad_mask(dataset.variate_mask) for dataset in datasets], dim=0),
            splits=[item for dataset in datasets for item in dataset.splits],
            series_ids=[item for dataset in datasets for item in dataset.series_ids],
            window_ids=[item for dataset in datasets for item in dataset.window_ids],
            labels={name: [item for dataset in datasets for item in dataset.labels[name]] for name in label_names},
            freqs=[item for dataset in datasets for item in dataset.freqs],
            item_ids=[item for dataset in datasets for item in dataset.item_ids],
            num_target_variates=[item for dataset in datasets for item in dataset.num_target_variates],
            patch_size=datasets[0].patch_size,
            source_metadata=dict(datasets[0].source_metadata),
        )


@dataclass
class ProbeArtifact:
    """
    Serializable representation of a fitted probe or comparison method artifact.
    """

    label_spec: LabelSpec
    layer: int
    token_position: TokenPosition
    pooling_mode: str
    coef: torch.Tensor
    intercept: torch.Tensor
    metrics: dict[str, float]
    baseline_metrics: dict[str, float]
    shuffled_metrics: dict[str, float]
    method: str = "linear_probe"
    model_id: str = "Datadog/Toto-Open-Base-1.0"
    weight_source: str = "pretrained"
    backbone_train_mode: str = "frozen"
    checkpoint_path: str | None = None
    randomize_scope: str | None = None
    randomize_layers: tuple[int, ...] | None = None
    seed: int = 0
    artifact_metadata: dict[str, Any] = field(default_factory=dict)
    class_names: tuple[str, ...] | None = None
    mean_difference_vector: torch.Tensor | None = None
    feature_mean: torch.Tensor | None = None
    feature_std: torch.Tensor | None = None
    raw_feature_mean: torch.Tensor | None = None
    raw_feature_std: torch.Tensor | None = None
    raw_baseline_feature_names: tuple[str, ...] | None = None
    raw_baseline_coef: torch.Tensor | None = None
    raw_baseline_intercept: torch.Tensor | None = None
    positive_threshold: float | None = None
    negative_threshold: float | None = None

    def save(self, path: str | Path) -> None:
        torch.save(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ProbeArtifact":
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        defaults = {
            "method": "linear_probe",
            "model_id": "Datadog/Toto-Open-Base-1.0",
            "weight_source": "pretrained",
            "backbone_train_mode": "frozen",
            "checkpoint_path": None,
            "randomize_scope": None,
            "randomize_layers": None,
            "seed": 0,
            "artifact_metadata": {},
        }
        for name, value in defaults.items():
            if not hasattr(artifact, name):
                setattr(artifact, name, value)
        return artifact


@dataclass(frozen=True)
class InterventionConfig:
    layer_indices: tuple[int, ...]
    token_position: TokenPosition
    mode: InterventionMode
    vector: torch.Tensor
    strength: float = 0.0
    normalize_by_residual: bool = True
    decode_steps: tuple[int, ...] | None = None
