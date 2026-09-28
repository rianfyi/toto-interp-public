from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from .confounds import STRUCTURAL_CONFOUND_LABELS
from .types import ActivationBatch

StructuralHoldoutMode = Literal["combination", "domain"]


@dataclass(frozen=True)
class StructuralHoldoutSpec:
    """A predeclared nuisance combination withheld from both train and validation."""

    mode: StructuralHoldoutMode
    target_label: str
    holdout_axes: tuple[str, ...]
    holdout_values: tuple[str, ...]
    candidate_count: int
    rotation: int

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _key(batch: ActivationBatch, index: int, axes: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(str(batch.labels[axis][index]) for axis in axes)


def _target_values(batch: ActivationBatch, indices: list[int], target_label: str) -> set[str]:
    return {str(batch.labels[target_label][index]) for index in indices}


def _eligible_holdout_values(
    batch: ActivationBatch,
    *,
    target_label: str,
    axes: tuple[str, ...],
) -> list[tuple[str, ...]]:
    candidates = sorted({_key(batch, index, axes) for index in range(len(batch))})
    eligible: list[tuple[str, ...]] = []
    for candidate in candidates:
        train = [
            index
            for index, split in enumerate(batch.splits)
            if split == "train" and _key(batch, index, axes) != candidate
        ]
        validation = [
            index
            for index, split in enumerate(batch.splits)
            if split == "val" and _key(batch, index, axes) != candidate
        ]
        test = [
            index
            for index, split in enumerate(batch.splits)
            if split == "test" and _key(batch, index, axes) == candidate
        ]
        train_targets = _target_values(batch, train, target_label)
        val_targets = _target_values(batch, validation, target_label)
        test_targets = _target_values(batch, test, target_label)
        # Retain only meaningful categorical evaluations: the withheld test
        # slice must contain at least two seen target values, and validation
        # must not introduce an unseen target class during view selection.
        if (
            len(train_targets) >= 2
            and len(test_targets) >= 2
            and test_targets.issubset(train_targets)
            and val_targets
            and val_targets.issubset(train_targets)
        ):
            eligible.append(candidate)
    return eligible


def choose_structural_holdout(
    batch: ActivationBatch,
    *,
    target_label: str,
    mode: StructuralHoldoutMode,
    rotation: int,
) -> StructuralHoldoutSpec:
    """Choose one deterministic, support-valid structural holdout rotation.

    For a combination holdout, the tuple consists of the *other* two
    structural labels.  This keeps the target variable capable of varying in
    the held-out evaluation set.  A domain holdout similarly evaluates one of
    the non-domain target labels on a domain absent from train and validation.
    """

    if target_label not in STRUCTURAL_CONFOUND_LABELS:
        raise ValueError(f"target_label must be one of {STRUCTURAL_CONFOUND_LABELS}, got {target_label!r}")
    if mode == "domain":
        if target_label == "domain":
            raise ValueError("Held-out-domain evaluation cannot use domain as its prediction target.")
        axes = ("domain",)
    elif mode == "combination":
        axes = tuple(label for label in STRUCTURAL_CONFOUND_LABELS if label != target_label)
    else:
        raise ValueError(f"Unsupported structural holdout mode {mode!r}.")
    candidates = _eligible_holdout_values(batch, target_label=target_label, axes=axes)
    if not candidates:
        raise ValueError(
            f"No support-valid {mode} holdout candidates for target {target_label!r}. "
            "The held-out test slice needs at least two target classes that are present in training."
        )
    return StructuralHoldoutSpec(
        mode=mode,
        target_label=target_label,
        holdout_axes=axes,
        holdout_values=candidates[rotation % len(candidates)],
        candidate_count=len(candidates),
        rotation=rotation,
    )


def build_structural_holdout_batch(batch: ActivationBatch, spec: StructuralHoldoutSpec) -> ActivationBatch:
    """Re-label retained rows as train/val/test while excluding the holdout.

    The source train and validation records matching the structural holdout are
    dropped.  Only source-test records matching it become evaluation records.
    Thus no held-out tuple/domain can enter model fitting or validation-view
    selection, and no source-train record is re-used as test data.
    """

    keep: list[int] = []
    synthetic_splits: list[str] = []
    for index, source_split in enumerate(batch.splits):
        is_heldout = _key(batch, index, spec.holdout_axes) == spec.holdout_values
        if source_split in {"train", "val"} and not is_heldout:
            keep.append(index)
            synthetic_splits.append(source_split)
        elif source_split == "test" and is_heldout:
            keep.append(index)
            synthetic_splits.append("test")
    if not keep:
        raise ValueError("Structural holdout produced no usable activation records.")
    result = batch.subset_indices(keep)
    result.splits = synthetic_splits

    train_keys = {
        _key(result, index, spec.holdout_axes) for index, split in enumerate(result.splits) if split == "train"
    }
    val_keys = {_key(result, index, spec.holdout_axes) for index, split in enumerate(result.splits) if split == "val"}
    test_keys = {_key(result, index, spec.holdout_axes) for index, split in enumerate(result.splits) if split == "test"}
    if spec.holdout_values in train_keys or spec.holdout_values in val_keys:
        raise AssertionError("Held-out structural value leaked into training or validation.")
    if test_keys != {spec.holdout_values}:
        raise AssertionError("Structural-holdout test records include a non-held-out tuple/domain.")

    result.source_metadata.update(
        {
            "structural_holdout": spec.as_dict(),
            "structural_holdout_train_leakage_checked": True,
            "structural_holdout_source_test_only": True,
        }
    )
    return result
