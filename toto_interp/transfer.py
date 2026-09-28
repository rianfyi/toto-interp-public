from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset

from toto.evaluation.lsf.lsf_datasets import LSFDataset, LSFDatasetName

from .boom import sample_window_starts
from .fev_tasks import FEVTaskSpec, get_fev_task
from .labels import compute_dynamic_regime_labels
from .lsf import ensure_lsf_datasets, required_archives_for_lsf_datasets
from .types import WindowExample

FEV_DATASET_REPO_ID = "autogluon/fev_datasets"
FEV_RESERVED_FIELDS = frozenset({"id", "item_id", "timestamp", "start", "freq"})


def infer_freq_from_timestamp(timestamp: Sequence[object]) -> str:
    index = pd.to_datetime(timestamp)
    if len(index) >= 3:
        inferred = pd.infer_freq(index)
        if inferred:
            return str(inferred)
    if len(index) >= 2:
        delta = index[1] - index[0]
        if delta is not pd.NaT:
            return pd.tseries.frequencies.to_offset(delta).freqstr
    return "unknown"


def infer_fev_target_fields(dataset: Dataset, ev_fields: Sequence[str] = ()) -> list[str]:
    if len(dataset) == 0:
        raise ValueError("Cannot infer target fields from an empty FEV dataset.")

    row = dataset[0]
    excluded = set(FEV_RESERVED_FIELDS) | set(ev_fields)
    target_fields: list[str] = []
    for column in dataset.column_names:
        if column in excluded:
            continue
        value = np.asarray(row[column])
        if value.ndim in (1, 2) and np.issubdtype(value.dtype, np.number):
            target_fields.append(column)
    if not target_fields:
        raise ValueError("Could not infer any numeric FEV target fields.")
    return target_fields


def ensure_variate_first_array(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 1:
        return array[None, :]
    if array.ndim != 2:
        raise ValueError(f"Expected a 1D or 2D time series array, got shape {array.shape}.")
    if array.shape[0] <= array.shape[1]:
        return array
    return array.transpose(1, 0)


def _stack_named_series(row: dict[str, object], field_names: Sequence[str]) -> np.ndarray:
    arrays = [ensure_variate_first_array(np.asarray(row[field], dtype=np.float32)) for field in field_names]
    if not arrays:
        raise ValueError("Expected at least one field to stack into a target array.")
    return np.concatenate(arrays, axis=0)


def build_window_examples_from_target(
    *,
    series_id: str,
    item_id: str,
    split: str,
    target: np.ndarray,
    context_length: int,
    patch_size: int,
    freq: str,
    dataset_name: str,
    benchmark_name: str,
    max_windows_per_series: int = 16,
    include_heldout_late: bool = False,
) -> list[WindowExample]:
    if target.ndim == 1:
        target = target[None, :]
    target_tensor = torch.as_tensor(target, dtype=torch.float32)
    starts, heldout_start = sample_window_starts(
        target_tensor.shape[-1],
        context_length=context_length,
        patch_size=patch_size,
        max_windows_per_series=max_windows_per_series,
    )
    if include_heldout_late and heldout_start >= 0:
        starts = sorted(set(starts + [heldout_start]))

    windows: list[WindowExample] = []
    for start in starts:
        stop = start + context_length
        context = target_tensor[:, start:stop]
        next_patch = target_tensor[:, stop : stop + patch_size]

        labels = compute_dynamic_regime_labels(context, next_patch, num_target_variates=int(target_tensor.shape[0]))
        labels["benchmark_name"] = benchmark_name
        labels["dataset_name"] = dataset_name
        labels["is_heldout_late"] = start == heldout_start

        windows.append(
            WindowExample(
                series_id=series_id,
                window_id=f"{series_id}:{start}",
                split=split,
                context=context,
                next_patch=next_patch,
                patch_size=patch_size,
                freq=freq,
                item_id=item_id,
                num_target_variates=int(target_tensor.shape[0]),
                labels=labels,
            )
        )
    return windows


def load_fev_dataset(config_name: str, *, split: str = "train") -> Dataset:
    dataset = load_dataset(FEV_DATASET_REPO_ID, config_name, split=split)
    dataset.set_format("numpy")
    return dataset


def build_fev_windows_from_dataset(
    dataset: Dataset,
    *,
    dataset_name: str,
    context_length: int,
    patch_size: int,
    target_fields: Sequence[str] | None = None,
    ev_fields: Sequence[str] = (),
    max_series: int = 0,
    max_windows_per_series: int = 16,
    include_heldout_late: bool = False,
    split: str = "transfer",
) -> list[WindowExample]:
    if target_fields is None:
        target_fields = infer_fev_target_fields(dataset, ev_fields=ev_fields)

    windows: list[WindowExample] = []
    total_series = len(dataset) if max_series <= 0 else min(max_series, len(dataset))
    for row_index in range(total_series):
        row = dataset[row_index]
        target = _stack_named_series(row, target_fields)
        if target.shape[-1] < context_length + patch_size:
            continue

        series_id = str(row.get("id", f"{dataset_name}:{row_index}"))
        item_id = str(row.get("item_id", series_id))
        freq = str(row.get("freq") or infer_freq_from_timestamp(row["timestamp"]))
        windows.extend(
            build_window_examples_from_target(
                series_id=series_id,
                item_id=item_id,
                split=split,
                target=target,
                context_length=context_length,
                patch_size=patch_size,
                freq=freq,
                dataset_name=dataset_name,
                benchmark_name="fev",
                max_windows_per_series=max_windows_per_series,
                include_heldout_late=include_heldout_late,
            )
        )
    return windows


def build_fev_windows(
    *,
    config_name: str,
    context_length: int,
    patch_size: int,
    task_name: str | None = None,
    target_fields: Sequence[str] | None = None,
    ev_fields: Sequence[str] = (),
    max_series: int = 0,
    max_windows_per_series: int = 16,
    include_heldout_late: bool = False,
    split: str = "transfer",
) -> list[WindowExample]:
    task_spec: FEVTaskSpec | None = get_fev_task(task_name or config_name)
    if task_spec is not None:
        target_fields = tuple(target_fields or task_spec.target_fields)
        ev_fields = tuple(task_spec.exogenous_fields)
    dataset = load_fev_dataset(config_name, split="train")
    return build_fev_windows_from_dataset(
        dataset,
        dataset_name=config_name,
        context_length=context_length,
        patch_size=patch_size,
        target_fields=target_fields,
        ev_fields=ev_fields,
        max_series=max_series,
        max_windows_per_series=max_windows_per_series,
        include_heldout_late=include_heldout_late,
        split=split,
    )


def build_lsf_windows(
    *,
    dataset_name: str | LSFDatasetName,
    context_length: int,
    patch_size: int,
    lsf_path: str | Path,
    mode: str = "M",
    split: str = "test",
    max_series: int = 0,
    max_windows_per_series: int = 16,
    include_heldout_late: bool = False,
) -> list[WindowExample]:
    dataset_enum = dataset_name if isinstance(dataset_name, LSFDatasetName) else LSFDatasetName(dataset_name)
    lsf_dataset = LSFDataset(dataset_enum.value, mode=mode, split=split, lsf_path=str(lsf_path))

    windows: list[WindowExample] = []
    for row_index, row in enumerate(lsf_dataset):
        if max_series > 0 and row_index >= max_series:
            break
        target = ensure_variate_first_array(np.asarray(row["target"], dtype=np.float32))
        if target.shape[-1] < context_length + patch_size:
            continue

        windows.extend(
            build_window_examples_from_target(
                series_id=f"{dataset_enum.value}:{row_index}",
                item_id=f"{dataset_enum.value}:{row_index}",
                split=split,
                target=target,
                context_length=context_length,
                patch_size=patch_size,
                freq=lsf_dataset.freq,
                dataset_name=dataset_enum.value,
                benchmark_name="lsf",
                max_windows_per_series=max_windows_per_series,
                include_heldout_late=include_heldout_late,
            )
        )
    return windows


def collect_transfer_windows(
    *,
    fev_configs: Iterable[str] = (),
    fev_tasks: Iterable[str] = (),
    lsf_datasets: Iterable[str] = (),
    context_length: int,
    patch_size: int,
    fev_target_fields: Sequence[str] | None = None,
    fev_ev_fields: Sequence[str] = (),
    lsf_path: str | Path | None = None,
    max_series: int = 0,
    max_windows_per_series: int = 16,
    include_heldout_late: bool = False,
) -> list[WindowExample]:
    windows: list[WindowExample] = []
    for task_name in fev_tasks:
        task_spec = get_fev_task(task_name)
        if task_spec is None:
            raise ValueError(f"Unknown FEV task: {task_name}")
        windows.extend(
            build_fev_windows(
                config_name=task_spec.dataset_config,
                task_name=task_name,
                context_length=context_length,
                patch_size=patch_size,
                target_fields=task_spec.target_fields,
                ev_fields=task_spec.exogenous_fields,
                max_series=max_series,
                max_windows_per_series=max_windows_per_series,
                include_heldout_late=include_heldout_late,
            )
        )
    for config_name in fev_configs:
        windows.extend(
            build_fev_windows(
                config_name=config_name,
                context_length=context_length,
                patch_size=patch_size,
                target_fields=fev_target_fields,
                ev_fields=fev_ev_fields,
                max_series=max_series,
                max_windows_per_series=max_windows_per_series,
                include_heldout_late=include_heldout_late,
            )
        )

    if lsf_datasets:
        if lsf_path is None:
            raise ValueError("lsf_path is required when collecting LSF transfer windows.")
        for dataset_name in lsf_datasets:
            windows.extend(
                build_lsf_windows(
                    dataset_name=dataset_name,
                    context_length=context_length,
                    patch_size=patch_size,
                    lsf_path=lsf_path,
                    max_series=max_series,
                    max_windows_per_series=max_windows_per_series,
                    include_heldout_late=include_heldout_late,
                )
            )
    return windows
