# Contains a helper adapted from Datadog Toto, licensed under Apache-2.0.
# Copyright [2005-Present] Datadog, Inc.
# Modified for toto-interp: standalone BOOM downloading and audit integration.
# See THIRD_PARTY_NOTICES.md and licenses/ for attribution and license terms.

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from datasets import load_from_disk
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download

from toto.data.util.dataset import MaskedTimeseries

from .labels import build_raw_baseline_features, build_taxonomy_labels, compute_dynamic_regime_labels
from .types import WindowExample

BOOM_REPO_ID = "Datadog/BOOM"


def load_boom_taxonomy(
    repo_id: str = BOOM_REPO_ID,
    *,
    local_path: Path | None = None,
) -> dict[str, dict[str, Any]]:
    if local_path is not None:
        taxonomy_file = local_path / "dataset_taxonomy.json"
    else:
        taxonomy_file = Path(hf_hub_download(repo_id, "dataset_taxonomy.json", repo_type="dataset"))
    with open(taxonomy_file, "r") as handle:
        return json.load(handle)


def download_full_boom_dataset(local_path: str | Path, *, repo_id: str = BOOM_REPO_ID) -> Path:
    """
    Download the full BOOM dataset repository to a local directory.

    This ports the practical notebook helper shipped in the upstream Toto repo
    into the standalone interpretability package so users do not need that repo
    checked out locally.
    """

    local_root = Path(local_path).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)
    dataset_dir = local_root / "boom_benchmark"
    if dataset_dir.exists():
        return dataset_dir

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for file_name in list_repo_files(repo_id, repo_type="dataset"):
        hf_hub_download(
            repo_id=repo_id,
            filename=file_name,
            local_dir=str(dataset_dir),
            local_dir_use_symlinks=False,
            repo_type="dataset",
        )
    return dataset_dir


def split_boom_series_ids(
    taxonomy: dict[str, dict[str, Any]],
    *,
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
) -> dict[str, list[str]]:
    series_ids = sorted(taxonomy.keys())
    rng = random.Random(seed)
    rng.shuffle(series_ids)

    num_train = int(len(series_ids) * train_fraction)
    num_val = int(len(series_ids) * val_fraction)
    return {
        "train": series_ids[:num_train],
        "val": series_ids[num_train : num_train + num_val],
        "test": series_ids[num_train + num_val :],
    }


def ensure_boom_snapshot(
    series_ids: Iterable[str],
    *,
    repo_id: str = BOOM_REPO_ID,
    extra_files: tuple[str, ...] = ("dataset_taxonomy.json",),
    local_path: Path | None = None,
) -> Path:
    if local_path is not None:
        return local_path
    allow_patterns = list(extra_files)
    allow_patterns.extend(f"{series_id}/*" for series_id in series_ids)
    snapshot_path = snapshot_download(repo_id, repo_type="dataset", allow_patterns=allow_patterns)
    return Path(snapshot_path)


def load_boom_series(snapshot_path: Path, series_id: str) -> tuple[dict[str, Any], np.ndarray]:
    dataset = load_from_disk(str(snapshot_path / series_id))
    row = dataset[0]
    target = np.asarray(row["target"], dtype=np.float32)
    if target.ndim == 1:
        target = target[None, :]
    return row, target


def build_masked_timeseries(context: torch.Tensor) -> MaskedTimeseries:
    num_variates, context_length = context.shape
    return MaskedTimeseries(
        series=context,
        padding_mask=torch.ones(num_variates, context_length, dtype=torch.bool, device=context.device),
        id_mask=torch.zeros(num_variates, context_length, dtype=torch.long, device=context.device),
        timestamp_seconds=torch.arange(context_length, device=context.device).expand(num_variates, context_length),
        time_interval_seconds=torch.ones(num_variates, dtype=torch.long, device=context.device),
        num_exogenous_variables=0,
    )


def sample_window_starts(
    total_length: int,
    *,
    context_length: int,
    patch_size: int,
    max_windows_per_series: int,
) -> tuple[list[int], int]:
    max_start = total_length - context_length - patch_size
    if max_start < 0:
        return [], -1

    heldout_start = max_start
    if max_start == 0:
        return [0], 0

    num_windows = min(max_windows_per_series, max_start + 1)
    starts = np.linspace(0, max_start, num=num_windows, dtype=int).tolist()
    starts = sorted(set(start for start in starts if start != heldout_start))
    return starts, heldout_start


def build_window_examples(
    *,
    series_id: str,
    split: str,
    snapshot_path: Path,
    taxonomy: dict[str, dict[str, Any]],
    context_length: int,
    patch_size: int,
    max_windows_per_series: int = 16,
    include_heldout_late: bool = False,
) -> list[WindowExample]:
    row, target = load_boom_series(snapshot_path, series_id)
    starts, heldout_start = sample_window_starts(
        target.shape[-1],
        context_length=context_length,
        patch_size=patch_size,
        max_windows_per_series=max_windows_per_series,
    )

    if include_heldout_late and heldout_start >= 0:
        starts = sorted(set(starts + [heldout_start]))

    taxonomy_labels = build_taxonomy_labels(series_id, taxonomy[series_id])
    examples: list[WindowExample] = []
    for start in starts:
        stop = start + context_length
        context = torch.as_tensor(target[:, start:stop], dtype=torch.float32)
        next_patch = torch.as_tensor(target[:, stop : stop + patch_size], dtype=torch.float32)

        labels = dict(taxonomy_labels)
        labels.update(
            compute_dynamic_regime_labels(
                context,
                next_patch,
                num_target_variates=int(taxonomy_labels["num_variates"]),
            )
        )
        labels["is_heldout_late"] = start == heldout_start

        examples.append(
            WindowExample(
                series_id=series_id,
                window_id=f"{series_id}:{start}",
                split=split,
                context=context,
                next_patch=next_patch,
                patch_size=patch_size,
                freq=str(row["freq"]),
                item_id=str(row["item_id"]),
                num_target_variates=int(taxonomy_labels["num_variates"]),
                labels=labels,
            )
        )
    return examples


def build_boom_windows(
    *,
    series_ids: Iterable[str],
    split: str,
    snapshot_path: Path,
    taxonomy: dict[str, dict[str, Any]],
    context_length: int,
    patch_size: int,
    max_windows_per_series: int = 16,
    include_heldout_late: bool = False,
) -> list[WindowExample]:
    windows: list[WindowExample] = []
    for series_id in series_ids:
        windows.extend(
            build_window_examples(
                series_id=series_id,
                split=split,
                snapshot_path=snapshot_path,
                taxonomy=taxonomy,
                context_length=context_length,
                patch_size=patch_size,
                max_windows_per_series=max_windows_per_series,
                include_heldout_late=include_heldout_late,
            )
        )
    return windows


def raw_features_for_window(window: WindowExample) -> torch.Tensor:
    return build_raw_baseline_features(window.context, window.next_patch)
