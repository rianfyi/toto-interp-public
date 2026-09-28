from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gdown


@dataclass(frozen=True)
class LSFArchiveSpec:
    dataset_key: str
    archive_name: str
    url: str
    required_relative_files: tuple[str, ...]


LSF_ARCHIVES: dict[str, LSFArchiveSpec] = {
    "ett": LSFArchiveSpec(
        dataset_key="ett",
        archive_name="ett.zip",
        url="https://drive.google.com/file/d/1bnrv7gpn27yO54WJI-vuXP5NclE5BlBx/view?usp=drive_link",
        required_relative_files=(
            "ETT-small/ETTh1.csv",
            "ETT-small/ETTh2.csv",
            "ETT-small/ETTm1.csv",
            "ETT-small/ETTm2.csv",
        ),
    ),
    "electricity": LSFArchiveSpec(
        dataset_key="electricity",
        archive_name="electricity.zip",
        url="https://drive.google.com/file/d/1FHH0S3d6IK_UOpg6taBRavx4MragRLo1/view?usp=drive_link",
        required_relative_files=("electricity/electricity.csv",),
    ),
    "weather": LSFArchiveSpec(
        dataset_key="weather",
        archive_name="weather.zip",
        url="https://drive.google.com/file/d/1nXdMIJ7K201Bx3IBGNiaNFQ6FzeDEzIr/view?usp=drive_link",
        required_relative_files=("weather/weather.csv",),
    ),
}

LSF_DATASET_TO_ARCHIVE = {
    "ETTh1": "ett",
    "ETTh2": "ett",
    "ETTm1": "ett",
    "ETTm2": "ett",
    "electricity": "electricity",
    "weather": "weather",
}


def default_lsf_data_path(base_dir: str | Path | None = None) -> Path:
    base = Path(base_dir) if base_dir is not None else Path.cwd()
    return base.expanduser().resolve() / "data" / "lsf_datasets"


def required_archives_for_lsf_datasets(dataset_names: Iterable[str]) -> tuple[str, ...]:
    archive_keys: list[str] = []
    for dataset_name in dataset_names:
        key = LSF_DATASET_TO_ARCHIVE.get(str(dataset_name))
        if key is None:
            raise ValueError(
                f"Unsupported LSF dataset {dataset_name!r}. Expected one of {sorted(LSF_DATASET_TO_ARCHIVE)}."
            )
        archive_keys.append(key)
    return tuple(sorted(set(archive_keys)))


def _resolve_archive_keys(requested: Iterable[str] | None) -> tuple[str, ...]:
    if requested is None:
        return tuple(LSF_ARCHIVES.keys())
    keys = [str(item).lower() for item in requested]
    invalid = [key for key in keys if key not in LSF_ARCHIVES]
    if invalid:
        raise ValueError(f"Unknown LSF archive keys: {invalid}. Expected one of {sorted(LSF_ARCHIVES)}.")
    return tuple(sorted(set(keys)))


def expected_lsf_files(root: str | Path, archive_keys: Iterable[str] | None = None) -> dict[str, Path]:
    root_path = Path(root).expanduser().resolve()
    files: dict[str, Path] = {}
    for key in _resolve_archive_keys(archive_keys):
        for relative_path in LSF_ARCHIVES[key].required_relative_files:
            files[relative_path] = root_path / relative_path
    return files


def missing_lsf_files(root: str | Path, archive_keys: Iterable[str] | None = None) -> list[str]:
    missing: list[str] = []
    for relative_path, path in expected_lsf_files(root, archive_keys).items():
        if not path.exists():
            missing.append(relative_path)
    return sorted(missing)


def normalize_lsf_layout(root: str | Path, archive_keys: Iterable[str] | None = None) -> dict[str, Path]:
    root_path = Path(root).expanduser().resolve()
    root_path.mkdir(parents=True, exist_ok=True)

    resolved: dict[str, Path] = {}
    for relative_path, destination in expected_lsf_files(root_path, archive_keys).items():
        if destination.exists():
            resolved[relative_path] = destination
            continue

        file_name = destination.name
        candidates = [candidate for candidate in root_path.rglob(file_name) if candidate.is_file() and candidate != destination]
        if not candidates:
            continue

        source = candidates[0]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        resolved[relative_path] = destination
    return resolved


def _download_archive(spec: LSFArchiveSpec, archive_dir: Path, *, force: bool = False) -> Path:
    archive_path = archive_dir / spec.archive_name
    if archive_path.exists() and not force:
        return archive_path

    archive_dir.mkdir(parents=True, exist_ok=True)
    downloaded = gdown.download(url=spec.url, output=str(archive_path), quiet=False)
    if downloaded is None:
        raise RuntimeError(f"Failed to download LSF archive for {spec.dataset_key}.")
    return Path(downloaded)


def _extract_archive(archive_path: Path, destination_root: Path) -> None:
    shutil.unpack_archive(str(archive_path), extract_dir=str(destination_root))


def download_lsf_datasets(
    root: str | Path,
    *,
    archive_keys: Iterable[str] | None = None,
    force: bool = False,
) -> Path:
    root_path = Path(root).expanduser().resolve()
    archive_dir = root_path / "_archives"
    for key in _resolve_archive_keys(archive_keys):
        spec = LSF_ARCHIVES[key]
        archive_path = _download_archive(spec, archive_dir, force=force)
        _extract_archive(archive_path, root_path)
    normalize_lsf_layout(root_path, archive_keys)
    return root_path


def validate_lsf_layout(root: str | Path, archive_keys: Iterable[str] | None = None) -> tuple[bool, list[str]]:
    missing = missing_lsf_files(root, archive_keys)
    return len(missing) == 0, missing


def ensure_lsf_datasets(
    root: str | Path,
    *,
    archive_keys: Iterable[str] | None = None,
    download: bool = False,
    force: bool = False,
) -> Path:
    root_path = Path(root).expanduser().resolve()
    normalize_lsf_layout(root_path, archive_keys)
    is_valid, missing = validate_lsf_layout(root_path, archive_keys)
    if is_valid:
        return root_path

    if download:
        download_lsf_datasets(root_path, archive_keys=archive_keys, force=force)
        normalize_lsf_layout(root_path, archive_keys)
        is_valid, missing = validate_lsf_layout(root_path, archive_keys)
        if is_valid:
            return root_path

    required = "\n".join(f"- {item}" for item in missing)
    raise FileNotFoundError(
        "LSF datasets are missing from the expected local layout.\n"
        f"Root: {root_path}\n"
        f"Missing files:\n{required}\n\n"
        "Use `python scripts/download_lsf_datasets.py --output-dir data/lsf_datasets` to fetch them, "
        "or place the CSVs under the expected structure documented in docs/lsf_setup.md."
    )
