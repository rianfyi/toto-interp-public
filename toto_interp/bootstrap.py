from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def ensure_toto_importable() -> Path | None:
    """
    Ensure the upstream Toto package is importable.

    Import order:
    1. Existing Python environment if `toto` is already installed.
    2. `TOTO_REPO_PATH` environment variable, if provided for local development.
    """

    if importlib.util.find_spec("toto") is not None:
        return None

    candidates: list[Path] = []
    env_path = os.environ.get("TOTO_REPO_PATH")
    if env_path:
        candidates.append(Path(env_path).expanduser())

    for candidate in candidates:
        if not candidate.exists():
            continue
        candidate = candidate.resolve()
        if str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))
        if importlib.util.find_spec("toto") is not None:
            return candidate

    raise ModuleNotFoundError(
        "Could not import the Toto library. Install `toto-ts` or set TOTO_REPO_PATH to a local Toto checkout."
    )
