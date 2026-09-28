from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.lsf import LSF_ARCHIVES, download_lsf_datasets, ensure_lsf_datasets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and normalize the official public LSF datasets.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "lsf_datasets")
    parser.add_argument("--datasets", nargs="*", choices=sorted(LSF_ARCHIVES), default=sorted(LSF_ARCHIVES))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.validate_only:
        resolved = ensure_lsf_datasets(args.output_dir, archive_keys=args.datasets, download=False)
    else:
        download_lsf_datasets(args.output_dir, archive_keys=args.datasets, force=args.force)
        resolved = ensure_lsf_datasets(args.output_dir, archive_keys=args.datasets, download=False)
    print(resolved)


if __name__ == "__main__":
    main()
