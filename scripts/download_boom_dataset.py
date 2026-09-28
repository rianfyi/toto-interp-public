from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.boom import download_full_boom_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the full public BOOM dataset locally.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_dir = download_full_boom_dataset(args.output_dir)
    print(dataset_dir)


if __name__ == "__main__":
    main()
