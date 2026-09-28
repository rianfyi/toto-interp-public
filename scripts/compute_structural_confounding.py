from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.confounds import (
    STRUCTURAL_CONFOUND_LABELS,
    pairwise_cramers_v,
    series_label_frame,
)
from toto_interp.types import ActivationBatch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure pairwise structural-label association on series-level activation metadata."
    )
    parser.add_argument("--activation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=list(STRUCTURAL_CONFOUND_LABELS),
        choices=STRUCTURAL_CONFOUND_LABELS,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    batch = ActivationBatch.concatenate([ActivationBatch.load(path) for path in args.activation_files])
    frame = series_label_frame(batch, args.labels)
    all_rows: list[pd.DataFrame] = []
    tables: dict[str, object] = {}
    for split in ("all", "train", "val", "test"):
        rows, split_tables = pairwise_cramers_v(frame, args.labels, split=split)
        all_rows.append(rows)
        tables.update(split_tables)
    results = pd.concat(all_rows, ignore_index=True)
    results.to_csv(args.output_dir / "pairwise_cramers_v.csv", index=False)
    with open(args.output_dir / "pairwise_cramers_v_tables.json", "w") as handle:
        json.dump(
            {
                "unit": "series",
                "labels": list(args.labels),
                "series_count": int(len(frame)),
                "tables": tables,
            },
            handle,
            indent=2,
        )


if __name__ == "__main__":
    main()
