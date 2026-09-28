from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.boom import BOOM_REPO_ID, load_boom_taxonomy, split_boom_series_ids
from toto_interp.loader import load_toto_with_fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prestage BOOM series and the Toto checkpoint on a networked login node.")
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--max-series-per-split", type=int, default=500)
    parser.add_argument("--model-id", type=str, default="Datadog/Toto-Open-Base-1.0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    taxonomy = load_boom_taxonomy()
    series_ids: set[str] = set()
    for seed in args.seeds:
        splits = split_boom_series_ids(taxonomy, seed=seed)
        for ids in splits.values():
            series_ids.update(ids[: args.max_series_per_split])
    snapshot_download(
        BOOM_REPO_ID,
        repo_type="dataset",
        local_dir=str(args.snapshot_path),
        allow_patterns=["dataset_taxonomy.json", *[f"{series_id}/*" for series_id in sorted(series_ids)]],
    )
    model = load_toto_with_fallback(args.model_id, device="cpu")
    del model
    args.snapshot_path.mkdir(parents=True, exist_ok=True)
    (args.snapshot_path / "reviewer3_stage_manifest.json").write_text(
        json.dumps(
            {
                "seeds": args.seeds,
                "max_series_per_split": args.max_series_per_split,
                "series_count": len(series_ids),
                "model_id": args.model_id,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
