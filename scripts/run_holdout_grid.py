from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_structural_holdout_probes import fit_structural_holdout
from scripts.run_conditional_probes import fit_conditional_probes
from toto_interp.conditional import aggregate_series_records
from toto_interp.types import ActivationBatch

HOLDOUT_GRID = (
    ("frequency_bucket", "combination"),
    ("metric_type", "combination"),
    ("frequency_bucket", "domain"),
    ("metric_type", "domain"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the taxonomy-control structural holdouts with one activation load per source."
    )
    parser.add_argument("--activation-files", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conditional-output-dir", type=Path, required=True)
    parser.add_argument("--source", choices=("pretrained", "random_init"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--rotation", type=int, required=True)
    parser.add_argument(
        "--token-positions",
        nargs="+",
        choices=("final_context", "first_decode"),
        default=("final_context", "first_decode"),
    )
    parser.add_argument("--reuse-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pending = []
    for target_label, holdout_mode in HOLDOUT_GRID:
        output_dir = (
            args.output_root
            / holdout_mode
            / target_label
            / f"rotation_{args.rotation}"
            / args.source
        )
        done_path = output_dir / "structural_holdout_selected.csv"
        if args.reuse_existing and done_path.exists():
            print("Skipping completed structural holdout:", done_path)
            continue
        pending.append((target_label, holdout_mode, output_dir))
    conditional_done = (
        args.conditional_output_dir / "conditional_probe_selected.csv"
    )
    conditional_pending = not (
        args.reuse_existing and conditional_done.exists()
    )
    if not pending and not conditional_pending:
        return

    batch = ActivationBatch.concatenate([ActivationBatch.load(path) for path in args.activation_files])
    batch = aggregate_series_records(batch)
    for target_label, holdout_mode, output_dir in pending:
        fit_structural_holdout(
            batch,
            output_dir=output_dir,
            seed=args.seed,
            target_label=target_label,
            holdout_mode=holdout_mode,
            rotation=args.rotation,
            token_positions=tuple(args.token_positions),
        )
    if conditional_pending:
        fit_conditional_probes(
            batch,
            output_dir=args.conditional_output_dir,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
