from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _normalize_command(command: list[str]) -> list[str]:
    """Canonicalize absolute path arguments before comparing resume signatures."""
    return [
        str(Path(token).resolve(strict=False)) if token.startswith("/") else token
        for token in command
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit taxonomy-control unconditional and conditional probes from completed activation dumps."
    )
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-jobs", type=int, default=8)
    parser.add_argument("--raw-width", type=int, default=32)
    parser.add_argument("--raw-layers", type=int, default=3)
    parser.add_argument("--raw-modes", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--skip-post-controls",
        action="store_true",
        help="Release the parallel allocation after unconditional probes; run conditional/holdout controls separately.",
    )
    parser.add_argument(
        "--structural-holdout-target", choices=("frequency_bucket", "metric_type"), default="frequency_bucket"
    )
    parser.add_argument("--structural-holdout-mode", choices=("combination", "domain"), default="combination")
    parser.add_argument("--structural-holdout-rotation", type=int, default=0)
    return parser.parse_args()


def run(command: list[str], done_path: Path, reuse_existing: bool) -> None:
    signature_path = done_path.with_name(f"{done_path.name}.command.json")
    signature = {"command": _normalize_command(command)}
    if reuse_existing and done_path.exists():
        try:
            prior = json.loads(signature_path.read_text())
            prior_signature = {
                "command": _normalize_command(prior.get("command", []))
            }
            if prior_signature == signature:
                print("Skipping completed step with matching command signature:", done_path)
                return
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        print("Recomputing completed step with missing or incompatible command signature:", done_path)
    print("$", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)
    signature_path.write_text(json.dumps(signature, indent=2))


def activation_files(root: Path) -> list[str]:
    return [str(root / f"{split}_activations.pt") for split in ("train", "val", "test")]


def window_files(root: Path) -> list[str]:
    return [str(root / f"{split}_windows.pt") for split in ("train", "val", "test")]


def main() -> None:
    args = parse_args()
    seed_root = args.runs_root / f"seed_{args.seed}"
    pretrained_root = seed_root / "pretrained_activations"
    confounding_dir = seed_root / "structural_confounding"
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compute_structural_confounding.py"),
            "--activation-files",
            *activation_files(pretrained_root),
            "--output-dir",
            str(confounding_dir),
        ],
        confounding_dir / "pairwise_cramers_v.csv",
        args.reuse_existing,
    )
    raw_gbdt_dir = seed_root / "raw_controls" / "gbdt"
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "fit_toto_probes.py"),
            "--activation-files",
            *activation_files(pretrained_root),
            "--window-files",
            *window_files(pretrained_root),
            "--output-dir",
            str(raw_gbdt_dir),
            "--method",
            "gbdt",
            "--label-group",
            "taxonomy",
            "--seed",
            str(args.seed),
            "--device",
            "cpu",
            "--fno-width",
            str(args.raw_width),
            "--fno-layers",
            str(args.raw_layers),
            "--fno-modes",
            str(args.raw_modes),
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
        ],
        raw_gbdt_dir / "probe_results.csv",
        args.reuse_existing,
    )
    for source, activation_root in (
        ("pretrained", seed_root / "pretrained_activations"),
        ("random_init", seed_root / "random_init_activations"),
    ):
        output_dir = seed_root / "unconditional" / source
        unconditional_command = [
            sys.executable,
            str(ROOT / "scripts" / "fit_toto_probes.py"),
            "--activation-files",
            *activation_files(activation_root),
            "--output-dir",
            str(output_dir),
            "--method",
            "linear_probe",
            "--label-group",
            "taxonomy",
            "--seed",
            str(args.seed),
            "--n-jobs",
            str(args.n_jobs),
        ]
        if args.reuse_existing:
            unconditional_command.append("--reuse-artifacts")
        run(
            unconditional_command,
            output_dir / "probe_results.csv",
            args.reuse_existing,
        )
        if args.skip_post_controls:
            continue

        structural_holdout_dir = (
            seed_root
            / "structural_holdout"
            / args.structural_holdout_mode
            / args.structural_holdout_target
            / f"rotation_{args.structural_holdout_rotation}"
            / source
        )
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_structural_holdout_probes.py"),
                "--activation-files",
                *activation_files(activation_root),
                "--output-dir",
                str(structural_holdout_dir),
                "--seed",
                str(args.seed),
                "--target-label",
                args.structural_holdout_target,
                "--holdout-mode",
                args.structural_holdout_mode,
                "--rotation",
                str(args.structural_holdout_rotation),
            ],
            structural_holdout_dir / "structural_holdout_selected.csv",
            args.reuse_existing,
        )

        conditional_dir = seed_root / "conditional" / source
        run(
            [
                sys.executable,
                str(ROOT / "scripts" / "run_conditional_probes.py"),
                "--activation-files",
                *activation_files(activation_root),
                "--output-dir",
                str(conditional_dir),
                "--seed",
                str(args.seed),
            ],
            conditional_dir / "conditional_probe_selected.csv",
            args.reuse_existing,
        )


if __name__ == "__main__":
    main()
