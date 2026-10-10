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
        description="Run the taxonomy-control stress tests: raw-window controls and conditional structural probes."
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--context-length", type=int, default=1024)
    parser.add_argument("--max-series-per-split", type=int, default=500)
    parser.add_argument("--max-windows-per-series", type=int, default=4)
    parser.add_argument("--raw-width", type=int, default=32)
    parser.add_argument("--raw-layers", type=int, default=3)
    parser.add_argument("--raw-modes", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--structural-holdout-target", choices=("frequency_bucket", "metric_type"), default="frequency_bucket"
    )
    parser.add_argument("--structural-holdout-mode", choices=("combination", "domain"), default="combination")
    parser.add_argument(
        "--structural-holdout-rotation",
        type=int,
        default=0,
        help="Deterministic structural-holdout rotation; use 0--4 across the five seeded runs.",
    )
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Stop after the raw-window controls; useful when probe fitting is dispatched separately on CPUs.",
    )
    return parser.parse_args()


def run(command: list[str], *, done_path: Path | None, reuse_existing: bool) -> None:
    signature_path = None if done_path is None else done_path.with_name(f"{done_path.name}.command.json")
    signature = {"command": _normalize_command(command)}
    if reuse_existing and done_path is not None and done_path.exists() and signature_path is not None:
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
    if signature_path is not None:
        signature_path.write_text(json.dumps(signature, indent=2))


def activation_args(args: argparse.Namespace, *, output_dir: Path, weight_source: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "dump_toto_activations.py"),
        "--output-dir",
        str(output_dir),
        "--device",
        args.device,
        "--seed",
        str(args.seed),
        "--context-length",
        str(args.context_length),
        "--max-series-per-split",
        str(args.max_series_per_split),
        "--max-windows-per-series",
        str(args.max_windows_per_series),
        "--snapshot-path",
        str(args.snapshot_path),
        "--pooling-modes",
        "series_mean",
        "--weight-source",
        weight_source,
    ]


def activation_files(root: Path) -> list[str]:
    return [str(root / f"{split}_activations.pt") for split in ("train", "val", "test")]


def window_files(root: Path) -> list[str]:
    return [str(root / f"{split}_windows.pt") for split in ("train", "val", "test")]


def main() -> None:
    args = parse_args()
    seed_root = args.output_root / f"seed_{args.seed}"
    pretrained = seed_root / "pretrained_activations"
    random_init = seed_root / "random_init_activations"
    seed_root.mkdir(parents=True, exist_ok=True)

    run(
        activation_args(args, output_dir=pretrained, weight_source="pretrained"),
        done_path=pretrained / "activation_dump_summary.json",
        reuse_existing=args.reuse_existing,
    )
    run(
        activation_args(args, output_dir=random_init, weight_source="random_init"),
        done_path=random_init / "activation_dump_summary.json",
        reuse_existing=args.reuse_existing,
    )

    confounding_dir = seed_root / "structural_confounding"
    run(
        [
            sys.executable,
            str(ROOT / "scripts" / "compute_structural_confounding.py"),
            "--activation-files",
            *activation_files(pretrained),
            "--output-dir",
            str(confounding_dir),
        ],
        done_path=confounding_dir / "pairwise_cramers_v.csv",
        reuse_existing=args.reuse_existing,
    )

    for method in ("fno", "cnn", "transformer", "gbdt"):
        output_dir = seed_root / "raw_controls" / method
        command = [
            sys.executable,
            str(ROOT / "scripts" / "fit_toto_probes.py"),
            "--activation-files",
            *activation_files(pretrained),
            "--window-files",
            *window_files(pretrained),
            "--output-dir",
            str(output_dir),
            "--method",
            method,
            "--label-group",
            "taxonomy",
            "--seed",
            str(args.seed),
            "--device",
            "cpu" if method == "gbdt" else args.device,
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
        ]
        run(command, done_path=output_dir / "probe_results.csv", reuse_existing=args.reuse_existing)

    if args.raw_only:
        raw_manifest = {
            "seed": args.seed,
            "raw_control_methods": ["fno", "cnn", "transformer", "gbdt"],
            "structural_confounding": str(confounding_dir / "pairwise_cramers_v.csv"),
            "snapshot_path": str(args.snapshot_path),
            "context_length": args.context_length,
            "max_series_per_split": args.max_series_per_split,
            "max_windows_per_series": args.max_windows_per_series,
            "raw_hyperparameters": {
                "width": args.raw_width,
                "layers": args.raw_layers,
                "modes": args.raw_modes,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
            },
        }
        (seed_root / "taxonomy_raw_manifest.json").write_text(json.dumps(raw_manifest, indent=2))
        return

    for source_name, activation_root in (("pretrained", pretrained), ("random_init", random_init)):
        output_dir = seed_root / "unconditional" / source_name
        command = [
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
            "4",
        ]
        if args.reuse_existing:
            command.append("--reuse-artifacts")
        run(command, done_path=output_dir / "probe_results.csv", reuse_existing=args.reuse_existing)

    for source_name, activation_root in (("pretrained", pretrained), ("random_init", random_init)):
        output_dir = seed_root / "conditional" / source_name
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_conditional_probes.py"),
            "--activation-files",
            *activation_files(activation_root),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(args.seed),
        ]
        run(command, done_path=output_dir / "conditional_probe_selected.csv", reuse_existing=args.reuse_existing)

    for source_name, activation_root in (("pretrained", pretrained), ("random_init", random_init)):
        output_dir = seed_root / "structural_holdout" / source_name
        command = [
            sys.executable,
            str(ROOT / "scripts" / "run_structural_holdout_probes.py"),
            "--activation-files",
            *activation_files(activation_root),
            "--output-dir",
            str(output_dir),
            "--seed",
            str(args.seed),
            "--target-label",
            args.structural_holdout_target,
            "--holdout-mode",
            args.structural_holdout_mode,
            "--rotation",
            str(args.structural_holdout_rotation),
        ]
        run(command, done_path=output_dir / "structural_holdout_selected.csv", reuse_existing=args.reuse_existing)

    manifest = {
        "seed": args.seed,
        "pretrained_activation_dir": str(pretrained),
        "random_init_activation_dir": str(random_init),
        "raw_control_methods": ["fno", "cnn", "transformer", "gbdt"],
        "unconditional_sources": ["pretrained", "random_init"],
        "conditional_sources": ["pretrained", "random_init"],
        "snapshot_path": str(args.snapshot_path),
        "context_length": args.context_length,
        "max_series_per_split": args.max_series_per_split,
        "max_windows_per_series": args.max_windows_per_series,
        "structural_confounding": str(confounding_dir / "pairwise_cramers_v.csv"),
        "structural_holdout": {
            "target": args.structural_holdout_target,
            "mode": args.structural_holdout_mode,
            "rotation": args.structural_holdout_rotation,
            "sources": ["pretrained", "random_init"],
        },
        "raw_hyperparameters": {
            "width": args.raw_width,
            "layers": args.raw_layers,
            "modes": args.raw_modes,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
        },
    }
    (seed_root / "taxonomy_suite_manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
