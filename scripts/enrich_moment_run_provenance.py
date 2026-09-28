from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from toto_interp.boom import load_boom_taxonomy, split_boom_series_ids
from toto_interp.types import ProbeArtifact


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Enrich legacy MOMENT run metadata with split/eval provenance that is "
            "derivable from the recorded seed, split hashes, and eval-only loader."
        )
    )
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--snapshot-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--weight-source",
        choices=("pretrained", "random_init"),
        default="pretrained",
    )
    args = parser.parse_args()

    activation_root = args.seed_root / "activations"
    summary_path = activation_root / "activation_dump_summary.json"
    original_summary_sha256 = sha256_file(summary_path)
    summary = json.loads(summary_path.read_text())
    original_summary_sha256 = str(
        summary.get("pre_provenance_enrichment_sha256") or original_summary_sha256
    )
    if int(summary.get("seed", -1)) != args.seed:
        raise ValueError(
            f"Activation summary seed={summary.get('seed')} does not match {args.seed}."
        )
    if str(summary.get("weight_source")) != args.weight_source:
        raise ValueError(
            f"Activation summary weight_source={summary.get('weight_source')!r}, "
            f"expected {args.weight_source!r}."
        )

    taxonomy = load_boom_taxonomy(local_path=args.snapshot_path)
    split_ids = split_boom_series_ids(taxonomy, seed=args.seed)
    max_series = int(summary.get("max_series_per_split", 0))
    if max_series > 0:
        split_ids = {name: ids[:max_series] for name, ids in split_ids.items()}
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    overlap = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError(f"Reconstructed BOOM splits overlap: {overlap}")
    split_manifest = {
        name: {
            "series_count": len(ids),
            "series_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        }
        for name, ids in split_ids.items()
    }
    for split_name, values in split_manifest.items():
        recorded = summary["splits"][split_name]
        if int(recorded["series_count"]) != int(values["series_count"]):
            raise ValueError(f"{split_name} series count mismatch during enrichment.")
        if str(recorded["series_id_sha256"]) != str(values["series_id_sha256"]):
            raise ValueError(f"{split_name} series hash mismatch during enrichment.")

    attestation = {
        "schema_version": 1,
        "method": "reconstructed_seeded_splits_and_eval_only_loader_invariant",
        "measurement_values_unchanged": True,
        "loader_function": "load_moment_with_fallback",
        "weight_source": args.weight_source,
    }
    summary.update(
        {
            "task_name": "embedding",
            "model_training": False,
            "split_overlap_counts": overlap,
            "split_manifest": split_manifest,
            "pre_provenance_enrichment_sha256": original_summary_sha256,
            "provenance_enrichment": attestation,
        }
    )
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    series_hashes = {
        name: values["series_id_sha256"] for name, values in split_manifest.items()
    }
    enriched = 0
    for probe_dir in ("probes", "dynamic_probes"):
        artifact_root = args.seed_root / probe_dir / "artifacts"
        if not artifact_root.exists():
            continue
        for artifact_path in sorted(artifact_root.glob("*.pt")):
            probe = ProbeArtifact.load(artifact_path)
            original_sha256 = str(
                probe.artifact_metadata.get("pre_provenance_enrichment_sha256")
                or sha256_file(artifact_path)
            )
            raw_signature = probe.artifact_metadata.get("activation_source_signature")
            try:
                signature = json.loads(raw_signature)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"Probe {artifact_path} has no parseable source signature."
                ) from exc
            if int(signature.get("seed", -1)) != args.seed:
                raise ValueError(f"Probe {artifact_path} has the wrong activation seed.")
            signature.update(
                {
                    "task_name": "embedding",
                    "model_training": False,
                    "split_overlap_counts": overlap,
                    "series_id_sha256_by_split": series_hashes,
                    "provenance_enrichment": attestation,
                }
            )
            probe.artifact_metadata.update(
                {
                    "activation_source_signature": json.dumps(
                        signature, sort_keys=True, default=str
                    ),
                    "pre_provenance_enrichment_sha256": original_sha256,
                    "provenance_enrichment": attestation,
                }
            )
            probe.save(artifact_path)
            enriched += 1

    manifest = {
        "seed": args.seed,
        "weight_source": args.weight_source,
        "activation_summary": str(summary_path),
        "pre_provenance_enrichment_summary_sha256": original_summary_sha256,
        "split_manifest": split_manifest,
        "attestation": attestation,
        "enriched_probe_artifact_count": enriched,
    }
    (args.seed_root / "provenance_enrichment_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
