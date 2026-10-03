"""
Dump MOMENT activation traces over BOOM research windows.

Mirrors `dump_toto_activations.py` but routes through MOMENT (T5-encoder) so
the existing probe-fitting code (`fit_toto_probes.py`) can consume the result
without modification. Output schema is identical: per-window per-layer
activations with the same `labels` dict (taxonomy metadata + dynamic regime
features), so both backbones share the probe-fitting code and label definitions.

Scope notes:
  - MOMENT uses fixed patch_size=8 (Toto uses 64); we adjust window construction
    accordingly. The dynamic regime labels (future_burstiness, etc.) are
    computed from the raw context+next_patch and the structural taxonomy labels
    (metric_type, domain, ...) come from BOOM metadata; both are model-agnostic.
  - MOMENT is channel-independent: each variate is processed as its own
    univariate sequence inside the encoder. Per-variate pooling captures this
    natively; series_mean averages across channels.
  - Default seq_len=512, layers=[3, 6, 9, 11] (the layers read in the
    paper); a four-layer slice tests whether the taxonomy readouts recur
    without a full layer sweep.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from toto_interp.boom import (
    build_boom_windows,
    ensure_boom_snapshot,
    load_boom_taxonomy,
    raw_features_for_window,
    split_boom_series_ids,
)
from toto_interp.labels import RAW_FEATURE_NAMES
from toto_interp.loader import resolve_device
from toto_interp.moment_loader import (
    MOMENT_BASE_ID,
    load_moment_with_fallback,
    moment_hidden_dim,
    moment_layer,
    moment_num_layers,
    moment_patch_size,
)
from toto_interp.types import ActivationBatch, WindowDataset, WindowExample

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--model-id", type=str, default=MOMENT_BASE_ID)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--weight-source", choices=("pretrained", "random_init"), default="pretrained")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq-len", type=int, default=512,
                   help="MOMENT context length. Must be a multiple of 8.")
    p.add_argument("--max-windows-per-series", type=int, default=4)
    p.add_argument("--max-series-per-split", type=int, default=200)
    p.add_argument("--splits", nargs="+", choices=("train", "val", "test"),
                   default=["train", "val", "test"])
    p.add_argument("--layers", type=int, nargs="+", default=[3, 6, 9, 11])
    p.add_argument("--token-positions", nargs="+", choices=("all_context", "final_context"),
                   default=["all_context", "final_context"])
    p.add_argument("--pooling-modes", nargs="+", choices=("per_variate", "series_mean"),
                   default=["per_variate", "series_mean"])
    p.add_argument("--snapshot-path", type=Path, default=None)
    p.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), default="fp32")
    return p.parse_args()


# ---------------------------------------------------------------------------
# MOMENT forward pass with hooks
# ---------------------------------------------------------------------------


class _Recorder:
    def __init__(self, layers: tuple[int, ...]):
        self.layers = layers
        self.records: dict[int, torch.Tensor] = {}

    def hook(self, idx: int):
        def fn(_module, _inputs, output):
            if idx in self.records:
                return
            # T5 encoder block returns a tuple; the first element is the hidden state
            tensor = output[0] if isinstance(output, tuple) else output
            if not isinstance(tensor, torch.Tensor):
                return
            self.records[idx] = tensor.detach().cpu().float()

        return fn


@torch.no_grad()
def extract_moment_activations(
    pipeline,
    windows: list[WindowExample],
    *,
    layers: tuple[int, ...],
    seq_len: int,
    token_positions: tuple[str, ...],
    pooling_modes: tuple[str, ...],
    device: str,
) -> ActivationBatch:
    """
    Forward each window through MOMENT, capture per-layer hidden states, and
    flatten into the ActivationBatch schema.

    MOMENT input: (B, n_channels, seq_len) with input_mask of shape (B, seq_len).
    The encoder reshapes to (B*n_channels, n_patches, hidden_dim) internally and
    each block's output retains that shape; we reshape back to
    (B, n_channels, n_patches, hidden_dim) for the per-variate / series-mean
    pooling we use elsewhere.
    """
    storage = {
        "activations": [], "raw_features": [], "layer_indices": [],
        "patch_indices": [], "variate_indices": [], "token_positions": [],
        "pooling_modes": [], "series_ids": [], "window_ids": [], "splits": [],
        "labels": defaultdict(list),
    }
    hidden_dim = moment_hidden_dim(pipeline)

    for window in windows:
        context = window.context.float()  # (V, T_orig)
        n_var, t_orig = context.shape

        # Pad / truncate to seq_len
        if t_orig >= seq_len:
            ctx_in = context[:, -seq_len:]  # right-align
            mask_in = torch.ones(seq_len, dtype=torch.float32)
        else:
            pad = seq_len - t_orig
            ctx_in = torch.nn.functional.pad(context, (pad, 0))  # left-pad
            mask_in = torch.cat([torch.zeros(pad), torch.ones(t_orig)])

        x_enc = ctx_in.unsqueeze(0).to(device=device, dtype=next(pipeline.parameters()).dtype)
        input_mask = mask_in.unsqueeze(0).to(device=device)

        recorder = _Recorder(tuple(layers))
        handles = []
        for idx in layers:
            handles.append(moment_layer(pipeline, idx).register_forward_hook(recorder.hook(idx)))

        try:
            _ = pipeline(x_enc=x_enc, input_mask=input_mask)
        finally:
            for h in handles:
                h.remove()

        raw_feat = raw_features_for_window(window).cpu()

        for layer_idx, hs in recorder.records.items():
            # hs shape after T5 block: (B*n_channels, n_patches, hidden) typically
            # MOMENT-specific reshape: (1*n_var, n_patches, hidden) -> (1, n_var, n_patches, hidden)
            if hs.dim() == 3:
                bn, n_patches, h = hs.shape
                if bn % n_var == 0:
                    hs = hs.view(bn // n_var, n_var, n_patches, h)
                else:
                    # Fall back: assume batch dim 1, treat as (1, ?, n_patches, h)
                    hs = hs.view(1, bn, n_patches, h)
            elif hs.dim() == 4:
                pass  # already (B, V, P, H)
            else:
                logger.warning("Unexpected hidden shape %s at layer %d; skipping", hs.shape, layer_idx)
                continue

            data = hs[0]  # (V, P, H)
            v, p, h = data.shape

            for tp in token_positions:
                if tp == "all_context":
                    selected = data  # (V, P, H)
                    patch_idx_list = list(range(p))
                elif tp == "final_context":
                    selected = data[:, -1:, :]  # (V, 1, H)
                    patch_idx_list = [p - 1]
                else:
                    continue

                if "per_variate" in pooling_modes:
                    flat = selected.permute(1, 0, 2).reshape(-1, h)  # (P*V, H)
                    repeated_p = [pi for pi in patch_idx_list for _ in range(v)]
                    repeated_v = [vi for _ in patch_idx_list for vi in range(v)]
                    for row_idx in range(flat.shape[0]):
                        storage["activations"].append(flat[row_idx])
                        storage["raw_features"].append(raw_feat)
                        storage["layer_indices"].append(layer_idx)
                        storage["patch_indices"].append(repeated_p[row_idx])
                        storage["variate_indices"].append(repeated_v[row_idx])
                        storage["token_positions"].append(tp)
                        storage["pooling_modes"].append("per_variate")
                        storage["series_ids"].append(window.series_id)
                        storage["window_ids"].append(window.window_id)
                        storage["splits"].append(window.split)
                        for ln, lv in window.labels.items():
                            storage["labels"][ln].append(lv)

                if "series_mean" in pooling_modes:
                    series = selected.mean(dim=0)  # (P, H) for all_context, (1, H) for final
                    for row_idx in range(series.shape[0]):
                        storage["activations"].append(series[row_idx])
                        storage["raw_features"].append(raw_feat)
                        storage["layer_indices"].append(layer_idx)
                        storage["patch_indices"].append(patch_idx_list[row_idx])
                        storage["variate_indices"].append(-1)
                        storage["token_positions"].append(tp)
                        storage["pooling_modes"].append("series_mean")
                        storage["series_ids"].append(window.series_id)
                        storage["window_ids"].append(window.window_id)
                        storage["splits"].append(window.split)
                        for ln, lv in window.labels.items():
                            storage["labels"][ln].append(lv)

    if not storage["activations"]:
        raise ValueError("No activations were captured.")

    return ActivationBatch(
        activations=torch.stack(storage["activations"]).to(torch.float32),
        raw_features=torch.stack(storage["raw_features"]).to(torch.float32),
        raw_feature_names=RAW_FEATURE_NAMES,
        layer_indices=torch.tensor(storage["layer_indices"], dtype=torch.long),
        patch_indices=torch.tensor(storage["patch_indices"], dtype=torch.long),
        variate_indices=torch.tensor(storage["variate_indices"], dtype=torch.long),
        token_positions=list(storage["token_positions"]),
        pooling_modes=list(storage["pooling_modes"]),
        series_ids=list(storage["series_ids"]),
        window_ids=list(storage["window_ids"]),
        splits=list(storage["splits"]),
        labels={n: list(v) for n, v in storage["labels"].items()},
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    dtype_map = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    logger.info("Loading MOMENT %s on %s (%s, weight_source=%s)",
                args.model_id, device, args.dtype, args.weight_source)

    pipeline = load_moment_with_fallback(
        args.model_id,
        device=device,
        weight_source=args.weight_source,
        task_name="embedding",
        seq_len=args.seq_len,
        dtype=dtype,
    )
    if pipeline.training:
        raise RuntimeError("MOMENT must be in evaluation mode during activation extraction.")
    patch_size = moment_patch_size(pipeline)
    n_layers = moment_num_layers(pipeline)
    logger.info("MOMENT loaded: patch_size=%d, n_layers=%d, hidden_dim=%d",
                patch_size, n_layers, moment_hidden_dim(pipeline))
    invalid_layers = [l for l in args.layers if l < 0 or l >= n_layers]
    if invalid_layers:
        raise ValueError(f"Layers {invalid_layers} out of range for MOMENT (0..{n_layers - 1})")

    if args.seq_len % patch_size != 0:
        raise ValueError(f"seq_len ({args.seq_len}) must be divisible by patch_size ({patch_size})")

    taxonomy = load_boom_taxonomy(local_path=args.snapshot_path)
    split_ids = split_boom_series_ids(taxonomy, seed=args.seed)
    if args.max_series_per_split > 0:
        split_ids = {
            k: v[: args.max_series_per_split] for k, v in split_ids.items()
        }
    split_sets = {name: set(ids) for name, ids in split_ids.items()}
    split_overlap_counts = {
        "train_val": len(split_sets["train"] & split_sets["val"]),
        "train_test": len(split_sets["train"] & split_sets["test"]),
        "val_test": len(split_sets["val"] & split_sets["test"]),
    }
    if any(split_overlap_counts.values()):
        raise RuntimeError(f"BOOM series splits overlap: {split_overlap_counts}")
    split_manifest = {
        name: {
            "series_count": len(ids),
            "series_id_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        }
        for name, ids in split_ids.items()
    }
    series_id_sha256_by_split = {
        name: str(values["series_id_sha256"])
        for name, values in split_manifest.items()
    }
    snapshot_path = ensure_boom_snapshot(
        [sid for ids in split_ids.values() for sid in ids],
        local_path=args.snapshot_path,
    )

    source_metadata = {
        "model_id": args.model_id,
        "weight_source": args.weight_source,
        "task_name": "embedding",
        "seed": args.seed,
        "context_length": args.seq_len,
        "patch_size": patch_size,
        "max_series_per_split": args.max_series_per_split,
        "max_windows_per_series": args.max_windows_per_series,
        "model_training": bool(pipeline.training),
        "split_overlap_counts": split_overlap_counts,
        "series_id_sha256_by_split": series_id_sha256_by_split,
    }
    summary = {
        "seed": args.seed,
        "model_id": args.model_id,
        "weight_source": args.weight_source,
        "task_name": "embedding",
        "patch_size": patch_size,
        "context_length": args.seq_len,
        "max_series_per_split": args.max_series_per_split,
        "max_windows_per_series": args.max_windows_per_series,
        "snapshot_path": None if args.snapshot_path is None else str(args.snapshot_path),
        "device": device,
        "dtype": args.dtype,
        "model_training": bool(pipeline.training),
        "layers": args.layers,
        "token_positions": args.token_positions,
        "pooling_modes": args.pooling_modes,
        "split_overlap_counts": split_overlap_counts,
        "split_manifest": split_manifest,
        "splits": {},
    }

    for split_name, series_ids in split_ids.items():
        if split_name not in args.splits:
            continue
        windows = build_boom_windows(
            series_ids=series_ids,
            split=split_name,
            snapshot_path=snapshot_path,
            taxonomy=taxonomy,
            context_length=args.seq_len,
            patch_size=patch_size,
            max_windows_per_series=args.max_windows_per_series,
            include_heldout_late=False,
        )
        logger.info("split=%s: %d windows", split_name, len(windows))

        batch = extract_moment_activations(
            pipeline,
            windows,
            layers=tuple(args.layers),
            seq_len=args.seq_len,
            token_positions=tuple(args.token_positions),
            pooling_modes=tuple(args.pooling_modes),
            device=device,
        )
        batch.source_metadata.update(source_metadata)
        bp = args.output_dir / f"{split_name}_activations.pt"
        batch.save(bp)
        wd = WindowDataset.from_windows(windows, source_metadata=source_metadata)
        wp = args.output_dir / f"{split_name}_windows.pt"
        wd.save(wp)
        summary["splits"][split_name] = {
            "series_count": len(series_ids),
            "series_id_sha256": hashlib.sha256("\n".join(series_ids).encode()).hexdigest(),
            "window_count": len(windows),
            "activation_count": len(batch),
            "path": str(bp),
            "window_dataset_path": str(wp),
        }

    with open(args.output_dir / "activation_dump_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    logger.info("Done. Summary: %s", args.output_dir / "activation_dump_summary.json")


if __name__ == "__main__":
    main()
