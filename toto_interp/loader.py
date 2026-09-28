from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import torch

from toto.model.toto import Toto


LAYER_PERMUTED_PRETRAINED = "layer_permuted_pretrained"


def _nonidentity_layer_permutation(*, layer_count: int, seed: int) -> tuple[list[int], int]:
    """Sample a seeded, uniformly non-identity permutation of transformer blocks.

    Rejection sampling from ``randperm`` is uniform over the non-identity
    permutations.  Excluding the identity keeps every layer-permuted control a
    genuine order intervention rather than (rarely) reproducing the ordered
    pretrained model.
    """

    if layer_count < 2:
        raise ValueError("layer_permuted_pretrained requires at least two transformer layers.")

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    identity = list(range(layer_count))
    attempts = 0
    while True:
        attempts += 1
        permutation = torch.randperm(layer_count, generator=generator).tolist()
        if permutation != identity:
            return permutation, attempts


def _layer_permuted_pretrained_copy(model: Toto, *, seed: int) -> Toto:
    """Return an independent pretrained-model copy with reordered blocks.

    ``from_pretrained`` callers can retain or serialize their original ordered
    model: no modules or parameter storages on that object are mutated.  Toto
    modules cannot be ``deepcopy``-ed because their rotary-embedding objects
    hold static methods, so rebuild a fresh Toto from the Hub config and load
    the complete pretrained state before reordering its cloned blocks.
    """

    try:
        ordered_layers = model.model.transformer.layers
    except AttributeError as exc:
        raise AttributeError("Toto model does not expose model.transformer.layers for permutation.") from exc
    if not isinstance(ordered_layers, torch.nn.ModuleList):
        raise TypeError(
            "Expected Toto model.transformer.layers to be torch.nn.ModuleList, "
            f"got {type(ordered_layers)!r}."
        )

    permutation, attempts = _nonidentity_layer_permutation(layer_count=len(ordered_layers), seed=seed)
    model_config = getattr(model, "_hub_mixin_config", None)
    if not isinstance(model_config, dict):
        raise TypeError(
            "Toto model is missing the Hub constructor config required to create an independent "
            "layer-permuted control."
        )
    controlled_model = type(model)(**copy.deepcopy(model_config))
    missing_keys, unexpected_keys = controlled_model.load_state_dict(model.state_dict(), strict=True)
    if missing_keys or unexpected_keys:  # Defensive: strict=True should already reject this.
        raise RuntimeError(
            "Fresh Toto clone did not exactly load the ordered pretrained state: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )
    copied_layers = controlled_model.model.transformer.layers
    controlled_model.model.transformer.layers = torch.nn.ModuleList(
        [copied_layers[source_layer_index] for source_layer_index in permutation]
    )
    # This is deliberately plain metadata, not a registered Module/buffer, so
    # it cannot alter the model's state_dict serialization contract.
    controlled_model._toto_interp_weight_provenance = {
        "schema_version": 1,
        "weight_source": LAYER_PERMUTED_PRETRAINED,
        "base_weight_source": "pretrained",
        "control": "transformer_block_order_permutation",
        "permutation_algorithm": "torch.randperm_rejection_nonidentity_v1",
        "permutation_seed": seed,
        "permutation_attempts": attempts,
        "torch_version": torch.__version__,
        "layer_count": len(ordered_layers),
        # Runtime layer i executes the block originally trained at this index.
        "runtime_layer_to_pretrained_layer": permutation,
    }
    return controlled_model


def get_toto_weight_provenance(model: Toto) -> dict[str, Any] | None:
    """Return a defensive copy of control provenance attached by this loader."""

    provenance = getattr(model, "_toto_interp_weight_provenance", None)
    return None if provenance is None else copy.deepcopy(provenance)


def resolve_device(device: str | None = None) -> str:
    if device is None or device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return device


def _reset_module_parameters(module: torch.nn.Module) -> None:
    reset_fn = getattr(module, "reset_parameters", None)
    if callable(reset_fn):
        reset_fn()
        return

    for name, param in module.named_parameters(recurse=False):
        if param.ndim >= 2:
            torch.nn.init.xavier_uniform_(param)
        elif "bias" in name:
            torch.nn.init.zeros_(param)
        else:
            torch.nn.init.normal_(param, mean=0.0, std=0.02)


def _iter_resettable_modules(module: torch.nn.Module):
    for child in module.modules():
        has_direct_params = any(True for _ in child.named_parameters(recurse=False))
        if has_direct_params or callable(getattr(child, "reset_parameters", None)):
            yield child


def _reinitialize_model(
    model: Toto,
    *,
    randomize_scope: str = "full",
    randomize_layers: tuple[int, ...] | None = None,
) -> None:
    backbone = model.model
    if randomize_scope == "full":
        for module in _iter_resettable_modules(backbone):
            _reset_module_parameters(module)
        return

    if randomize_scope == "selected_layers":
        if not randomize_layers:
            raise ValueError("randomize_layers is required when randomize_scope='selected_layers'.")
        for layer_idx in randomize_layers:
            if layer_idx < 0 or layer_idx >= len(backbone.transformer.layers):
                raise ValueError(f"Layer index {layer_idx} is out of range for Toto.")
            for module in _iter_resettable_modules(backbone.transformer.layers[layer_idx]):
                _reset_module_parameters(module)
        return

    if randomize_scope == "head_only":
        matched_modules = [
            child
            for name, child in backbone.named_children()
            if any(token in name.lower() for token in ("head", "output", "decoder"))
        ]
        if not matched_modules:
            raise ValueError("Could not find an output/head module to randomize for head_only scope.")
        for module in matched_modules:
            for child in _iter_resettable_modules(module):
                _reset_module_parameters(child)
        return

    raise ValueError(f"Unsupported randomize_scope: {randomize_scope}")


def _load_checkpoint_state(model: Toto, checkpoint_path: str | Path, *, strict: bool = False) -> None:
    payload = torch.load(Path(checkpoint_path), map_location="cpu", weights_only=False)
    if isinstance(payload, Toto):
        state_dict = payload.state_dict()
    elif isinstance(payload, dict):
        if "state_dict" in payload and isinstance(payload["state_dict"], dict):
            state_dict = payload["state_dict"]
        elif "model_state_dict" in payload and isinstance(payload["model_state_dict"], dict):
            state_dict = payload["model_state_dict"]
        else:
            state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if strict and (missing_keys or unexpected_keys):
        raise RuntimeError(
            "Checkpoint load failed with strict=True: "
            f"missing_keys={missing_keys}, unexpected_keys={unexpected_keys}"
        )


def load_toto_with_fallback(
    model_id: str = "Datadog/Toto-Open-Base-1.0",
    *,
    map_location: str = "cpu",
    device: str | None = None,
    strict: bool = False,
    weight_source: str = "pretrained",
    checkpoint_path: str | Path | None = None,
    randomize_scope: str = "full",
    randomize_layers: tuple[int, ...] | None = None,
    layer_permutation_seed: int | None = None,
    **model_overrides: Any,
) -> Toto:
    """
    Load Toto with an explicit no-xFormers fallback.

    Toto's checkpoint loader already disables memory-efficient attention when
    xFormers is unavailable, but this helper keeps the fallback behavior
    centralized for the interpretability scripts and tests.
    """

    resolved_device = resolve_device(device)
    load_location = map_location
    if resolved_device != "cpu" and map_location == "cpu":
        load_location = "cpu"

    try:
        model = Toto.from_pretrained(model_id, map_location=load_location, strict=strict, **model_overrides)
    except AssertionError as exc:
        if "use_memory_efficient_attention" not in str(exc):
            raise
        safe_overrides = dict(model_overrides)
        safe_overrides["use_memory_efficient_attention"] = False
        model = Toto.from_pretrained(model_id, map_location=load_location, strict=strict, **safe_overrides)

    if weight_source == "random_init":
        _reinitialize_model(
            model,
            randomize_scope=randomize_scope,
            randomize_layers=randomize_layers,
        )
    elif weight_source == LAYER_PERMUTED_PRETRAINED:
        if layer_permutation_seed is None:
            raise ValueError(
                "layer_permutation_seed is required when "
                "weight_source='layer_permuted_pretrained'."
            )
        model = _layer_permuted_pretrained_copy(model, seed=layer_permutation_seed)
    elif weight_source == "checkpoint":
        if checkpoint_path is None:
            raise ValueError("checkpoint_path is required when weight_source='checkpoint'.")
        _load_checkpoint_state(model, checkpoint_path, strict=strict)
    elif weight_source != "pretrained":
        raise ValueError(f"Unsupported weight_source: {weight_source}")

    if resolved_device != "cpu" and hasattr(model, "to"):
        model.to(resolved_device)
    if hasattr(model, "eval"):
        model.eval()
    return model
