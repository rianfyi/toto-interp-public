from __future__ import annotations

import io

import torch
import pytest

from toto_interp import get_toto_weight_provenance, load_toto_with_fallback
from toto_interp.loader import resolve_device
from toto.model.toto import Toto

from .test_helpers import make_tiny_toto


def test_load_toto_with_fallback_disables_memory_efficient_attention(monkeypatch):
    sentinel = object()
    calls: list[dict[str, object]] = []

    def fake_from_pretrained(model_id, **kwargs):
        calls.append({"model_id": model_id, **kwargs})
        if len(calls) == 1:
            raise AssertionError("use_memory_efficient_attention kernel unavailable")
        return sentinel

    monkeypatch.setattr(Toto, "from_pretrained", fake_from_pretrained)

    loaded = load_toto_with_fallback("fake-model-id", map_location="cpu", strict=False)

    assert loaded is sentinel
    assert len(calls) == 2
    assert calls[1]["use_memory_efficient_attention"] is False


def test_resolve_device_respects_explicit_choice():
    assert resolve_device("cpu") == "cpu"


def test_load_toto_with_fallback_supports_random_init(monkeypatch):
    model = make_tiny_toto()
    original = next(model.model.parameters()).detach().clone()

    monkeypatch.setattr(Toto, "from_pretrained", lambda *args, **kwargs: model)

    loaded = load_toto_with_fallback("fake-model-id", weight_source="random_init", map_location="cpu", device="cpu")

    randomized = next(loaded.model.parameters()).detach()
    assert not torch.allclose(original, randomized)
    assert not loaded.training


def test_load_toto_with_fallback_supports_checkpoint_loading(monkeypatch, tmp_path):
    source = make_tiny_toto()
    target = make_tiny_toto()
    for param in source.model.parameters():
        torch.nn.init.constant_(param, 0.25)
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(source.state_dict(), checkpoint_path)

    monkeypatch.setattr(Toto, "from_pretrained", lambda *args, **kwargs: target)

    loaded = load_toto_with_fallback(
        "fake-model-id",
        weight_source="checkpoint",
        checkpoint_path=checkpoint_path,
        map_location="cpu",
        device="cpu",
    )

    source_param = next(source.parameters()).detach()
    loaded_param = next(loaded.parameters()).detach()
    assert torch.allclose(source_param, loaded_param)
    assert not loaded.training


def test_layer_permuted_pretrained_is_seeded_and_leaves_ordered_source_serializable(monkeypatch):
    ordered_model = make_tiny_toto()
    for layer_index, layer in enumerate(ordered_model.model.transformer.layers):
        for parameter in layer.parameters():
            torch.nn.init.constant_(parameter, float(layer_index + 1))
    ordered_model.train()
    state_before = {name: value.detach().clone() for name, value in ordered_model.state_dict().items()}

    monkeypatch.setattr(Toto, "from_pretrained", lambda *args, **kwargs: ordered_model)

    loaded = load_toto_with_fallback(
        "fake-model-id",
        weight_source="layer_permuted_pretrained",
        layer_permutation_seed=42,
        map_location="cpu",
        device="cpu",
    )

    provenance = get_toto_weight_provenance(loaded)
    assert loaded is not ordered_model
    assert ordered_model.training  # The shared ordered source was not put in eval mode.
    assert not loaded.training
    assert provenance is not None
    assert provenance["weight_source"] == "layer_permuted_pretrained"
    assert provenance["base_weight_source"] == "pretrained"
    assert provenance["permutation_seed"] == 42
    permutation = provenance["runtime_layer_to_pretrained_layer"]
    assert sorted(permutation) == list(range(len(ordered_model.model.transformer.layers)))
    assert permutation != list(range(len(ordered_model.model.transformer.layers)))

    # Each runtime block is an independent copy of exactly one original block.
    for runtime_index, source_index in enumerate(permutation):
        loaded_params = dict(loaded.model.transformer.layers[runtime_index].named_parameters())
        source_params = dict(ordered_model.model.transformer.layers[source_index].named_parameters())
        assert loaded_params.keys() == source_params.keys()
        for name, loaded_parameter in loaded_params.items():
            assert torch.equal(loaded_parameter, source_params[name])
            assert loaded_parameter.data_ptr() != source_params[name].data_ptr()

    # The original object still has its canonical order and an unchanged state_dict.
    assert list(ordered_model.model.transformer.layers) != list(loaded.model.transformer.layers)
    for name, value in ordered_model.state_dict().items():
        assert torch.equal(value, state_before[name])
    serialized = io.BytesIO()
    torch.save(ordered_model.state_dict(), serialized)
    assert serialized.getbuffer().nbytes > 0


def test_layer_permuted_pretrained_reuses_the_exact_seeded_order_without_mutating_provenance(monkeypatch):
    ordered_model = make_tiny_toto()
    monkeypatch.setattr(Toto, "from_pretrained", lambda *args, **kwargs: ordered_model)

    first = load_toto_with_fallback(
        "fake-model-id",
        weight_source="layer_permuted_pretrained",
        layer_permutation_seed=9,
        map_location="cpu",
        device="cpu",
    )
    second = load_toto_with_fallback(
        "fake-model-id",
        weight_source="layer_permuted_pretrained",
        layer_permutation_seed=9,
        map_location="cpu",
        device="cpu",
    )

    first_provenance = get_toto_weight_provenance(first)
    second_provenance = get_toto_weight_provenance(second)
    assert first_provenance == second_provenance
    assert first_provenance is not None
    first_provenance["runtime_layer_to_pretrained_layer"][0] = -1
    assert get_toto_weight_provenance(first)["runtime_layer_to_pretrained_layer"][0] != -1


def test_layer_permuted_pretrained_requires_an_explicit_seed(monkeypatch):
    monkeypatch.setattr(Toto, "from_pretrained", lambda *args, **kwargs: make_tiny_toto())

    with pytest.raises(ValueError, match="layer_permutation_seed"):
        load_toto_with_fallback(
            "fake-model-id",
            weight_source="layer_permuted_pretrained",
            map_location="cpu",
            device="cpu",
        )
