from __future__ import annotations

from toto_interp import TraceConfig, extract_activations
from toto_interp.labels import RAW_FEATURE_NAMES

from .test_helpers import make_tiny_toto, make_window_example


def test_extract_activations_matches_shape_with_and_without_kv_cache():
    model = make_tiny_toto()
    window = make_window_example()

    with_cache = extract_activations(
        model,
        [window],
        TraceConfig(
            layers=(0, 1, 2),
            token_positions=("all_context", "final_context", "first_decode"),
            pooling_modes=("per_variate", "series_mean"),
            capture_patch_embedding=True,
            use_kv_cache=True,
        ),
    )
    without_cache = extract_activations(
        model,
        [window],
        TraceConfig(
            layers=(0, 1, 2),
            token_positions=("all_context", "final_context", "first_decode"),
            pooling_modes=("per_variate", "series_mean"),
            capture_patch_embedding=True,
            use_kv_cache=False,
        ),
    )

    assert len(with_cache) == len(without_cache) == 48
    assert with_cache.activations.shape == without_cache.activations.shape
    assert with_cache.raw_features.shape == without_cache.raw_features.shape
    assert set(with_cache.layer_indices.tolist()) == {-1, 0, 1, 2}
    assert with_cache.raw_feature_names == RAW_FEATURE_NAMES
    assert max(with_cache.patch_indices.tolist()) == 2


def test_context_only_trace_does_not_construct_output_distribution(monkeypatch):
    model = make_tiny_toto()
    window = make_window_example()

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("The unused distribution head was called.")

    monkeypatch.setattr(model.model.output_distribution, "forward", fail_if_called)
    batch = extract_activations(
        model,
        [window],
        TraceConfig(
            layers=(0, 1, 2),
            token_positions=("all_context", "final_context"),
            pooling_modes=("series_mean",),
            capture_patch_embedding=True,
            use_kv_cache=True,
        ),
    )

    assert len(batch) == 12
