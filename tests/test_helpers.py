from __future__ import annotations

import torch

from toto_interp.types import WindowExample
from toto.model.toto import Toto


def make_tiny_toto() -> Toto:
    model = Toto(
        patch_size=4,
        stride=4,
        embed_dim=12,
        num_layers=3,
        num_heads=3,
        mlp_hidden_dim=24,
        dropout=0.0,
        spacewise_every_n_layers=3,
        scaler_cls="<class 'model.scaler.StdMeanScaler'>",
        output_distribution_classes=["<class 'model.distribution.StudentTOutput'>"],
        output_distribution_kwargs={},
        spacewise_first=False,
        use_memory_efficient_attention=False,
    )
    model.eval()
    return model


def make_window_example() -> WindowExample:
    context = torch.tensor(
        [
            [0.0, 0.5, 1.0, 1.5, 3.0, 3.5, 4.0, 4.5],
            [1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0],
        ],
        dtype=torch.float32,
    )
    next_patch = torch.tensor(
        [
            [5.0, 5.5, 6.0, 6.5],
            [3.0, 3.0, 3.0, 3.0],
        ],
        dtype=torch.float32,
    )
    return WindowExample(
        series_id="series-0",
        window_id="series-0:0",
        split="train",
        context=context,
        next_patch=next_patch,
        patch_size=4,
        freq="1min",
        item_id="item-0",
        num_target_variates=2,
        labels={
            "metric_type": "gauge",
            "domain": "Infrastructure",
            "frequency_bucket": "Short",
            "cardinality_bucket": "small_mv",
            "current_sparsity": 0.0,
            "future_sparsity": 0.0,
            "current_burstiness": 0.0,
            "future_burstiness": 0.0,
            "shift_risk": 0.0,
            "coordination": 1.0,
        },
    )
