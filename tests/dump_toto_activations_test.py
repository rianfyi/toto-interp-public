from __future__ import annotations

import random

import numpy as np
import torch

from scripts import dump_toto_activations


def _random_draws() -> tuple[float, float, torch.Tensor]:
    return random.random(), float(np.random.random()), torch.rand(3)


def test_activation_dump_seeding_is_reproducible_without_model_loading():
    dump_toto_activations.seed_process_randomness(123)
    first = _random_draws()
    dump_toto_activations.seed_process_randomness(123)
    second = _random_draws()

    assert first[:2] == second[:2]
    assert torch.equal(first[2], second[2])
