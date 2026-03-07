"""Tests for the reproducibility seed utilities."""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.utils.seed import set_seed, worker_init_fn


class TestSetSeed:
    def test_python_random_reproducible(self):
        set_seed(42)
        a = [random.random() for _ in range(10)]
        set_seed(42)
        b = [random.random() for _ in range(10)]
        assert a == b

    def test_numpy_random_reproducible(self):
        set_seed(42)
        a = np.random.rand(10).tolist()
        set_seed(42)
        b = np.random.rand(10).tolist()
        assert a == b

    def test_different_seeds_differ(self):
        set_seed(1)
        a = random.random()
        set_seed(2)
        b = random.random()
        assert a != b

    def test_torch_seed_if_available(self):
        pytest.importorskip("torch")
        import torch
        set_seed(42)
        a = torch.rand(5).tolist()
        set_seed(42)
        b = torch.rand(5).tolist()
        assert a == b


class TestWorkerInitFn:
    def test_callable(self):
        # Should not raise regardless of worker_id
        worker_init_fn(0)
        worker_init_fn(3)
