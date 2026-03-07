"""Global seed utilities for reproducibility."""

from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Fix all random seeds for reproducibility.

    Covers:
    - Python's built-in ``random`` module
    - NumPy
    - PyTorch (if available)
    - CUDA (if available)
    - Dataloader worker seeds (via ``worker_init_fn``)

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic at the cost of some speed
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass


def worker_init_fn(worker_id: int) -> None:
    """Seed individual DataLoader workers.

    Pass this as ``worker_init_fn`` to ``torch.utils.data.DataLoader`` to
    ensure each worker uses a different but deterministic seed.

    Parameters
    ----------
    worker_id:
        Worker index provided automatically by PyTorch.
    """
    worker_seed = int(np.random.get_state()[1][0]) + worker_id
    np.random.seed(worker_seed % (2**32))
    random.seed(worker_seed)
