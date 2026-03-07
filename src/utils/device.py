"""Device selection utility."""

from __future__ import annotations


def get_device(preference: str = "auto") -> str:
    """Return the best available compute device string.

    Parameters
    ----------
    preference:
        One of ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"mps"``.
        When ``"auto"`` the function selects CUDA > MPS > CPU.

    Returns
    -------
    str
        Device string suitable for ``torch.device()``.
    """
    if preference != "auto":
        return preference

    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass

    return "cpu"
