"""Maintained Pure Spin(8) SSM package.

Import :mod:`pure_spin8_ssm.torch_backend` explicitly so the package metadata
remains usable without importing PyTorch.
"""

SPIN8_DIM = 8
SPIN8_BIVECTOR_DIM = 28
TRIALITY_STATE_DIM = 24
__version__ = "1.1.0"

__all__ = [
    "SPIN8_BIVECTOR_DIM",
    "SPIN8_DIM",
    "TRIALITY_STATE_DIM",
    "__version__",
]
