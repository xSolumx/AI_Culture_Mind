"""Pure selective rotor SSM backends.

Import :mod:`pure_rotor_ssm.jax_backend` or
:mod:`pure_rotor_ssm.torch_backend` explicitly. Keeping this package initializer
backend-free preserves optional JAX/PyTorch installation boundaries.
"""

GA_DIM = 8
INVARIANT_FEATURES = 5
__version__ = "2.1.0"

__all__ = ["GA_DIM", "INVARIANT_FEATURES", "__version__"]
