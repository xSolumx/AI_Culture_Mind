"""Compatibility entry point for the canonical pure PyTorch rotor SSM.

The maintained implementation lives in :mod:`pure_rotor_ssm.torch_backend`.
This filename remains import-stable for the existing tests, scripts, and
historical reports; it deliberately contains no second model implementation.
"""

from pure_rotor_ssm.torch_backend import *
