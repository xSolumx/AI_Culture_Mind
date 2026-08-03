"""Overhauled recurrent SpinorModel, isolated from the historical prototype."""

from .algebra import (
    GA_DIM,
    RotorAffineTransition,
    associative_scan,
    compose_transitions,
    geometric_product,
    reversion,
    rotor_from_bivector,
    rotor_sandwich,
)
from .model import SpinorSSMConfig, SpinorSSMLanguageModel

__all__ = [
    "GA_DIM",
    "RotorAffineTransition",
    "SpinorSSMConfig",
    "SpinorSSMLanguageModel",
    "associative_scan",
    "compose_transitions",
    "geometric_product",
    "reversion",
    "rotor_from_bivector",
    "rotor_sandwich",
]
