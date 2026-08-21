"""Exceptional Albert/F4/E6 delta state-space model research package."""

from .action import ExceptionalAction, exponential_action, ordered_exponential_action
from .albert import (
    ALBERT_DIM,
    E6_DIM,
    F4_DIM,
    AlbertAlgebra,
    build_albert_algebra,
)
from .model import ExceptionalDeltaConfig, ExceptionalDeltaLM, ExceptionalDeltaState
from .scan import (
    TwoSidedAffineTransition,
    compile_delta_transition,
    parallel_delta_scan,
    recurrent_delta_scan,
)

__all__ = [
    "ALBERT_DIM",
    "E6_DIM",
    "F4_DIM",
    "AlbertAlgebra",
    "ExceptionalAction",
    "ExceptionalDeltaConfig",
    "ExceptionalDeltaLM",
    "ExceptionalDeltaState",
    "TwoSidedAffineTransition",
    "build_albert_algebra",
    "compile_delta_transition",
    "exponential_action",
    "ordered_exponential_action",
    "parallel_delta_scan",
    "recurrent_delta_scan",
]
