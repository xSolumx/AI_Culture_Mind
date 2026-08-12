"""Compatibility entry point for the canonical GA-SSM experiment.

The implementation moved to :mod:`ga_ssm` so it can be imported and tested.
"""

from ga_ssm import (
    GATransformerLM,
    GASSMLanguageModel,
    GASSMBlock,
    GeometricAttention,
    GeometricFFN,
    GeometricTransformerBlock,
    PositionalRotors,
    RotorResidual,
    SelectiveRotorSSM,
    TrainingConfig,
    create_train_state,
    data_generator,
    eval_step,
    evaluate_model,
    main,
    sample_text,
    initialize_recurrent_states,
    rotor_affine_scan,
    rotor_recurrent_scan,
    rotor_transition_step,
    train,
    train_step,
)

__all__ = [
    "GATransformerLM",
    "GASSMLanguageModel",
    "GASSMBlock",
    "GeometricAttention",
    "GeometricFFN",
    "GeometricTransformerBlock",
    "PositionalRotors",
    "RotorResidual",
    "SelectiveRotorSSM",
    "TrainingConfig",
    "create_train_state",
    "data_generator",
    "eval_step",
    "evaluate_model",
    "initialize_recurrent_states",
    "rotor_affine_scan",
    "rotor_recurrent_scan",
    "rotor_transition_step",
    "sample_text",
    "train",
    "train_step",
]


if __name__ == "__main__":
    main()
