# Spin-Motor Local Identification Protocol

Protocol frozen: **2026-08-16T20:12:37+02:00**
Parent gradient-trained direct-readout artifact:
`experiments/artifacts/spin_motor_direct_readout_pilot300.json`
Parent SHA-256:
`7a364b61ba51666db65f0ced909fc78d81855582fd14e9dd5e598d2d4d3ab1f2`

## Motivation

The randomly initialized, gradient-trained direct motor did not fit the frozen
300-step task. The matched direct-product state did retain the center sign on
all long splits but failed physical pose accuracy. This distinguishes
optimization failure from the already-tested exact realizability of the motor
state.

Every training prefix has a signed pose target. For right-composed group states
the token transition is observable locally:

`Delta M_t = inverse(M_{t-1}) M_t`.

In quaternion/translation coordinates:

- `q_delta = conjugate(q_prev) q_t`;
- `t_delta = R(q_prev)^T (t_t - t_prev)`.

The proposed estimator groups these local increments by token, averages the
repeated observations, installs the seven identified motors into the 49-scalar
direct motor scan, and performs no gradient optimization.

## Leakage boundary

The estimator may read only the already-frozen legal training inputs and their
prefix pose targets. It must not read:

- evaluation inputs or targets;
- any occurrence of `a^2`, `b^3`, or `(ab)^5` in training (the parent split
  audit already requires zero); or
- the exact generator table when fitting parameters.

The exact token motors are used only after fitting to measure identification
error. They are not copied into the model.

## Frozen execution

- regenerate the same coordinate `e`, seed `0`, 300-step/batch-16/length-16
  legal training schedule;
- require all parent split audits to pass;
- identify all seven token motors from all `76,800` supervised prefix
  transitions;
- evaluate the identified 8-scalar state on the same 18 paired splits at
  lengths 16, 64, and 128;
- use the same 15-degree rotation and 0.10 translation thresholds;
- save and hash one 49-parameter checkpoint and one JSON artifact.

## Preregistered gates

The identified model must meet all of the following:

1. every token has at least one legal training observation;
2. maximum fitted-vs-exact token quaternion error is below `0.1 degrees`;
3. maximum fitted-vs-exact token translation error is below `1e-6`;
4. every `L64` and `L128` split has signed-hemisphere accuracy at least 90%;
5. every `L64` and `L128` split has joint signed-pose accuracy at least 80%;
6. every split has paired double-cover pose accuracy at least 80%.

Expected result: all empirical accuracies are 100% up to floating-point scan
tolerance. A failure of the first three gates invalidates downstream model
interpretation. A success proves realizability and leak-free finite-data
identifiability for this deterministic token task, not superiority under
end-to-end learning and not a general theorem for continuous observations.
