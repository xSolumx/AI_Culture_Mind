# Spin(8) Q8 joint-retraction fresh cohort: prospective protocol

Status: frozen before training seeds 1--9.

Date: 2026-08-03

Seed 0 selected the orbit-first retraction and is excluded from the
confirmatory reliability count. Seeds 1--9 are untouched fresh seeds.

## Fixed training and intervention

- family: `pure_spin8_positive`;
- four channels, eight real state values per channel;
- unconstrained 28D tangent parameters, matrix exponential action;
- unchanged parity-complete 2,000-step Q8 curriculum and optimizer;
- the exact seed-0 retraction code is applied without threshold, rank, or
  hyperparameter changes;
- every channel is retracted to the rank-four signed central-pair frame;
- decoder weights remain frozen during retraction.

## Required reporting

For every seed report raw and retracted:

- dense minimum member and joint accuracy through L256;
- full and orbit homomorphism RMS;
- per-channel orbit singular values;
- frame and action projection RMS;
- Spin(8) exponential reconstruction residual;
- streaming residuals;
- long central-pair accuracy at 4,095/4,096/16,383/16,384.

No seed may be omitted because it fails to train, has a degenerate frame, or
is damaged by retraction. Those are separate failure categories.

## Frozen gates

Per seed, the retracted model passes only if:

- action reconstruction maximum is `<= 1e-5`;
- full homomorphism RMS is `<= 1e-5`;
- every dense central-pair member and joint accuracy is at least `99%`;
- every long central-pair member and joint accuracy is at least `99%`;
- streaming state residual is `<= 1e-5` and logit residual `<= 1e-4`.

The cohort supports a reliability claim if at least 8/9 fresh seeds pass every
per-seed gate. A 9/9 result is required before describing the intervention as
uniformly reliable under this protocol. Raw-model performance is descriptive
and cannot substitute for the retracted gate.
