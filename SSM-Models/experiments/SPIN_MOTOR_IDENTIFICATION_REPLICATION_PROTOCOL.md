# Spin-Motor Identification Replication Protocol

Protocol frozen: **2026-08-16T20:14:37+02:00**
Successful screening artifact:
`experiments/artifacts/spin_motor_identified_e_seed0.json`
Screening SHA-256:
`df132600d8be86505a4e5156b161a7e2ee33ce84dc4dba1dae26d3207653c62b`

## Replication matrix

Run local prefix-difference identification for the Cartesian product

- generator coordinates: `e`, `a`, `b`;
- legal schedule seeds: `0`, `1`, `2`.

This produces nine independently regenerated training/evaluation schedules.
Coordinate conjugation changes the exact rotation generators. Seed changes the
legal words and evaluation contexts.

## Frozen contract

Every run keeps the screening task and thresholds unchanged:

- 300 schedule batches, batch size 16, train length 16;
- zero training occurrences of `a^2`, `b^3`, `(ab)^5`;
- all 120 signed rotation states and all three translation tokens observed;
- 18 paired evaluation splits through length 128;
- 49 identified token parameters and an 8-scalar motor state;
- no gradient optimization and no evaluation data used for fitting.

Each run must pass all six gates from
`SPIN_MOTOR_IDENTIFICATION_PROTOCOL.md`. The aggregate passes only if all nine
runs pass. The compact replication artifact must retain every training schedule
hash, every evaluation schedule hash, every checkpoint hash, token
identification errors, minimum accuracies, and maximum numerical errors.

## Interpretation boundary

Success upgrades the finding from a one-coordinate screening result to a
replicated finite deterministic identification result. It still does not test:

- noisy pose supervision;
- continuous observations requiring a learned motor encoder;
- final-only supervision;
- natural data or language modeling;
- fused systems performance; or
- superiority to Mamba-2/DeltaProduct under a matched training algorithm.

Any one failed run is reported as a failed replication; no averaging away a
failure is permitted.
