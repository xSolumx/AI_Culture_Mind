# Spin-Delta oracle-address intervention

**Frozen:** 2026-08-22 before outcome training on seeds 431/433/439.

## Hypothesis

The learned-address capability gate found one 97%-accurate Spin-Delta seed but
failed robust capability and lost to maintained v1.2 on average. Programme 03
shows that delta overwrite is exact under orthogonal keys and that learned
continuous address interference can dominate the update law. The next causal
question is whether supplying the correct two-slot address removes Spin-Delta's
seed instability.

## Intervention

Both rows are identically initialized Spin-Delta models with the same two-slot
state, independent Spin transports, raw-CUDA recurrence, optimizer, data, and
parameter count. The oracle row alone receives a causal integer tensor:

- write events are enabled only at value-token positions;
- write and erase use the semantic key's exact one-hot slot;
- erase strength is one at those write events and zero elsewhere;
- the exact query slot is supplied only at the final query-key token.

Future values and the target are never supplied. This tests address inference
plus event timing; it is not autonomous learned retrieval and cannot be called
a language-model improvement.

## Frozen task and budget

- same two-key, 32-value repeated-overwrite task;
- train at 8 writes; evaluate at 8, 16, and 32 writes;
- `d_model=64`, two Spin(8) layers, two heads and two slots;
- 800 AdamW steps, batch 128, 32 evaluation batches per length;
- learning rate `3e-3`, weight decay `0.01`, clip norm `1.0`;
- seeds 431, 433, and 439;
- fixed order: learned addresses, then oracle addresses;
- every model parameter must be bitwise equal before the intervention.

## Frozen decisions

Oracle capacity passes only if every oracle seed reaches at least 95% accuracy
at both 8 and 16 writes.

Address-inference bottleneck passes only if oracle addressing additionally
improves 16-write accuracy by at least five percentage points in at least two
seeds and by at least five points on the three-seed mean.

The 32-write row is a stress frontier. Oracle success identifies address/event
inference as the bottleneck under this task. Oracle failure instead locates the
problem downstream—in representation, drive formation, recurrence optimization,
or readout—even when addresses are exact.
