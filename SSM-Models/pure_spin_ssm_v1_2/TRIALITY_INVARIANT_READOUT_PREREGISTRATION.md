# Triality-invariant readout promotion gate

**Frozen before training:** 2026-08-22  
**Status:** prospective internal v1.2 architecture gate

## Hypothesis selected from the research chain

The current readout applies one RMS normalization to the flattened
`channels x (8v + 8+ + 8-)` state. This retains direction and relative sector
content but makes the output approximately insensitive to a common positive
rescaling. v1.3 independently identified this as an accidental restriction of
its Albert memory and restored scale using invariant scalar channels.

For each v1.2 channel, write the three triality sectors as

\[
v\in 8_v,\qquad p\in 8_+,\qquad m\in 8_-.
\]

The candidate retains the existing normalized 48-dimensional direction and
appends

\[
e_v=\log(1+\|v\|^2/8),\quad
e_+=\log(1+\|p\|^2/8),\quad
e_-=\log(1+\|m\|^2/8),
\]

together with the bounded cubic triality invariant

\[
\widehat\tau=\frac{\rho(v,m,p)}{\sqrt{1+\rho(v,m,p)^2}}.
\]

Thus two channels add eight scalars, and only the existing output projection
changes width from 48 to 56. There is no appended linear scalar: the three
nontrivial irreducible Spin(8) modules have no distinguished invariant linear
direction. The implementation test verifies that all appended scalars are
unchanged under the shared numerical Spin(8) action, while a positive state
rescaling is no longer invisible.

This is the narrow transfer justified by the repository evidence. The v1.3
exceptional action itself is not imported: dense F4/E6 transport failed its
fresh-seed generic-language gate. Delta memory is not imported because it
would change the recurrence rather than isolate the readout. Algebraically
equivalent contraction shortcuts are also excluded because v1.3 showed that a
changed float32 evaluation order can change training quality.

## Frozen candidates

- baseline: `readout=direction`;
- candidate: `readout=triality_invariants`;
- otherwise identical Pure Spin v1.2 with `raw_cuda_hybrid`, subgroup schedule
  `Spin(3) -> Spin(4) -> Spin(6) -> Spin(8)`, SwiGLU, four layers, width 128,
  and two triality channels.

Parameter counts are 626,516 for baseline and 630,612 for candidate, an
increase of 4,096 or 0.654%. The established fused Mamba-2 reference has
623,740 parameters; candidate/Mamba mismatch is 1.090%, within the maintained
5% comparison ceiling.

## Frozen data and training protocol

- pinned disjoint Tiny Shakespeare byte splits already recorded by SHA-256;
- fresh seeds `71`, `73`, and `79`;
- 300 AdamW updates, learning rate `3e-3`, weight decay `0.01`;
- batch size 8, sequence length 256, 16 fixed validation batches;
- gradient clipping at 1.0;
- Torch 2.10 cu126 under the maintained WSL environment;
- both readouts receive identical initialization seeds, training batches, and
  validation windows within a seed.

The paired runner records source hashes, exact parameters, environment, data,
losses, memory, and sequential timing. Sequential timing is diagnostic only;
any speed statement requires a later order-balanced benchmark.

## Decision rule

Let improvement be `direction bpb - triality-invariants bpb`, so positive is
better. Promote the candidate as the maintained v1.2 readout only if all hold:

1. it wins at least two of three seeds;
2. mean paired improvement is at least `0.0100` bpb;
3. no seed regresses by more than `0.0500` bpb;
4. all runs and gradients remain finite and provenance compatibility passes.

If the gate fails, `direction` remains the default and the candidate is kept
only as a documented research control. If it passes, the next gate is a fresh
three-seed matched comparison with fused Mamba-2 plus an order-balanced systems
check. No result changes the exact Spin(8) recurrence theorem or implies a
general language-model advantage.
