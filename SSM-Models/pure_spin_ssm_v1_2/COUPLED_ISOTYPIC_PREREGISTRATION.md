# Coupled isotypic recurrence: algebra and frozen promotion gates

**Frozen before training:** 2026-08-22

## Why this is the next recurrence

The earlier readout-only multiplicity router failed its frozen Shakespeare
gate. Moving multiplicity transport inside the recurrence is mathematically
stronger, but it also exposes a constraint that must not be hidden: arbitrary
multiplicity mixing commutes with Spin only when the equivalent copies carry
the **same** Spin action. The established v1.2 recurrence gives its two channels
separate controllers, so adding a channel rotation to that recurrence and
calling it Schur-legal would be false.

The candidate therefore uses a shared ordered Spin action on each of the three
inequivalent triality sectors and an independent action on the two-dimensional
multiplicity factor. For state

\[
H_t\in \mathbb R^2\otimes(8_v\oplus8_+\oplus8_-),
\]

one step is

\[
H_t=L_t H_{t-1}R_t^T+D_t,
\qquad
L_t=\operatorname{diag}(s_t)Q_t,
\qquad Q_t\in SO(2).
\]

The same learned angle builds `Q_t` in all three sectors. The ordered Spin
factors build `R_t=(R_{v,t},R_{+,t},R_{-,t})` from one shared coordinate vector.
No dense `24 x 24` transition is materialized in the CUDA path.

The transition family is closed under chronological composition:

\[
(L_2,R_2,D_2)\circ(L_1,R_1,D_1)
=
(L_2L_1,\ R_2R_1,\ D_2+L_2D_1R_2^T).
\]

This is an associative two-sided affine monoid and therefore admits a prefix
compiler. Orthogonality of `Q_t` and each `R_t` gives

\[
\|L_t\otimes R_t\|_2=\|L_t\|_2
\leq \max_c s_{t,c}<1,
\]

so coupling does not remove the existing contractive stability bound. At zero
multiplicity angle it reduces exactly to independent multiplicity copies with a
shared Spin action. It does **not** reduce to the established v1.2 model, whose
copies have separately learned Spin actions; that architectural change is why
the gates below are staged.

## Implemented compiler layers

- `coupled_isotypic_scan.py` is the transparent PyTorch oracle and logarithmic-
  depth associative prefix scan.
- `raw_cuda.py` exposes a full-training autograd operation.
- `csrc/spin_scan_cuda.cu` assigns one warp to both multiplicity copies and all
  three triality sectors. It reuses one coordinate stream, applies the shared
  Spin factors, and then performs the `2 x 2` left action.
- Backward replays the pre-affine state rather than inverting `L_t`. This keeps
  reverse mode defined for zero retention and singular left actions.

Before this freeze, exact-shape tests established transition associativity,
recurrent/prefix output and gradient parity, zero-angle reduction, contraction,
causality, and full-model parity. CUDA tests on the RTX 2070 SUPER established
output and coordinate/left/drive/initial gradient parity for 3, 6, 15, and 28
Spin factors. The complete v1.2 suite passed 45 tests.

## Stage A: does recurrent multiplicity transport help?

This isolates `Q_t` while holding the shared-action architecture fixed.

- baseline: `coupled_isotypic`, `recurrent_multiplicity=identity`;
- candidate: `coupled_isotypic`, `recurrent_multiplicity=orthogonal`;
- backend: `raw_cuda_coupled` for both;
- seeds: `109`, `113`, `127`;
- 300 steps, batch 8, length 256, 16 fixed validation batches;
- Tiny Shakespeare raw UTF-8 bytes with the maintained 90/5/5 contiguous split;
- group schedule `(3,4,6,8)`, direction readout, no readout router;
- paired initialization, batches, validation windows, and cu126 WSL runtime.

The identity model has 619,808 parameters. The orthogonal controller adds 516
parameters, giving 620,324. The fused Mamba-2 reference is 623,740 parameters.

Positive improvement is `shared-identity bpb - shared-orthogonal bpb`. Stage A
passes only if all hold:

1. the candidate wins at least two of three seeds;
2. mean improvement is at least `0.0100` bpb;
3. no seed regresses by more than `0.0500` bpb;
4. all values and gradients are finite and artifact hashes are complete.

## Stage B: does the candidate beat established v1.2?

Stage B may be run only if Stage A passes. It compares the promoted coupled
candidate with the established independent-action `raw_cuda_hybrid` v1.2 using
fresh seeds `131`, `137`, and `139`; all other data and optimization settings
remain identical. The coupled candidate has 6,192 fewer parameters than the
626,516-parameter established model.

Positive improvement is `independent v1.2 bpb - shared-orthogonal bpb`. The
same four acceptance thresholds apply. If Stage B passes, a separate
order-balanced steady-step campaign must show no more than a 10% throughput
regression before changing the maintained default. Sequential timers embedded
in the quality artifacts are diagnostic only.

Failure at Stage A rejects learned recurrent multiplicity transport, not the
shared-action compiler. Failure at Stage B rejects replacement of the current
v1.2 default, not the exact algebra or CUDA implementation. Either negative
result is retained as evidence rather than relabeled as a breakthrough.
