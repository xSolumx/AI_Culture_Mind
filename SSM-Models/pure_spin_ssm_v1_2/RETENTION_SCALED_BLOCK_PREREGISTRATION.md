# Retention-scaled block coupling gate

**Frozen before training:** 2026-08-22

## Mechanism

The free-angle independent-action block recurrence was algebraically valid but
failed its prospective Shakespeare gate. It applied an `SO(2)` rotation at
every token without using the recurrence's learned time scale.

For continuous damping,

\[
s_{t,c}=e^{-\lambda_c\Delta t}.
\]

The geometric mean retention is

\[
\bar s_t=\sqrt{s_{t,1}s_{t,2}}
=e^{-\bar\lambda\Delta t},
\]

so

\[
1-\bar s_t=\bar\lambda\Delta t+O(\Delta t^2).
\]

The candidate therefore uses

\[
\theta_t=(1-\bar s_t)\frac\pi2\tanh(w^Tx_t+b),
\qquad
Q_t=e^{\theta_tJ},
\]

inside the already certified independent-action block transition. The unknown
mean decay rate is absorbed into the learned controller. This factor is
bounded in `[0,1)`, becomes exactly zero at identity retention, and introduces
no parameter or cache increase over the failed free-angle candidate.

This rule was derived from the time-discretization mismatch after the complete
free-angle gate closed. Its formula, angle cap, and acceptance thresholds are
frozen here before any successor seed is trained; no previous result is
relabelled.

The transition remains

\[
A_{t,r}=(\operatorname{diag}(s_t)Q_t\otimes I_8)
\operatorname{diag}(R_{t,1,r},R_{t,2,r}),
\]

so equivariance, 16-dimensional block-affine closure, and the exact norm bound
`||A_t||_2=max(s_t)` are unchanged. The semantic and raw-CUDA paths differ only
in the precomputed angle. Focused tests establish the exponential-step formula
and full-model CUDA output/gradient parity.

## Frozen protocol

- baseline: maintained independent recurrence with `raw_cuda_hybrid`;
- candidate: `independent_block`, `orthogonal` recurrent multiplicity,
  `retention_step` coupling scale, and `raw_cuda_block`;
- fresh seeds: `211`, `223`, and `227`;
- Tiny Shakespeare raw UTF-8 bytes, maintained 90/5/5 split;
- 300 steps, batch 8, sequence length 256, 16 fixed validation batches;
- group schedule `(3,4,6,8)`, direction readout, SwiGLU, no readout router;
- paired common initialization, batches, validation windows, optimizer, and
  WSL PyTorch 2.10/cu126 runtime.

The candidate has 627,032 parameters versus 626,516 for maintained v1.2. The
516-parameter increase is 0.0824%; scaling itself adds no parameters.

Positive improvement is `maintained bpb - retention-scaled bpb`. Authorize a
speed gate only if all hold:

1. candidate wins at least two of three seeds;
2. mean improvement is at least `+0.0100` bpb;
3. no seed regresses by more than `0.0500` bpb;
4. values and gradients remain finite and implementation hashes agree.

Sequential timers remain diagnostic. A quality pass still requires a separate
order-balanced complete-step campaign with no more than 10% throughput
regression before any default change. Failure leaves maintained v1.2 unchanged
and closes this continuous-time `SO(2)` route at the tested scale.
