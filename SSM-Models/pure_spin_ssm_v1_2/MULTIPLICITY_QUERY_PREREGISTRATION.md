# Schur-legal multiplicity-query promotion gate

**Frozen before training:** 2026-08-22

## Mechanism

Pure Spin v1.2 carries two equivalent copies of each triality representation.
For a shared Spin(8) action on `V tensor R^2`, Programme 1's isotypic
decomposition permits an arbitrary operator on the multiplicity factor. The
candidate uses the smallest bounded query-dependent instance:

\[
Q_t=M(\theta_t)\otimes I_V,
\qquad
M(\theta)=
\begin{pmatrix}
\cos\theta&-\sin\theta\\
\sin\theta&\cos\theta
\end{pmatrix},
\qquad
\theta_t=\frac\pi2\tanh(w^Tx_t+b).
\]

Because

\[
(M_t\otimes I)(I\otimes\rho(g))
=(I\otimes\rho(g))(M_t\otimes I),
\]

the router is exactly compatible with the shared vector/positive/negative
Spin(8) action. The same `M_t` is used for all three sectors. It acts only on
the already-computed readout state, so the bounded recurrence, cache, and raw
CUDA scan are unchanged.

The controller is zero-initialized, making the candidate exactly the identity
router before learning. Tests verify bitwise identity at initialization,
nonzero finite controller gradients, and numerical commutation with a shared
Spin(8) action. This is a token-conditioned query, not static capacity already
absorbed by the output projection.

The failed amplitude-readout gate is not bundled into this candidate:
`readout=direction` remains fixed, isolating multiplicity routing. Dense F4/E6
transport and a new delta recurrence are likewise excluded.

## Frozen protocol and rule

- baseline `multiplicity_router=none`;
- candidate `multiplicity_router=orthogonal_query`;
- fresh seeds `83`, `89`, and `97`;
- otherwise the exact 300-step Tiny Shakespeare protocol in
  `TRIALITY_INVARIANT_READOUT_PREREGISTRATION.md`;
- paired initialization, batches, validation windows, and cu126 WSL runtime.

The baseline has 626,516 parameters. The router adds one 128-to-1 controller
per layer, for 627,032 parameters: 516 extra, or 0.0824%. The fused Mamba-2
reference remains 623,740 parameters, a candidate/reference gap of 0.525%.

Positive improvement is `no-router bpb - orthogonal-query bpb`. Promote only
if all hold:

1. candidate wins at least two of three seeds;
2. mean improvement is at least `0.0100` bpb;
3. no seed regresses by more than `0.0500` bpb;
4. all values and gradients are finite and artifact compatibility passes.

Sequential timing is diagnostic only. A passing candidate still requires a
fresh matched Mamba-2 quality gate and order-balanced speed measurement. A
failure leaves the router as a research control and points the next experiment
toward multiplicity mixing inside the recurrence rather than at readout.
