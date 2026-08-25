# G15A-S spanning-chart and center-sensitive transfer protocol

**Frozen:** 2026-08-25, after the complete G15A through G15A-R chain and
before any G15A-S smoke or quality metric is inspected

**Bound predecessor:**
[`artifacts/g15ar_first_order_repair_sm75_2026-08-25.json`](artifacts/g15ar_first_order_repair_sm75_2026-08-25.json),
SHA-256 `be004dea821e9bf38e140f627fe387829d67949767be045310cf2080ac7d6fe8`

## Why this track exists

G15A-R proves that a 17-by-28 token lookup can learn sixteen signed actions on
eight fixed off-torus planes from a four-probe final-state loss. It does not
show that the result spans all 28 Spin(8) generators, transfers to unseen
probe banks, or remains compatible with the global/discrete Spin center.

The old results remain distinct:

- supplied-coordinate G15A passed;
- the Clifford/negative read was unnecessary in its conditional control;
- G15A-L failed because its scalar cosine observation quotiented out the
  vector carrier;
- G15A-F restored observability but failed precision because inactive
  coordinate leakage accumulated;
- G15A-R repaired precision with longer first-order training; fixed-rate
  random training itself eventually passed, so neither decay, block moments,
  nor primitive curriculum was proven necessary.

G15A-S asks the strongest remaining pre-G15B question: can the selected AR
recipe learn a full signed Lie-algebra dictionary and integrate it around
unseen global Spin words? Addressing, write timing, and query selection remain
oracle-fixed and are not smuggled into this track.

## Controller and hidden action dictionary

The controller is a `57 x 28` table:

- token zero is masked identity;
- 56 active tokens represent the positive and negative directions of all 28
  coordinate planes;
- the token-to-plane/sign dictionary is hidden by a fresh seed-specific
  permutation;
- every active primitive has magnitude
  `delta = pi / 16 = 0.196349...`, below the model's `0.25` chart bound.

Training supplies tokens only. It never supplies primitive labels, coordinate
targets, carrier matrices, center labels, or intermediate-state losses.

## Training observation and its exact limitation

For four orthogonal probes per example, training observes only

\[
M_T^{(j)} = V_T M_0^{(j)} P_T^\top,
\]

with raw elementwise Frobenius MSE. `V` and `P` are the shared vector and
positive-spin actions. This retains G15A-R's composition-only objective.

The product observation has central character `chi_v * chi_p`; one nontrivial
central element acts with the same sign on both carriers and is therefore
invisible in `V M P^T`. More probes cannot repair that discrete quotient. At
evaluation only, G15A-S also reports `V` and `P` directly. Those two carrier
representations together resolve all four center signatures. Direct-carrier
evaluation is an assay, not a training label.

## Probe-bank transfer

Each seed has 64 deterministic training banks and 64 disjoint evaluation
banks. Every bank contains four independently sampled orthogonal 8-by-8
matrices. Before training, every bank must have:

- independent-carrier tangent Jacobian rank 56;
- `sigma_min / sigma_max >= 0.10`; and
- every one of the 28 target tied primitive tangents at relative projection
  residual at least `0.05` outside the complete `S-broken` tied tangent image.

Training examples choose among only the training pool. All reported random
composition metrics use only the held-out pool. Pool membership and hashes
are frozen in the artifact.

## Arms and optimization

| Arm | Transport |
|---|---|
| `I` | identity |
| `C` | fixed `SO(2)^4` mask |
| `S` | exact shared vector/positive Spin(8) lift |
| `S-broken` | frozen non-automorphic positive-carrier coordinate map |

Every arm stores the same 1,596 scalar table. Inactive coordinates are masked,
not removed.

- fresh seeds: 2281, 2287, 2293;
- exact FP32 SM75 execution;
- 600 updates, batch 32, logical length 16;
- fresh uniform random compositions of two to six signed primitives;
- `ScalarSecondMomentAdamW`, no weight decay, gradient clip 1;
- learning rate 0.05 for updates 1--100, 0.01 for 101--300, and 0.002 for
  301--600;
- no curriculum, development sweep, early stopping, or best-checkpoint
  selection.

The zero initialization, token dictionary, probe pools, and complete sequence
schedule must byte-match across arms.

## Held-out evaluation

Random compositions use 80 examples at 8, 12, and 16 actions, paired across
arms and drawn only from held-out probe banks. Each example receives a bank
selected independently of its action word.

Structured center schedules are never used for training:

1. positive and negative 2-pi loops in every plane: 32 repeats;
2. positive and negative 4-pi loops in every plane: 64 repeats;
3. the positive and negative volume words: 16 repeats in each of
   `(0,1),(2,3),(4,5),(6,7)`;
4. each volume word preceded by an off-torus 2-pi loop, generating its other
   center coset; and
5. loop-plus-noncommuting-primitive continuations, preventing a hard-coded
   word-length/parity solution.

An exact-coordinate FP64 oracle must match analytic carrier signs for all
four center elements and replay every structured schedule within `1e-10`
before learned metrics are admissible.

## Frozen gates

All conditions must pass in every seed; there is no mean-over-seeds rescue.

1. On each random length, S frame mean relative error is at most `0.01`, p95
   at most `0.02`, and maximum at most `0.05`.
2. I, C, and S-broken are each at least `0.05` worse than S in mean frame
   error, and S-broken mean error is at least twice S.
3. Across all 56 learned primitive rows, maximum active-coordinate absolute
   error is at most `5e-4`, and inactive-coordinate RMS is at most `1e-4`.
4. On every structured word, learned S vector and positive-carrier relative
   Frobenius errors are each at most `0.01`.
5. Every 2-pi loop closes in the vector carrier and reaches the opposite
   positive-spin sheet; every 4-pi loop returns both carriers to identity.
6. Volume, center-coset, and loop-plus-primitive words match the exact signed
   continuation.
7. At least one nonidentity center word must be invisible to two-sided frame
   scoring while direct V/P scoring detects it. This is the required
   projective-assay sensitivity control.
8. Parameters, initialization, token map, probe pools, action schedules, and
   all finite-value checks pass exactly.

If the exact-coordinate oracle fails, execution stops as an implementation or
protocol failure. If learned S fits random frames but fails any direct-carrier
center word, G15A-R established only a local/projective chart. If S-broken ties
S on the full-28 fresh-frame gate, shared coupling is still unidentifiable at
full algebra support. Neither failure may be threshold-adjusted or rescued by
G15B.

## Claim boundary

A pass supports only that a composition-only token lookup learns a complete
28-generator signed dictionary whose frozen coordinates remain compatible
with the repository's hard-coded shared vector/positive Spin lift on unseen
frames and center-sensitive words under oracle edit timing. It does not mean
the controller discovered Spin topology from raw data. It does not establish
negative-spin/Clifford benefit, full triality utility, learned address/write/
query behavior, generic association, language quality, scaling, or fused
efficiency.
