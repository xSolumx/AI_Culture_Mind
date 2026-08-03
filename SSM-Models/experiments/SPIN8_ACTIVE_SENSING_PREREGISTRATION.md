# Spin(8) Active Triality Sensing Preregistration

**Frozen:** 2026-08-03, before implementation outcomes or sensor optimization
results were inspected.

## Question

Can a learned query policy discover a well-conditioned five-query experiment
that identifies an unknown shared Spin(8) action, and does active adaptation to
the unknown action provide any local information advantage?

The preceding gate established that five generic probes spanning at least two
triality representations are locally identifying, while four mixed probes or
five single-view probes leave a three-dimensional stabilizer. It did not ask
which identifying five-probe design is best under noisy observations, or
whether the probe design must depend on the unknown action.

## Frozen theorem candidate: adaptivity is locally unnecessary

For query `k`, choose a representation `r_k` and unit probe `x_k`. The observed
endpoint is

\[
y_k = Q_{r_k}x_k.
\]

Parameterize a local unknown perturbation by right multiplication,

\[
Q_{r_k}(\delta)=Q_{r_k}\exp\!\left(\sum_{a=1}^{28}\delta_aG_{r_k,a}\right).
\]

At the identity perturbation,

\[
J_{k,Q}[:,a]=Q_{r_k}G_{r_k,a}x_k.
\]

Because `Q_r` is orthogonal,

\[
J_{k,Q}^{\mathsf T}J_{k,Q}
=J_{k,I}^{\mathsf T}J_{k,I}.
\]

Therefore the local Fisher/information matrix

\[
\mathcal I(\mathcal D)=\sum_kJ_k^{\mathsf T}J_k
\]

depends only on the chosen probe design `D`, not on the unknown action or its
observed endpoints. If verified, no observation-adaptive policy can improve a
local D-optimal objective over the best universal five-query design.

This is a local statement. It does not rule out global disambiguation benefits
from adaptive queries outside the sampled action neighborhood.

## Frozen sensor variants

All variants receive exactly five unit-norm state queries.

| Variant | Construction | Purpose |
|---|---|---|
| `learned_hard_doptimal` | straight-through hard representation choices plus learned probe vectors; maximize `log det(I)` | Can optimization discover the universal sensor? |
| `oracle_doptimal` | enumerate every five-query representation allocation and optimize continuous probes with multiple restarts | Numerical ceiling |
| `random_mixed` | random unit probes, conditioned on using at least two views | Generic identifiable baseline |
| `fixed_1_4_0` | previous `(1 V, 4 S+, 0 S-)` design | Known minimal completion baseline |
| `single_view_doptimal` | five optimized probes in `V` only | Rank-25 negative control |

The learned forward pass always selects one real representation per query. A
soft distribution may be used only as a straight-through gradient surrogate;
hard-view rank and conditioning determine every reported result.

No diversity, rank, or mixed-view penalty is permitted in the learned sensor
objective. Otherwise the policy would be told the answer.

## Frozen protocol

### Sensor development and reliability

- seed 0 is a development smoke used only to select optimizer budget and
  annealing schedule;
- after that smoke, hyperparameters are frozen;
- untouched sensor seeds 10-19 form the ten-seed reliability cohort;
- all probe initialization occurs in canonical CPU float64 streams;
- every hard sensor is audited in float64.

### Oracle search

- enumerate all 21 allocations `(n_V,n_+,n_-)` with total five;
- use the same continuous optimization budget for every allocation;
- use at least four deterministic restarts;
- rank-deficient allocations remain in the table and may not be silently
  discarded;
- select by unregularized hard-design log determinant among rank-28 designs.

### Noisy held-out recovery

After sensor selection is frozen, each sensor seed is evaluated on an untouched
teacher family. The five endpoint vectors receive matched isotropic Gaussian
noise with standard deviation `1e-3`. Every action estimator:

- uses the same shared 28-parameter Spin(8) chart;
- uses the same Adam/L-BFGS budget and initialization scale;
- sees the same teacher and matched noise realization;
- is evaluated on unseen states and shared token words at lengths
  `16,32,64,128,256,512,1024,2048`.

The single-view family is expected to remain nonidentifiable. Its failure is a
structural control, not evidence about optimizer quality.

## Frozen measurements

For each hard design:

- view allocation;
- information rank and nullity;
- minimum information eigenvalue;
- log determinant;
- condition number;
- `trace(I^-1)` for rank-28 designs;
- one-step action cosine after noisy fitting, including the least-observed
  representation (ties resolved by fixed representation order);
- dense long-composition cosine;
- triality equivariance, scan parity, and norm drift.

The action-independence theorem is checked against random noncommuting teacher
actions by directly comparing `I_Q` with `I_identity`.

## Frozen gates

### A. Action-independence gate

Across ten random action families and every evaluated sensor design:

- maximum absolute information-matrix difference below `1e-10`;
- numerical rank and spectrum agree within `1e-10` relative tolerance.

### B. Learned-design gate

At least 8/10 untouched learned sensors must:

- use at least two triality views after hard rounding;
- have information rank 28;
- have log determinant no more than `0.10` below its same-seed oracle;
- have `trace(I^-1)` no more than 10% above the oracle.

### C. Noisy recovery gate

At least 8/10 learned sensors must:

- fit their noisy observations to the noise floor without numerical failure;
- beat the matched random-mixed sensor in least-observed-view one-step cosine;
- beat the matched random-mixed sensor in worst-representation cosine at
  length 2048;
- remain within `0.02` of the oracle sensor at length 2048;
- preserve triality equivariance below `1e-8`, scan parity below `1e-9`, and
  absolute log-norm drift below `1e-5`.

### D. Structural control gate

- `single_view_doptimal` must have rank 25 and nullity 3 in every seed;
- `random_mixed` and `fixed_1_4_0` must generically have rank 28;
- all recovery variants must receive exactly five queries and identical noise.

## Interpretation boundaries

- Passing A would prove only local action-independence of Fisher information,
  not global irrelevance of adaptive queries.
- Passing B would show that a hard learned sensor can discover the mixed-view
  geometry without a hand-coded diversity term.
- Passing C is required before claiming that D-optimal sensing has practical
  value beyond the noiseless five-probe rank theorem.
- A failure of C with a pass on A/B would mean that the D-optimal objective is
  mathematically valid but not the right finite-noise/long-composition design
  criterion.
- No language-model, semantic-routing, or sample-efficiency claim is authorized
  by this gate.
