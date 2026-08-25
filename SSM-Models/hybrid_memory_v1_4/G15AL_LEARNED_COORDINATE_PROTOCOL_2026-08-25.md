# G15A-L learned-coordinate protocol

**Frozen:** 2026-08-25, after the G15A conditional controls completed and
before any G15A-L smoke or quality metric was produced or inspected

**Required conditional evidence SHA-256:**
`78cb06c0e7d088db74651fc93f3c40380f7d3f0f04d447bf54b75ff263c3ffe9`
for
[`artifacts/g15a_conditional_controls_sm75_2026-08-25.json`](artifacts/g15a_conditional_controls_sm75_2026-08-25.json)

## Question

G15A showed that a shared vector/positive Spin lift matters when exact action
coordinates are supplied, while the fixed Clifford/negative-spin second read
is unnecessary. Can a deliberately minimal token controller learn those
coordinates from delayed end loss and generalize to unseen ordered action
compositions? Or can the broken carrier absorb its mismatch into a learned
coordinate reparameterization?

This is the missing learning problem exposed by G15A. It is tested before
adding more geometry or spending on generic association and natural text.

## Frozen arms and scored sector

| Arm | Transport | Read mode |
|---|---|---|
| `I` | identity | identity |
| `C` | fixed `SO(2)^4` | Clifford |
| `S` | full factorized Spin(8) | Clifford |
| `S+identity-read` | full factorized Spin(8) | identity |
| `S-broken` | frozen non-automorphic coordinate permutation on the positive carrier | Clifford |

Only the positive eight-dimensional read is scored. The completed conditional
control already showed that the second read is unnecessary on this task.
`S+identity-read` is retained as an exact parity control; `I+C` is omitted
because it is identical to I on the scored positive sector.

Every arm trains exactly one `17 x 28` raw coordinate table (476 scalars).
Token zero is hard-masked to identity and its 28 stored scalars are reported as
structurally unused. The other sixteen tokens represent positive and negative
increments in the eight frozen off-torus planes. A deterministic seed-specific
permutation hides the token-to-plane map. Coordinates are bounded as

\[
c_t = 0.25\tanh(w_{\mathrm{token}_t}),
\]

and the exact target increment magnitude is `0.12`.

## Workload

Each episode writes one random unit vector at a random unit key. It then
observes a sparse ordered sequence of primitive action tokens. Erase is zero,
write is one only at the initial position, and retention is `0.999999`.

The final query is the exact vector-carrier transport of the initial key. This
oracle query forces the learned coordinate map to remain compatible with both
the vector and positive carriers. The target is the positive read of the exact
S teacher. Neither exact coordinates nor the target action are supplied to the
learned controller.

Training uses random unseen-on-replay compositions at length 16 with two to six
actions. Evaluation uses fresh deterministic compositions:

| sequence length | actions per episode | examples |
|---:|---:|---:|
| 64 | 8 | 80 |
| 256 | 12 | 80 |
| 1,024 | 16 | 80 |

Evaluation is microbatched by eight. Keys, values, token permutations,
positions, signs, and action orders are paired exactly across arms.

## Optimizer and budget

- Seeds: `2153`, `2161`, and `2179`.
- FP32; exact local compute capability 7.5.
- 300 updates, batch size 16, length 16.
- `ScalarSecondMomentAdamW`, learning rate `0.05`, no weight decay.
- Loss: mean `1 - cosine` on the final positive read.
- Gradient clipping at norm 1.
- Controller initialization is exactly zero and must byte-match across arms.

The optimizer keeps one scalar second moment for the whole coordinate tensor;
unlike coordinatewise AdamW, its second moment is invariant under an
orthogonal change of the 28-coordinate chart. This is a covariance contract,
not a general optimizer-superiority claim.

## Frozen adjudication

For every seed and every evaluation length separately:

1. S mean cosine must be at least `0.995` and minimum-example cosine at least
   `0.98`.
2. S and S+identity-read mean cosine must agree within `1e-5`, and their
   learned coordinate tables within `1e-5` maximum absolute difference.
3. S mean cosine must exceed each of I, C, and S-broken by at least `0.05`.
4. Parameter count, initialization hash, and training-schedule hash must match
   exactly across all arms.

All conditions must hold in all three seeds; there is no averaging rescue. If
S-broken catches S, shared coupling is not identifiable under this learned
controller and oracle-query observation map. That is a negative result for
autonomous attribution, not a reason to change the task or threshold after
inspection.

## Claim boundary

A pass would establish learned primitive coordinates and compositional
generalization under oracle edit controls and an oracle transported query. It
would not establish learned addressing, learned querying, generic association,
natural language, long-context factual recall, scaling, fused efficiency, or
a moving `G2/SU(3)` frame. A failure localizes the next repair to observability
or controller identifiability rather than the memory law.
