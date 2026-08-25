# G15A-R first-order learning-repair protocol

**Frozen:** 2026-08-25, after the G15A-F failure and bound chart-error
decomposition, and before any G15A-R smoke, development, or confirmation
metric was produced or inspected

**Required G15A-F evidence SHA-256:**
`cdfdcb1785e2bf2a85ea592e2100a61596d1a06ea219a9d75c058f1d11e74296`

**Required post-hoc decomposition SHA-256:**
`6b36752060da872b14047da38d592a5a51d32e4e215fe00d577e92186877cb67`

## Diagnosis and question

G15A-F restored observability and comparator separation but missed every
absolute precision gate. The bound post-hoc decomposition is decisive: keeping
the learned active-plane amplitudes while zeroing only inactive coordinates
reduces mean relative error from `0.0705--0.1433` to `0.0051--0.0183` and
maximum error below `0.04` in every seed and length. Restoring exact amplitudes
while retaining leakage barely improves the failed tables.

The learned controller therefore finds the correct axes and nearly correct
angles, but small off-axis rotations compound through long compositions. Can a
strictly first-order repair remove that leakage without supplying primitive
support? Which of learning-rate decay, a per-token covariant moment, and a
balanced primitive/inverse curriculum is actually necessary?

## Development ablation

Development uses fresh seeds `2237`, `2239`, and `2243`, full Spin transport
only, the unchanged dense `17 x 28` controller, FP32 on exact SM75, batch size
16, logical length 16, 600 updates, zero initialization, no weight decay, and
gradient clipping at 1.

| Arm | Second moment | Training data | Learning rate |
|---|---|---|---|
| `G-fixed/random` | one scalar for the whole table | original random 2--6 actions | `0.05` throughout |
| `G-decay/random` | one scalar for the whole table | original random 2--6 actions | staged decay |
| `B-decay/random` | one scalar per token row | original random 2--6 actions | staged decay |
| `G-decay/curriculum` | one scalar for the whole table | frozen curriculum | staged decay |
| `B-decay/curriculum` | one scalar per token row | frozen curriculum | staged decay |

The staged schedule is fixed by update number:

- updates 1--100: `0.05`;
- updates 101--300: `0.01`;
- updates 301--600: `0.002`.

The curriculum is:

- updates 1--100: every batch contains each of the sixteen signed singleton
  primitives exactly once;
- updates 101--200: every batch contains both orders of all eight inverse pairs
  exactly once;
- updates 201--600: the original random two-to-six-action sampler.

The curriculum supplies token identities and end targets only. It does not
supply token-to-coordinate labels, primitive support, learned coordinates, or
intermediate losses.

## Validation and selection

A fixed, untouched 32-composition validation bank at lengths/actions
`(64,8)`, `(256,12)`, and `(1024,16)` is evaluated after updates
`1,25,50,100,150,200,300,450,600`. It is diagnostic only: no early stopping,
best-checkpoint selection, retry, or adaptive schedule is allowed.

Final development adjudication uses a separate fresh 80-composition bank at
the same three lengths. A candidate qualifies only if, in every seed and
length, mean relative Frobenius error is at most `0.05`, p95 at most `0.10`,
and maximum at most `0.20`.

If multiple candidates qualify, choose the least intervention in this frozen
order:

1. `G-decay/random`;
2. `B-decay/random`;
3. `G-decay/curriculum`;
4. `B-decay/curriculum`.

`G-fixed/random` is a longer-budget failure control and cannot be selected. If
no candidate qualifies, G15A-R fails without a confirmation run.

## Fresh confirmation

The programmatically selected recipe is then reinitialized and run once on
untouched seeds `2251`, `2267`, and `2273` with all four transports `I`, `C`,
`S`, and `S-broken`. The recipe, 600-update budget, evaluation bank, and all
other settings are shared exactly across arms.

The original G15A-F gates remain binding in every seed and length:

1. S mean relative error `<=0.05`, p95 `<=0.10`, and maximum `<=0.20`.
2. I, C, and S-broken mean error each exceed S by at least `0.05`.
3. S-broken mean error is at least twice S.
4. Parameter count, initialization, probe bank, and training schedule match
   exactly; all values remain finite.

There is no seed averaging rescue. Development and confirmation results are
stored in one source-bound artifact so selection cannot be changed between
stages.

## Optimizer contract

`BlockScalarSecondMomentAdamW` stores one scalar second moment per final-axis
vector block. For the coordinate table this is shape `17 x 1`. It preserves a
shared orthogonal change of the 28-coordinate chart and equivariance under
token-row permutations. This is a targeted ablation, not a prior superiority
claim; the global scalar optimizer remains the less-intervention control.

## Claim boundary

A pass would establish curriculum- or decay-assisted primitive-coordinate
identification and longer compositional generalization under the existing
four-probe oracle-frame objective and oracle edit timing. If a random-data
decay arm wins, the stronger composition-only wording is retained. If a
curriculum arm wins, the claim is explicitly weakened to curriculum-assisted
primitive identification. No pass establishes learned addressing, natural
language, generic association, negative-spin/Clifford benefit, full triality,
moving exceptional geometry, scaling, or fused efficiency.
