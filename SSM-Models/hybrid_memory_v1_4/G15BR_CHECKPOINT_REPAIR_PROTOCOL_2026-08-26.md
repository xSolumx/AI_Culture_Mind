# G15B-R0 gauge-preserving checkpoint repair protocol

**Frozen:** 2026-08-26, after the complete G15B artifact was recorded and
before any repair-intervention retrieval metric was inspected  
**Parent result:**
[`G15B_INTERLEAVED_CONTROLLER_RESULTS.md`](G15B_INTERLEAVED_CONTROLLER_RESULTS.md)  
**Status:** prospective diagnostic; no model-promotion authority

## Question

G15B learned content addresses and causally used its recurrent association
matrix, but failed last-write-wins editing. Its erase label asks a token-local
controller to decide whether the current key appeared arbitrarily earlier.
Before another trained cohort, G15B-R0 asks the narrower causal question:

> Do the retained identity checkpoints already contain usable learned
> addresses, values, reads, and a write-role signal when the edit law is changed
> from collision-conditioned erase to a delta correction?

The delta correction ties erase and write strength:

\[
M_t=(I-\beta_t k_tk_t^\top)r_tM_{t-1}
    +\beta_t k_tv_t^\top.
\]

It erases the addressed component on every write. A first write to an empty,
orthogonal address is a harmless no-op; an overwrite replaces the old value.
With nonorthogonal learned keys, cross-address interference remains empirical
and is measured rather than assumed away.

## Bound checkpoints and data

Use only the three retained G15B identity checkpoints for seeds 2309, 2311,
and 2333. Their hashes and configurations must match the completed G15B
artifact. No parameter is updated.

Replay the exact held-out G15B evaluation schedule: MQAR, overwrite, selective
copy, and needle at lengths 128, 512, and 1,024, with 4,096 query decisions per
cell and evaluation batch cap 16. Exact baseline replay against the recorded
checkpoint metrics is an integrity gate. Reusing these held-out cells is
intentional for a paired post-training intervention and cannot support a fresh
generalization or promotion claim.

## Frozen interventions

Every intervention preserves the learned key/query/value vectors, output
decoder, Clifford/readout path, retention, and identity transport. This avoids
the one-hot address-gauge mismatch in the original G15B learned-path oracle.

1. `learned`: untouched checkpoint controls.
2. `soft_delta`: set learned erase strength equal to learned write strength.
3. `exact_collision_timing`: use exact valid-write timing for write and exact
   collision-only timing for erase, while retaining the learned address/value
   gauge.
4. `exact_delta_timing`: set both erase and write to one on every valid write
   and zero elsewhere, retaining the learned address/value gauge.

The two exact-timing arms are diagnostics, not deployable learned controllers.

## Temporal-observability witness

Construct two sequences whose width-four windows at a write-value position are
identical. In one sequence the key occurred earlier; in the other it did not.
The frozen overwrite labels therefore differ while every control produced by
the one-block token/local-convolution path at that position is identical.
The runner must record exact token-window equality, opposite collision labels,
and a maximum FP32 control residual at most `5e-7`. Failure stops the
diagnostic.
The repaired valid-write target must also replay exactly from the local grammar:
at a value position it is one exactly when the token two positions earlier is
`WRITE` or `SELECT`. This equality is checked on every evaluation batch.

## Frozen decisions

All three checkpoint baselines must replay their recorded query accuracies
within `1e-12`.

`soft_delta` is sufficient to authorize a fresh delta-law training cohort only
if, on every three-seed mean cell:

- overwrite improves by at least `0.10` absolute over `learned`;
- MQAR and selective copy trail `learned` by no more than `0.02`;
- needle accuracy is at least `0.999`.

If `soft_delta` fails but `exact_delta_timing` reaches at least `0.95` on every
MQAR, overwrite, and selective mean cell and at least `0.999` on every needle
cell, a fresh identity delta-law controller cohort is still authorized. That
outcome says the state/read/value system is adequate but write-role learning
must be repaired during fresh training.

If `exact_delta_timing` fails those thresholds, do not train the delta-law
successor. Investigate key orthogonality, value decoding, and write-tail
interference first.

`exact_collision_timing` is reported as a gauge-preserving reference. It has no
independent promotion threshold.

## Nonclaims

No outcome promotes G15C, natural-text training, Spin transport, autonomous
controller discovery, an optimizer, or a model family. This is a frozen
checkpoint intervention on commissioned synthetic data. G15A-S remains
separate supplied/oracle-timing geometry evidence, and the completed G15B
failure is never rewritten.
