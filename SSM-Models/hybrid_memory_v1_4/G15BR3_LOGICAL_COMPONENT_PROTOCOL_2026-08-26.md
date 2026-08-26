# G15B-R3 exact logical-component replacement protocol

**Frozen:** 2026-08-26, after G15B-R2 and before any R3 intervention metric is
inspected.

**Entry evidence:**
[`G15BR2_COLLISION_ERASE_RESULTS.md`](G15BR2_COLLISION_ERASE_RESULTS.md)

**Status:** prospective zero-update checkpoint diagnostic

**Pre-execution amendment:** repository/generator audit found that the token
immediately after a write may be the next event marker rather than filler. No
retained-checkpoint R3 metric had been inspected. The frozen convention below
therefore assigns the in-range `t+1` injection by temporal position, reports
its role, resets on every locally observable valid write, and adds a constructed
guard cohort that populates the unrelated-overwrite stratum.

## Question

G15B-R2 supplies perfect causal collision timing and preserves the learned
two-token write microprogram, but symmetric learned-key erase still lowers
post-same-key-overwrite recall by 10.3--12.1 points. R3 asks the narrower
representation question:

> Can the retained decoder use exact last-write-wins when a frozen learned
> value-token plus one-token-tail contribution is assigned to its logical key
> and that whole component, rather than a rank-one direction, is replaced at a
> true overwrite?

This is an oracle expanded-slot decomposition of an already trained linear
state. It tests the edit law learned by the checkpoint; it is not an autonomous
controller or a compute-matched architecture.

## Bound evidence and runtime

Use the retained G15B identity checkpoints for seeds 2309, 2311, and 2333, the
exact G15B evaluation namespace, 4,096 decisions per task/length cell, and batch
cap 16 at lengths 128, 512, and 1,024. Perform zero optimizer updates.

Frozen artifact hashes:

- G15B: `f74d860e30ab40ec747521dfcecd74aac2bb75151206c25b7104d334727429eb`;
- G15B-R0: `4d92d6af2fb062cf2baaa035c4e4eff89d494dfcb56b9b666523bbbdbfe3cf9c`;
- G15B-R1: `c015b128846e4b5c63d927778815a87728a7d613369163b1027ed3dd9f0b2912`;
- G15B-R2: `90652fe7034e5901b968eb5d139f02eb8bc714b0417c0889e16a2fdd6b7cf924`.

All parents must be clean-start evidentiary `quality` results from exact SM75,
use the frozen seeds, bind their direct parents, and have failed their
registered adjudications.

## Frozen state decomposition

Let the ordinary no-symmetric-erase recurrence be

\[
M_t=A_tM_{t-1}B_t^\top+U_t,
\]

where `A_t` retains the learned retention and identity-transport action, `B_t`
is the learned positive-carrier action, and the learned rank-one erase operator
is disabled. Decompose

\[
M_t=C_t^{(0)}+\sum_{j=1}^{K}C_t^{(j)}.
\]

`C^(0)` receives every background injection. `C^(j)` receives the learned
injection at each value token written to logical key `j` and at the immediately
following in-range token. That token may be filler or the next event marker;
its role is reported. This is a frozen temporal ownership convention, not a
claim that the learned representation has a unique semantic decomposition.
Every component otherwise receives the same learned linear transition.

At every valid write of logical key `j`, `erase_free_lww` sets the left
transition for `C^(j)` to zero at that value-token position before adding the
new learned injection. The first reset is a provable no-op because that key
component is initially empty; subsequent resets implement last-write-wins
without an unobservable collision label. All other components remain
untouched:

\[
C_t^{(j)}=U_t^{(j)}
\]

at the write event, followed by the untouched learned one-token tail.

The one-token tail is frozen because R0 identified it prospectively as a strong
structured continuation. Every write owns its value-token injection and, when
`t+1` exists, that following-token injection. A final-token write has no tail
and is reported separately; ownership conflicts fail closed. R3 does not claim
that this convention captures the complete learned logical association.

## Frozen interventions

1. `learned`: untouched checkpoint recurrence.
2. `learned_decomposed_replay`: learned transition and frozen injection
   partition, with no reset. This integrity arm must reconstruct arm 1.
3. `erase_free_no_reset`: the same decomposition with scalar symmetric erase
   disabled and no reset. It must reconstruct an internal erase-free monolithic
   control.
4. `erase_free_lww`: arm 3 with exact logical-key component reset at every
   valid write.

Learned query, key, value, write, retention, output gate, decoder, identity
transport, and the frozen two-token write ownership remain unchanged in all
non-baseline arms. Disabling symmetric erase is part of the frozen state-law
pivot: the component reset replaces its erase function rather than stacking a
second erase mechanism on top.

## Query strata

For overwrite cells, report the same mutually exclusive causal strata as R2:

1. `before_any_overwrite`;
2. `after_unrelated_overwrite_only`;
3. `after_same_key_overwrite`.

If a frozen schedule does not populate a stratum, report zero support and make
no claim about it.

R3 also evaluates a deterministic constructed guard schedule at every length:

```text
write A -> write B -> query A -> query B -> overwrite A
-> query B -> query A -> query B -> query A -> query B -> query A
```

It yields two `before_any_overwrite`, three
`after_unrelated_overwrite_only`, and three `after_same_key_overwrite` queries
per episode. Keys, values, event gaps, rows, and seeds are deterministic and
balanced; the guard uses the same vocabulary and validated causal batch
contract. Failure to populate any guard stratum invalidates adjudication.

**Smoke-calibrated numerical amendment before quality:** the retained-checkpoint
smoke preserved query predictions and reconstructed the component-summed FP32
state within `3.6e-7`, but downstream RMS normalization amplified maximum logit
residual to `3.1e-4`. No quality result had run and no performance threshold
was changed. The parity gate below therefore requires state residual at most
`2e-6`, logit residual at most `5e-4`, and identical query predictions.

**Fail-closed pre-metric quality amendment:** the first quality invocation
stopped before evaluating a batch because one frozen schedule writes at its
final token. The ownership convention now records that write's value injection
without inventing an out-of-range tail. No quality metric or artifact was
produced, and all performance gates remain unchanged.

## Integrity

- parent and checkpoint hashes must match exactly;
- learned query accuracy, exact-episode accuracy, and bits/query must replay
  G15B within `1e-12`;
- ordinary model-forward and reconstructed learned-control logits must be bit-
  identical;
- every token injection must belong to exactly one component;
- every valid value-token plus following in-range token injection must belong to the
  matching live-key component;
- learned-decomposed logits must reconstruct `learned`, and erase-free
  decomposed logits must reconstruct an erase-free monolithic recurrence, each
  with maximum state residual at most `2e-6`, maximum logit residual at most
  `5e-4`, and identical query predictions;
- exact valid-write/reset masks, the local-write decoder, and the R0 temporal-
  observability witness must pass;
- the result must start from a clean commit on CUDA compute capability `(7,5)`.

Report expanded diagnostic state separately. With eight live keys it is nine
times the base recurrent state before temporary scan storage. This experiment
is explicitly not state-, parameter-, compute-, or wall-time-matched.

## Frozen decision

`erase_free_lww` supports a fresh explicit-slot/occupancy state-law screen
only if every three-seed mean satisfies:

- overall overwrite improves by at least `0.10` over both `learned` and
  `erase_free_no_reset` at every length;
- `after_same_key_overwrite` improves by at least `0.10` over both controls at
  every populated length;
- `before_any_overwrite` and `after_unrelated_overwrite_only` trail both
  `erase_free_no_reset` and `learned` by no more than `0.02` in the constructed
  guard cohort;
- MQAR and selective copy trail `learned` by no more than `0.02` at
  every length;
- needle accuracy is at least `0.999` at every length;
- every integrity gate passes.

A pass authorizes only a separately frozen fresh-training screen with explicit
slots, occupancy state, parameter/compute controls, and no oracle labels at
inference. It does not revive G15C or the present token-local controller.

If replacement improves post-same-key recall but misses a guard, inspect tail
ownership and background/component coupling before training. If it does not
improve post-same-key recall, reject this frozen post-hoc ownership/reset
construction. That failure would not prove why the decoder fails, and would not
falsify last-write-wins learning, GDN2/KDA, dual-address edits, or training a
new explicit-slot architecture from scratch.

## Exact reproduction

From the repository root in the SM75-capable WSL environment:

```bash
PYTHONPATH=SSM-Models /home/local/.venvs/sm75-native-2026/bin/python \
  -m hybrid_memory_v1_4.g15br3_logical_component \
  --mode quality --device cuda \
  --checkpoint-directory /home/local/g15b_bd5045a_quality_attempt2_checkpoints \
  --output /home/local/g15br3_<commit>_quality.json
```

## Nonclaims

No R3 result is fresh training, autonomous collision detection, ordinary-text
evidence, or a matched model comparison. The component labels and reset timing
come from commissioned task metadata, and the state expands with the number of
live keys. R3 cannot promote Spin transport, G15C, external-loss-only learning,
an optimizer, natural language, scaling, or a model family. All earlier G15
results retain their original boundaries.
