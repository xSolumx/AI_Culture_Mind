# G15B-R1 event-anchored erase protocol

**Frozen:** 2026-08-26, after G15B-R0 and before any R1 intervention metric is
inspected.

**Entry evidence:**
[`G15BR_CHECKPOINT_REPAIR_RESULTS.md`](G15BR_CHECKPOINT_REPAIR_RESULTS.md)

**Status:** prospective zero-update checkpoint diagnostic

## Question

G15B-R0 rejects naive erase-equals-write delta because the retained identity
controllers use a structured one-token write continuation. R1 asks:

> Can last-write-wins improve when the learned write microprogram is preserved
> exactly and only erase is moved from unobservable collision timing to the
> locally observable write event?

The candidate retains independent controls:

\[
M_t=(I-e_tk_tk_t^\top)r_tM_{t-1}+k_t(w_tv_t)^\top.
\]

`w_t` is untouched, including its learned continuation. `e_t` is nonzero only
at the labelled write-value token, whose marker is visible two tokens earlier
inside the width-four causal window.

## Bound checkpoints and evaluation

Use the same retained G15B identity checkpoints for seeds 2309, 2311, and 2333
and the same 4,096-decision held-out cells at MQAR, overwrite, selective copy,
and needle lengths 128, 512, and 1,024. Baseline replay must match the recorded
G15B artifact exactly. No parameter is updated.

The frozen parent hashes are:

- G15B: `f74d860e30ab40ec747521dfcecd74aac2bb75151206c25b7104d334727429eb`;
- G15B-R0: `4d92d6af2fb062cf2baaa035c4e4eff89d494dfcb56b9b666523bbbdbfe3cf9c`.

Both inputs must be clean-start, evidentiary `quality` artifacts from SM75 with
the exact seeds above, and R0 must bind the supplied G15B hash.

## Frozen interventions

All modes preserve learned query, key, value, write, retention, output gate,
decoder, and identity transport.

1. `learned`: untouched controls.
2. `soft_event_erase`: erase equals learned write strength at exact valid-write
   positions and zero elsewhere; learned write remains untouched everywhere.
3. `exact_event_erase`: erase is one at exact valid-write positions and zero
   elsewhere; learned write remains untouched everywhere.

Both repair arms deliberately preserve the post-write continuation. They
change only erase. The exact arm is a mechanism diagnostic, not a deployable
controller; the soft arm still uses the exact local event mask and is likewise
not autonomous discovery.

## Integrity and interference diagnostics

- all checkpoint and parent-artifact hashes must match;
- all baseline query accuracy, exact-episode accuracy, and bits/query cells must
  replay within `1e-12`, while the ordinary model-forward logits and the
  learned-control logits must be bit-identical;
- the G15B-R0 temporal-observability witness and exact local write decoder must
  pass;
- intervention controls other than erase must be bitwise identical in every
  retained-checkpoint evaluation batch;
- report mean and maximum absolute off-diagonal cosine between learned live-key
  prototypes. This measures collateral rank-one erase interference.

## Frozen decision

An intervention authorizes fresh identity event-erase training only if every
three-seed mean cell satisfies:

- overwrite improves by at least `0.10` absolute over learned;
- MQAR and selective copy trail learned by no more than `0.02`;
- needle accuracy is at least `0.999`.

If both pass, choose the smaller mean nonnegative accuracy degradation over the
six MQAR and selective-copy cells, breaking ties in favor of
`soft_event_erase`. Overwrite is the required benefit and needle already has an
absolute gate, so neither participates in this tie-break. If neither passes,
do not train this controller. High prototype cross-cosine with unique-key
damage selects an orthogonalization/address-capacity repair; low cross-cosine
selects a more explicit state-aware correction or separate erase address.

## Exact reproduction

From the repository root in the SM75-capable WSL environment:

```bash
PYTHONPATH=SSM-Models /home/local/.venvs/sm75-native-2026/bin/python \
  -m hybrid_memory_v1_4.g15br1_event_erase \
  --mode quality --device cuda \
  --checkpoint-directory /home/local/g15b_bd5045a_quality_attempt2_checkpoints \
  --output /home/local/g15br1_<commit>_quality.json
```

## Nonclaims

No R1 result promotes G15C, external-loss-only learning, ordinary next-token
training, an optimizer, Spin transport, natural language, scaling, or a model
family. It is a paired retained-checkpoint causal diagnostic. All earlier G15
results and their original claim boundaries remain unchanged.
