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
- all baseline cells must replay within `1e-12`;
- the G15B-R0 temporal-observability witness and exact local write decoder must
  pass;
- intervention controls other than erase must be bitwise identical;
- report mean and maximum absolute off-diagonal cosine between learned live-key
  prototypes. This measures collateral rank-one erase interference.

## Frozen decision

An intervention authorizes fresh identity event-erase training only if every
three-seed mean cell satisfies:

- overwrite improves by at least `0.10` absolute over learned;
- MQAR and selective copy trail learned by no more than `0.02`;
- needle accuracy is at least `0.999`.

If both pass, choose the smaller mean absolute non-needle regression, breaking
ties in favor of `soft_event_erase`. If neither passes, do not train this
controller. High prototype cross-cosine with unique-key damage selects an
orthogonalization/address-capacity repair; low cross-cosine selects a more
explicit state-aware correction or separate erase address.

## Nonclaims

No R1 result promotes G15C, external-loss-only learning, ordinary next-token
training, an optimizer, Spin transport, natural language, scaling, or a model
family. It is a paired retained-checkpoint causal diagnostic. All earlier G15
results and their original claim boundaries remain unchanged.
