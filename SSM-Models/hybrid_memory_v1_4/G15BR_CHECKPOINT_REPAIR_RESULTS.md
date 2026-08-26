# G15B-R0 gauge-preserving checkpoint repair results

**Executed:** 2026-08-26 | **Start commit:**
`f30343594d116365d0bf6017346463b319851465` | **Start status:** clean |
**Runtime:** WSL2, RTX 2070 SUPER, SM75, PyTorch 2.9.0+cu128

**Artifact:**
[`g15br_checkpoint_repair_sm75_2026-08-26.json`](artifacts/g15br_checkpoint_repair_sm75_2026-08-26.json)

**Artifact SHA-256:**
`4d92d6af2fb062cf2baaa035c4e4eff89d494dfcb56b9b666523bbbdbfe3cf9c`

## Bottom line

Do not train the naive erase-equals-write delta successor.

The result sharpens the G15B diagnosis. The old collision-only erase target is
indeed unobservable to the width-four token-local controller, but tying erase
to every learned write is not the repair. `soft_delta` reduces every MQAR,
overwrite, and selective-copy three-seed mean. It never improves overwrite.
Exact atomic write/erase timing is worse still because the retained models
learned a deliberate one-token write continuation after the labelled value
position. Removing that continuation destroys much of the learned code.

The next candidate is therefore not a scalar tied edit. It is a locally
observable, event-anchored erase at the labelled write-value token while the
write drive remains free to use a short causal microprogram. This preserves
the useful learned continuation and removes the impossible collision
classifier. It must be tested prospectively before fresh training.

## Integrity

Every evidence gate passes:

- the three retained identity checkpoint hashes match the completed G15B
  artifact;
- all 36 baseline cells replay their recorded G15B accuracy exactly, with
  maximum absolute residual `0.0`;
- both histories in the temporal-observability witness have identical
  width-four windows, opposite collision labels, and exactly identical FP32
  controller outputs at the target position;
- the valid-write target is exactly decoded from the local `WRITE/SELECT`, key,
  value grammar on every evaluation batch;
- the run starts from a clean commit on the bound SM75 runtime;
- no optimizer step or parameter update occurs.

The run evaluates 4,096 decisions per task/length/checkpoint and takes 1,733.5
seconds. This is a paired retained-checkpoint intervention, not fresh
generalization evidence.

## Three-seed paired result

Mean query accuracy:

| Task and length | Learned | Soft delta | Exact collision timing | Exact delta timing |
|---|---:|---:|---:|---:|
| MQAR 128 | **0.9724** | 0.8936 | 0.5623 | 0.5267 |
| MQAR 512 | **0.9729** | 0.9011 | 0.5649 | 0.5261 |
| MQAR 1,024 | **0.9718** | 0.9057 | 0.5682 | 0.5326 |
| overwrite 128 | **0.7678** | 0.6746 | 0.3980 | 0.3894 |
| overwrite 512 | **0.8328** | 0.7143 | 0.3949 | 0.3919 |
| overwrite 1,024 | **0.8288** | 0.7237 | 0.4039 | 0.3942 |
| selective 128 | **0.9719** | 0.8613 | 0.5217 | 0.5210 |
| selective 512 | **0.9770** | 0.8704 | 0.5180 | 0.5210 |
| selective 1,024 | **0.9752** | 0.8646 | 0.5164 | 0.5147 |
| needle 128 | 1.0000 | **1.0000** | 0.6434 | 0.6434 |
| needle 512 | 1.0000 | **1.0000** | 0.6420 | 0.6420 |
| needle 1,024 | 1.0000 | **1.0000** | 0.6441 | 0.6441 |

`soft_delta` loses 6.6--7.9 points on MQAR, 9.3--11.8 points on overwrite,
and 10.7--11.1 points on selective copy. The frozen requirement was a gain of
at least 10 points on every overwrite cell with at most a two-point loss on
MQAR/selective. It fails every non-needle cell.

Exact timing is not a clean improvement hidden by the learned write parser.
Both exact-timing arms collapse well below their `0.95` gate, including the
gauge-preserving exact collision reference. The problem is not the old one-hot
oracle gauge mismatch: learned keys, queries, values, retention, and decoder
remain unchanged here.

## What the write-F1 failure actually contained

The labelled write position is not the complete learned edit. The proportion
of positions immediately following a valid write whose learned write gate is
at least 0.5 is:

| Seed | Head 0 | Head 1 | Head 2 | Head 3 |
|---:|---:|---:|---:|---:|
| 2309 | 0.9957 | 0.0000 | 0.0000 | 0.0153 |
| 2311 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| 2333 | 0.9999 | 0.9995 | 0.0000 | 0.0000 |

The corresponding positive rates over all filler positions are only about
`0.0076` in the active heads, matching the sparse one-step-after-write
locations. Thus most apparent write false positives are a structured causal
tail, not random filler firing. The model learned a two-step write microprogram
that the frozen token-level F1 metric incorrectly counted as uniformly bad.

This does not rescue G15B: overwrite accuracy and collision erase still fail,
and the original promotion remains blocked. It does change the repair target.

## Revised learning problem

Three mechanisms must be separated:

1. **Event recognition is locally observable.** The value token is a valid
   write exactly when `WRITE` or `SELECT` occurred two positions earlier.
2. **Collision recognition is not locally observable.** Whether that key was
   written much earlier cannot be inferred from the same width-four window.
3. **Write execution is temporally extended.** The learned system uses the
   next token in selected heads to finish its address/value code.

The correct local last-write-wins candidate is therefore

\[
M_t=(I-e_t k_tk_t^\top)r_tM_{t-1}+k_t(w_tv_t)^\top,
\]

where `e_t` is anchored to every observable write event, while `w_t` is allowed
to form a short learned write window. Erase and write remain independent, but
the erase target is valid-write timing rather than collision history. This is
closer to a GDN2/KDA-style decoupled correction than to a tied scalar delta
rule.

Before training it, the retained checkpoints should receive one final paired
intervention: preserve learned writes including their tail, and replace only
erase with an all-valid-write event gate. If that cannot improve overwrite
without damaging unique-key recall, investigate prototype orthogonality and
head routing rather than train.

That subsequent G15B-R1 intervention has now completed and failed. Both
erase-at-every-write arms reduce overwrite and every other non-needle mean;
learned-key prototype overlap is high. Event-erase training is not authorized.
See [`G15BR1_EVENT_ERASE_RESULTS.md`](G15BR1_EVENT_ERASE_RESULTS.md).

## Claim boundary

This is a completed exact-SM75, three-checkpoint, zero-update causal diagnostic.
It rejects the naive tied delta repair for the retained learned representation
and identifies a structured write continuation hidden inside G15B's aggregate
write-F1 failure. It does not establish that fresh delta training must fail,
that an event-anchored independent erase will pass, that ordinary next-token
learning will discover the controller, or that any Spin/natural-text/scaling
promotion is warranted. G15A-S and G15B retain their original separate claim
boundaries.
