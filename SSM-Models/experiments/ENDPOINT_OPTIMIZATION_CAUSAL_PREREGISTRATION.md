# Endpoint optimization causal audit: prospective contract

Date fixed: 2026-08-03, after fixed-L16 seeds 0--2 failed and after the
information/gradient audit, but before any endpoint curriculum, shuffled
mixture, or extended-persistence result.

## Hypotheses

The fixed-L16 plateau admits three separable explanations:

1. **First-order mixing:** individual token positions contain almost no final
   label information at L16, so stochastic action gradients cancel.
2. **Short-word exposure:** examples at L1/L2/L4/L8 provide the missing
   first-order signal, regardless of their ordering.
3. **Continuation ordering:** progressively increasing length is necessary;
   merely mixing the same short and long examples is insufficient.

Ordinary vanishing gradients is already disfavored because L16 batch gradient
norms are substantial while their directions are incoherent.

## Frozen arms

All arms use seed 0 first, the same endpoint compiler, endpoint-only task loss,
batch size 256, optimizer, held-out pair, model, holonomy, and retraction gate.

| Arm | Updates | Word schedule | Neural endpoint labels |
|---|---:|---|---:|
| F16-2k | 2,000 | L16 only; already observed | 512,000 |
| CURR | 2,000 | 250xL1, 250xL2, 250xL4, 250xL8, 1,000xL16, ordered | 512,000 |
| MIX | 2,000 | exact same generated batch multiset as CURR, deterministically shuffled | 512,000 |
| F16-8k | 8,000 | L16 only; first 2,000 batches bit-identical to F16-2k | 2,048,000 |

CURR is the already-frozen formal experiment and is run first. MIX and F16-8k
are explanatory controls; they cannot retroactively rescue a failed CURR gate.
The first 2,000 F16-8k batches must reproduce the existing seed-0 trajectory
apart from newly added diagnostic fields. Because persistence is the variable
under test, representation discovery remains read-only and eligible through
step 7,500 rather than being disabled after the original step-1,500 deadline.

## Interpretation table fixed before results

| CURR | MIX | F16-8k | Supported reading |
|---|---|---|---|
| pass | pass | fail | short-word information, not ordering, is sufficient |
| pass | fail | fail | staged continuation is necessary at this budget |
| pass | any | pass | curriculum helps efficiency; fixed length is not impossible |
| fail | pass | any | curriculum ordering is harmful; heterogeneous exposure helps |
| fail | fail | pass | label budget/persistence, not short-word bootstrapping |
| fail | fail | fail | current optimizer remains outside the useful basin |

"Pass" for seed 0 means trigger by step 1,500 for CURR/MIX, or by step 7,500 for
F16-8k, followed by all already-preregistered dense/long mechanism gates. The
ten-seed formal claim continues to belong only to CURR if its seed-0 smoke
passes.

F16-8k stops at exactly 8,000 updates. If no representation has triggered by
step 7,500, it is recorded as a failure of this persistence budget; the run is
not extended until something happens and no larger stopping point is inferred
post hoc.

## Claim boundary

L1 endpoints reveal one-token action outcomes. CURR and MIX are endpoint-only
because no example exposes an internal prefix, but neither is free of
short-transition information. A positive result must be described as
short-to-long endpoint curriculum or mixture learning—not passive fixed-L16
word-equivalence discovery.
