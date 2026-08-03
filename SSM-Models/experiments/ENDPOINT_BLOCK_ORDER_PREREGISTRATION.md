# Endpoint blocked-order causal control: prospective contract

Date fixed: 2026-08-03, after the monotonic curriculum, shuffled-mixture, and
fixed-L16 controls, and before any scrambled-block result.

## Question

The existing controls establish that the tested monotonic short-to-long
curriculum succeeds while fixed L16 and a fully shuffled mixture of the same
batches fail. They do **not** distinguish monotonic continuation from the
weaker hypothesis that exposing one length in a clean block at a time is
sufficient in any order.

## Frozen intervention

Reuse the exact five generated curriculum blocks, seeds, batch contents,
endpoint labels, optimizer, initialization, compiler, trigger, and 2,000-step
budget. Change only their block order:

`L8 (250) -> L1 (250) -> L16 (1000) -> L2 (250) -> L4 (250)`.

This is a deliberately non-monotonic permutation of the original
`L1 -> L2 -> L4 -> L8 -> L16` stages. It contains the identical multiset of
512,000 endpoint-labeled examples and preserves each fixed-length block.

Run seed 0 first as a prospective causal diagnostic. Do not tune the block
order, thresholds, optimizer, compiler, or budget after observing it.

## Frozen interpretation

- **Pass:** a faithful representation triggers by step 1,500 and passes the
  same compiler, dense L16--L256, L4096, and L16384 gates. Clean blocked
  exposure is sufficient in this seed; strict short-to-long monotonicity is
  not necessary.
- **Mechanism fail with behavioral learning:** no representation trigger, but
  endpoint behavior improves. Block separation alone is insufficient, and the
  evidence favors a continuation-path explanation over mere anti-mixing.
- **Optimization fail:** neither representation nor behavior emerges. The
  non-monotonic block order moves the run into a failed basin but does not by
  itself identify which transition caused it.

A one-seed result is a causal diagnostic for this fixed initialization, not a
reliability estimate. A pass or failure will not be promoted to a universal
claim over optimizers, learning rates, block permutations, or training budgets.
