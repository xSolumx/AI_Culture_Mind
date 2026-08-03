# Endpoint-only self-compiling retraction: prospective contract

Date fixed: 2026-08-03, after the CPU endpoint-oracle audit and before any
endpoint-only GPU training result.

## Gate

Remove dense prefix-state supervision from both finite-action compilation and
neural task loss. The environment may return one anonymous label for a complete
word. It never returns a prefix trace to the compiler. During training, the
generated prefix tensor is sliced to its final label and discarded before
device transfer, loss construction, holonomy, or model access.

## Frozen endpoint compiler

For each seed:

1. Query 1,024 independently sampled length-16 words and retain one complete
   word representative for every anonymous endpoint label.
2. Query the empty word once to locate the anonymous identity label.
3. Query `(0,1)`, `(0,2)`, and `(0,3)`. Exactly one returns identity, identifying
   token 0's inverse; the remaining two tokens are forced to pair.
4. For every state representative, query its extension by the lower-indexed
   token in each inverse pair: 60 states times 2 tokens = 120 queries.
5. Infer the 120 reverse transitions, reconstruct the regular action, derive
   its multiplication table, and compile both real 3D candidates.

Budget: 1,024 passive labels + 124 active membership queries = **1,148
endpoint labels**. No prefix label is counted or consumed. This is 7.136 times
fewer compiler labels than the previous two dense 256x16 recovery batches.

The CPU design audit is fixed and already observed:

- 1,000/1,000 passive corpora cover all 60 states; samples-to-coverage range
  132--704, mean 287.688.
- 1,000/1,000 infer `(1,0,3,2)` and reconstruct an exactly isomorphic action.
- 100/100 arbitrary permutations of all 60 endpoint labels remain exact;
  identity appears at 50 distinct anonymous labels.
- deleting one of the 120 extension queries is refused 100/100.

These CPU results selected the fixed 1,024-label passive budget. They are not
GPU training results.

## Frozen neural protocol

- Seeds 0 through 9.
- Four-channel pure Cl(3) rotor recurrence.
- 2,000 deterministic CUDA updates; batch 256; word length 16.
- **Endpoint cross-entropy only:** one supervised label per sequence, 512,000
  labels total instead of 8,192,000 prefix labels.
- The holonomy objective receives only endpoint labels. After exact action
  recovery it may compose those endpoints with the recovered algebra; it never
  reads observed prefix targets.
- Unconstrained ambient token gradients until representation discovery, then
  shared-conjugacy retraction of the entire token family.
- Held-out training bigram unchanged.
- Untouched changed-generator selection index 59:
  `(25341, 51342, 25413, 41532)`, disjoint from training tokens.

## Preregistered gates

Every seed must:

1. cover all 60 passive endpoint labels within 1,024 samples;
2. use exactly 1,148 endpoint labels, infer the correct inverse matching,
   observe 120 transitions, complete 120, and replay the action exactly;
3. reconstruct order 60 and compile with invariance and homomorphism RMS below
   `1e-10`;
4. trigger joint representation-manifold retraction by step 1,500;
5. reach at least 90% at every dense L16--L256 checkpoint on the original and
   untouched index-59 alphabets;
6. reach at least 90% at L4096 on both alphabets and L16384 on index 59;
7. pass a post-hoc 13-point index-59 sweep at every 1,024 tokens from L4096 to
   L16384.

The cohort passes only at 10/10 seeds. Means cannot substitute for the all-seed
count. Exact compiler error, float32 mechanism error, decoded accuracy, and path
drift remain separate metrics.

## Claim boundary

A pass establishes endpoint-membership-query recovery and endpoint-only neural
training for this finite A5 action. The active query protocol deliberately asks
representative extensions and is therefore much stronger than passive random
endpoint supervision. It does not establish unsupervised discovery, arbitrary
word-equivalence induction, noisy endpoints, applicability to language, or the
same query budget for another group or alphabet.
