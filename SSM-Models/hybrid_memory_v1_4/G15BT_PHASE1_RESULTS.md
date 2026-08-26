# G15B-T Phase-1 transactional-controller results

**Protocol:**
[`G15BT_TRANSACTIONAL_DELTA_PROTOCOL_2026-08-26.md`](G15BT_TRANSACTIONAL_DELTA_PROTOCOL_2026-08-26.md)

**Current status:** exact clean-SM75 smoke passed execution; the frozen quality
cohort is pending. The smoke is deliberately non-evidentiary for learning and
cannot promote any arm.

## Clean SM75 execution smoke

The sealed runner executed from clean commit
`b3fd2971090d9e755966e5d0dc73091a18813734` on an RTX 2070 SUPER with exact
compute capability 7.5, Python 3.11.16, PyTorch 2.9.0+cu128, and CUDA 12.8.
The artifact is
[`artifacts/g15bt_phase1_smoke_sm75_2026-08-26.json`](artifacts/g15bt_phase1_smoke_sm75_2026-08-26.json),
SHA-256
`f306e65b85ee3ab1a9b1ad8b8fce56e884227435a7e64f738dffb614f102673d`.

All preflight contracts passed. `F`, `T`, and `T-AUX` have identical initial
parameter hashes, 67,033 total/active parameters, and 4,864 bytes of FP32
streaming state per sequence. The commit projection is assigned to the
scalar-second-moment memory-control group. All declared gradients are finite
and nonzero, and independent reconstructed-forward logits are exact.

Each arm completed four deliberately tiny updates and 3,840 tokens. Standard
evaluation schedules are hash-identical across arms. `T` and `T-AUX` also
share the same intervention schedule. Their strict-history boundary audits
pass:

- current-token mutations leave edit controls, transition, and injection
  bit-identical;
- prior-history effects are nonzero (`0.19660` and `0.19670`);
- chunked and masked-step maximum logit residuals are `2.384e-7` and
  `3.576e-7`, with exact predictions;
- compacting arbitrary valid-mask holes gives exactly equal logits, memory,
  and convolution cache;
- commit-tail support includes filler, write-marker, and query-marker roles.

`F` is expected to fail the strict-history causal audit because its edit
controller intentionally consumes the full current-token view. This is not a
smoke failure and is not used as a T promotion gate.

## Interpretation boundary

The smoke adjudicator returns `passed=false` and
`eligible_for_promotion=false` by construction. Four updates produced no
meaningful retrieval accuracy, as expected. The artifact proves that the
frozen optimizer, losses, paired schedules, interventions, checkpoints, and
SM75 execution path run without fallback. It does not establish learned
commit timing, overwrite, natural-text quality, scaling, tokenizer or
optimizer superiority, efficiency, or a Spin advantage.

Only the three-seed, 3,400-update-per-arm quality cohort can adjudicate Phase
1. `T-AUX` receives the same absolute and causal gate vector as `T`, but
remains diagnostic-only and can never promote the architecture.
