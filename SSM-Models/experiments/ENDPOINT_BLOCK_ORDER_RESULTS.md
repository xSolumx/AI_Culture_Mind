# Endpoint blocked-order causal control: results

Date completed: 2026-08-03.

## Outcome

The prospectively frozen `L8 -> L1 -> L16 -> L2 -> L4` seed-0 control does not
discover a faithful representation by the step-1500 deadline. It is classified
as **mechanism failure with behavioral learning**, not total optimization
failure.

The schedule contains the exact same 2,000 generated batches and 512,000
endpoint labels as the successful short-to-long curriculum; only the five
blocks are permuted.

| Boundary | Length | Task loss | Batch accuracy | Representation |
|---:|---:|---:|---:|---:|
| 250 | 8 | 2.946 | 29.7% | none |
| 300 | 1 | 0.0435 | 100% | none |
| 500 | 1 | 0.0018 | 100% | none |
| 550 | 16 | 4.006 | 5.1% | none |
| 1,500 | 16 | 3.799 | 7.4% | none |
| 1,750 | 2 | 0.0111 | 100% | none |
| 2,000 | 4 | 0.0250 | 100% | none |

The step-2,000 total loss is `0.1435` because the unretracted algebraic terms
remain active; the endpoint task-loss value in the table is `0.0250`.

## Causal reading

This falsifies the simple hypothesis that any clean separation of lengths is
sufficient. An isolated L1 block can drive its own endpoint accuracy to 100%
without organizing the token actions into a compilable representation. The
immediate jump from L1 to already-mixed L16 then returns to near chance for the
entire 1,000-step block. Later L2 and L4 blocks again fit behaviorally, but do
not retroactively create the missing faithful action family.

Together with the successful monotonic curriculum, fully shuffled mixture,
and fixed-L16 controls, the best supported mechanism is **incremental depth
continuation**: high-information short words must not merely occur in a block;
their solution must be transported through intermediate composition depths
before the random walk has mixed.

This remains a one-seed, one-permutation causal diagnostic. It does not prove
that every non-monotonic block permutation fails, nor that another optimizer,
learning rate, or much larger budget cannot find the same basin.

Artifacts:

- `endpoint_scrambled_blocks_seed0.json`
- `endpoint_scrambled_blocks_seed0_checkpoints/uncompiled_retraction_seed0.pt`
- `ENDPOINT_BLOCK_ORDER_PREREGISTRATION.md`
