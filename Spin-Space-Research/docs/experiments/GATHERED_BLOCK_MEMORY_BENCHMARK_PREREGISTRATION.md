# Gathered-block memory benchmark preregistration

- **Frozen:** 2026-08-10, before implementation or timing inspection
- **Hardware target:** local RTX 2070 SUPER, Windows PyTorch CUDA
- **Processes:** three independent measurement processes
- **Purpose:** replace ideal selected-byte arithmetic with an actual gathered
  recurrent read/write implementation

## Implementations

The benchmark times routing, one memory write, and one memory read for both
direct overwrite and standard delta updates:

1. `dense_full`: score every fine slot and update the full state;
2. `block_masked_full`: score blocks, score eight gathered fine weights, expand
   the selected local route to the full slot axis, and update the full state;
3. `block_gathered`: score blocks, gather eight fine weights and the selected
   state block, then update and read that block in place.

`block_masked_full` and `block_gathered` must receive identical selected blocks
and local routes. Their resulting full states and predictions must agree before
timing is interpreted. The in-place gathered row is an inference recurrence;
no backward or training-through-mutation claim is made.

## Frozen grid

- logical slots: `64`, `256`, `1024`, `4096`;
- slots per block: `8`;
- alias dimension: `8`;
- value dimension: `8`;
- batches: `1`, `16`, `64`;
- dtype: `float32`;
- warmup calls: `50`;
- timing blocks: `25`;
- inner calls per block: `100`;
- cyclic implementation order across timing blocks;
- raw block samples retained in every artifact.

Shared-router and three-independent-router parameter bytes are reported
separately. The principal latency comparison uses the shared router so it does
not conflate gathering with router-table size.

## Frozen correctness and systems decisions

Each artifact must satisfy:

- masked-full versus gathered prediction error `<= 1e-5`;
- masked-full versus gathered final-state error `<= 1e-5`;
- all timings and CUDA memory measurements finite;
- identical logical state scalar counts within each slot/batch cell.

The gathered systems advantage is supported only if the median of the three
process medians is faster than `block_masked_full` for both update laws at
batch `16` and logical slot counts `1024` and `4096`. Peak incremental CUDA
allocation is reported independently and is not substituted for latency.

## Nonclaims

This benchmark is hardware- and implementation-specific. It does not establish
a fused training kernel, model-level quality, cross-hardware thresholds, or a
triality-specific capacity advantage.

