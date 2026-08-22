# Spin-Delta overwrite capability gate

**Frozen:** 2026-08-22 before outcome training on seeds 401/409/419.

## Question

The Shakespeare gate rejected Spin-Delta as a language-model successor. This
separate experiment asks whether the mechanism itself learns the operation it
was built to express: addressable overwrite and retrieval. It cannot rescue
the failed language result.

Prior Programme 03 results establish the control logic. Delta updates have
exact overwrite capacity under orthogonal keys, while learned continuous keys
can lose retrieval through cross-key interference. Spin-Delta has exactly two
slots, so this task uses exactly two semantic keys. Failure cannot be assigned
to an over-capacity key count.

## Frozen task

Each input episode is

```text
[WRITE, key, value] repeated W times, then [QUERY, key].
```

Keys are binary and values are uniformly sampled from 32 classes. The first
two writes cover both keys in random order; later writes randomly overwrite
either key. The target is the latest value written to the queried key. Chance
accuracy is 3.125%.

- train at `W=8`;
- evaluate at `W=8`, `16`, and `32` with frozen independent batches;
- loss is cross-entropy only at the final query position;
- the target is never appended to the input;
- local convolution has width four, so extrapolated queries require recurrent
  information rather than direct local access to most writes.

## Matched models and budget

- maintained independent v1.2 with `raw_cuda_hybrid`;
- Spin-Delta with `raw_cuda_delta`;
- `d_model=64`, two layers, two independent transport heads;
- Spin(8) in both layers, direction readout, SwiGLU;
- 800 AdamW steps, batch 128, learning rate `3e-3`, weight decay `0.01`;
- clip norm `1.0` and 32 evaluation batches per length;
- seeds 401, 409, and 419;
- identical initial common parameters and generated training/evaluation batches
  within each pair;
- maximum initial-logit residual `2e-6`.

## Frozen decisions

Candidate capability passes only if every seed reaches:

- at least 90% accuracy at the trained length `W=8`;
- at least 75% accuracy at extrapolated length `W=16`.

Differential advantage additionally requires:

- Spin-Delta exceeds maintained v1.2 by at least five percentage points at
  `W=16` in at least two seeds;
- mean `W=16` improvement is at least five percentage points;
- all artifacts are finite, compatible, and within the pairing bound.

`W=32` is a reported stress frontier, not a pass threshold. Capability without
differential advantage means the mechanism works but is unnecessary given the
maintained recurrent state. Failure of candidate capability closes this smooth
two-slot construction on the named overwrite task, not delta memory generally.
