# SpinorModel overhaul validation

Date: 2026-08-03.

The overhaul is additive. No historical file in the parent `SpinorModel`
directory was modified.

Raw diagnostic: `diagnostics_rtx2070.json`.
SHA-256: `22ec7b69635922a672f5a39681152fad17ec6903ce9e6feeaca47de1ba80109d`.

## Outcome

The new model closes the original implementation's main systems gap: it
exposes one persistent Cl(3, 0) state per layer/channel end to end, trains with
an associative rotor-affine prefix scan, and generates by priming the prompt
once and then reusing a constant-size recurrent cache.

Ten new load-bearing tests pass:

- basis multiplication and reversion;
- nonzero rotor tangent gradient at identity;
- affine-transition associativity and application parity;
- parallel scan/recurrent parity;
- full/chunk/token language-model parity;
- padding as an identity transition and causal outputs;
- exact initialized half-life schedule;
- numerical BIBO bound over all prefixes of a 128-step trajectory;
- prompt-once recurrent generation;
- exact checkpoint round trip.

The untouched original suite continues to pass its three tests.

## Mathematical core

Each recurrent step is

```text
h_t = a_t Ad(r_t, h_(t-1)) + (1-a_t) w_t u_t,
0 < a_t < 1,  0 < w_t < 1.
```

`Ad(r, h) = r h reverse(r)` is isometric within each grade. Therefore, for a
zero initial state and bounded effective candidates `w_t u_t`, repeated use of
the triangle inequality gives

```text
norm(h_t) <= max_(s<=t) norm(w_s u_s).
```

The test evaluates this bound at every prefix rather than checking only a
final state. Token-selective step size controls retention/erasure while leaving
the requested log-spaced initial half-lives exact. The rotor controller starts
at identity and uses an analytic sinc tangent, so it receives gradients there.

Transition composition is closed and chronological:

```text
(a2,r2,b2) o (a1,r1,b1)
  = (a2*a1, r2*r1, a2*Ad(r2,b1)+b2).
```

The parallel and recurrent paths implement this same operator.

## CUDA diagnostic

Hardware: NVIDIA GeForce RTX 2070 SUPER. Dtype: float32. Seed: 0.

| Measurement | Result |
|---|---:|
| L2048 parallel/recurrent maximum logit error | 1.83e-7 |
| L2048 parallel/recurrent maximum state error | 1.62e-7 |
| original parameters, vocab 512 / width 8 / 2 layers / 2 heads | 9,056 |
| width-matched overhaul parameters, vocab 512 / 1 channel / 2 layers | 4,737 |
| width-matched recurrent cache | 16 scalars |
| research-default parameters, vocab 512 / 8 channels / 4 layers | 46,984 |
| research-default recurrent cache | 256 scalars |
| original reference throughput, B2 x L512 | 82,052 tokens/s |
| overhaul reference parallel throughput, B2 x L512 | 34,388 tokens/s |

Throughput is the median-free result of one synchronized five-repeat run after
two warmups, so it is a diagnostic rather than a stable benchmark estimate.
The conclusion is nevertheless unambiguous: the Python/Torch
Hillis--Steele reference does not beat the original attention implementation
on this hardware. It has logarithmic dependency depth but `O(L log L)` work
and many unfused geometric products. A work-efficient fused scan is required
before any speed claim.

## End-to-end smoke

A one-epoch CUDA run on the parent `corpus.txt` completed training, exact
label-weighted validation, checkpoint serialization, strict reload, prompt
priming, and recurrent generation. Its validation loss is not reported as a
quality result: one epoch on this tiny corpus is only an execution test.

## Claim boundary

Established:

- persistent per-layer recurrent state with context-independent size;
- exact prompt-once generation rather than context recomputation;
- parallel/recurrent/chunk equivalence;
- stable identity initialization and requested half-life initialization;
- a rigorous bounded-state update law;
- self-contained training, evaluation, checkpoint, and diagnostic tools.

Not established:

- quality or sample-efficiency improvement over the original model;
- an advantage attributable specifically to geometric algebra;
- production throughput or memory advantage during parallel training;
- compatibility with an unseen historical 44.8 MB checkpoint;
- scaling behavior on a modern tokenizer or language corpus;
- superiority over delta-rule, attention, Householder, complex, or ordinary
  dense selective SSM baselines.

## Next gates

1. Add parameter- and state-matched scalar, complex, Householder, and
   identity-rotation rows behind the same model/training API.
2. Replace the reference scan with a fused work-efficient CUDA/Triton operator
   and remeasure forward, backward, and streaming latency.
3. Train on a real tokenizer/corpus split with fixed seeds, exact token-weighted
   metrics, dense context sweeps, and checkpoint-resume determinism.
4. Measure learned rotor angles, retention, write strength, state norms, and
   gradient trajectories; require the geometric mechanism to be used.
5. Advance to larger language runs only if rotor ablations beat matched
   non-geometric controls on quality, extrapolation, or state efficiency.
