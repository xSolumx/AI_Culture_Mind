# Hybrid Memory SSM v1.4

**Status:** structurally tested research prototype with a negative frozen G4a
retrieval result. It is not a released successor to v1.2/v1.3, has no trained
checkpoint, and has no quality or speed promotion.

This track combines four explicit mixer kinds in one causal language-model
shell:

- bounded sliding-window RoPE attention with a complete streaming KV cache;
- the maintained repository DeltaProduct reference;
- hierarchical selected-block affine memory;
- bounded rung-routed Spin(8) memory.

Layer schedules are explicit `layer_plan` tuples. Full-sequence, arbitrary
chunk, and token-step execution carry complete recurrent, convolution, and
attention state. `state_byte_report` reports actual and configured-capacity
bytes rather than an advertised partial cache.

## The important negative result

The original preregistered G3 claim was false. Always-live reads do not make a
non-final read affect a final-only loss in a one-block model. Coarse block
selection was also a hard `argmax`, so retrieval loss could not train its
logits.

The failure is retained as evidence, not reworded as a success. The successor
mechanism adds explicit dense `soft` and `straight_through` route modes while
keeping sparse `physical_gather` hard-only. Deterministic float64 replay now
separates three facts:

1. hard one-block routing has disconnected coarse logits and zero non-final
   read credit;
2. straight-through routing connects coarse logits but cannot change the
   one-block temporal dependency, so non-final read credit remains zero;
3. a `selected_block -> attention` schedule with straight-through training
   has a real non-final coarse and fine read path to final-only loss.

See [`RESULTS.md`](RESULTS.md),
[`PREREGISTRATION.md`](PREREGISTRATION.md), and the checked artifact
[`artifacts/temporal_observability_2026-08-24.json`](artifacts/temporal_observability_2026-08-24.json).

The subsequent three-seed, 600-update G4a MQAR campaign also failed. The
label-free candidate reached only 1.563%--3.125% L512 accuracy per seed; the
explicitly supervised-routing ablation reached 1.172%--2.344%. The matched
DeltaProduct shell also remained near chance, so this falsifies the frozen
capability claim without isolating one memory architecture as the cause. See
[`RESULTS.md`](RESULTS.md) for the complete adjudication.

## Implemented surfaces

- [`tasks.py`](tasks.py): causal MQAR, overwrite retrieval, exact-distance
  needle retrieval, and selective copy. Scored answers are absent from query
  input positions.
- [`selected_block.py`](selected_block.py): bounded affine memory with
  independent write/erase/read routes; sparse hard gather/scatter; dense hard,
  soft, and straight-through semantic modes; recurrent/parallel oracles.
- [`attention.py`](attention.py): bounded causal local attention and cache.
- [`structured_tier.py`](structured_tier.py) and
  [`structured_memory.py`](structured_memory.py): data-routed subgroup ladder
  and a recurrent Spin(8) state that actually changes the model output.
- [`model.py`](model.py): four-kind hybrid shell, local convolution caches,
  checkpointing, diagnostics, and exact cache-byte accounting.
- [`fla_adapter.py`](fla_adapter.py): fail-closed semantic and optional official
  FLA DeltaRule operator adapters.
- [`baselines.py`](baselines.py): precise implementation registry and claim
  boundaries. Static ProductKey memory is explicitly non-episodic; FLA
  adapters are operators, not complete language models; official Mamba-2 is
  used only when its exact availability probe passes.
- [`audits.py`](audits.py): precision horizon, complete chunk replay, cache
  drift, structured rung/gauge, and temporal observability audits.
- [`experiments.py`](experiments.py): fresh deterministic episodes, paired
  cohorts, pre/post-training evaluation, parameter gate, explicit route
  training mode, source hashes, dirty/untracked Git status, state-size sweeps,
  and atomic JSON-compatible schemas.
- [`retrieval_screen.py`](retrieval_screen.py): exact non-evidentiary smoke and
  frozen G4a matched MQAR campaign builders plus gate adjudication.
- [`long_context_screen.py`](long_context_screen.py): non-training streaming
  screen at configurable lengths.
- [`temporal_observability_screen.py`](temporal_observability_screen.py):
  deterministic replay of the failed original G3 and accepted successor
  topology.
- [`precision_screen.py`](precision_screen.py): direct-float64 fp16/fp32 error
  curves through length 65,536 for generic affine, selected-block, and
  maintained DeltaProduct transitions.

## Baseline boundary

The normal matched quality control is a `HybridMemoryLM` with the same shell
and a `delta_product` layer plan. A static ProductKey table is not a fair
episodic MQAR control. The FLA DeltaRule registry entries are operator-level
semantic/systems controls, not silently promoted language models. Official
Mamba-2 is an optional separate complete-model comparison and fails closed
when unavailable.

## Validation

From `SSM-Models`:

```powershell
python -m pytest hybrid_memory_v1_4/tests -q
python -m hybrid_memory_v1_4.temporal_observability_screen --output hybrid_memory_v1_4/artifacts/temporal_observability_2026-08-24.json
python -m hybrid_memory_v1_4.precision_screen --output hybrid_memory_v1_4/artifacts/precision_horizon_65536_cpu_2026-08-24.json
python -m hybrid_memory_v1_4.long_context_screen --device cuda --output hybrid_memory_v1_4/artifacts/mechanical_cuda_smoke_2026-08-24.json
```

The CUDA screen is a random-token mechanical smoke: finiteness, chunk replay,
bounded cache, and raw throughput only. It is explicitly not model-quality or
matched speed evidence.

## Nonclaims

- The frozen G4a retrieval capability claim failed in both routing cohorts.
- The matched control also failed, so G4a does not isolate selected memory as
  the cause.
- No trained checkpoint or successful model-quality result exists.
- Straight-through routing establishes a gradient estimator, not successful
  label-free routing.
- The selected-attention topology result establishes a causal gradient path,
  not aligned descent or improved accuracy.
- Spin(8) structure and learned rung use are hypotheses, not promoted
  advantages.
- Reference Python timings are not fused-kernel comparisons.
