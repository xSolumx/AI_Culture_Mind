# Hybrid Memory v1.4 evidence protocol

**Opened:** 2026-08-24
**Current status:** pre-quality, post-falsification architecture repair.

## Provenance correction

The first version of this document described a prototype that was later found
to contain invalid tasks and state surfaces and was comprehensively rewritten
before any quality run. It therefore cannot honestly be described as a fully
frozen protocol for the rebuilt implementation.

Its most important criterion is preserved exactly in substance:

> Original G3 required a one-block model under final-only loss to give nonzero
> query-controller gradients at non-final positions, with norm greater than
> `1e-9` at zero-initialized controllers.

That gate is **falsified**. One-block reads affect contemporaneous output, not
future transitions; zeroing every controller also leaves zero content to
address. Hard coarse routing added an independent failure because `argmax`
disconnects its logits. No G4 quality run was started before this result.

## Original gate ledger

| Gate | Original criterion | Current disposition |
|---|---|---|
| G1 | selected-block associativity and recurrent/parallel output and gradient parity | structural coverage passes, but the originally listed configuration matrix has not been replayed as one frozen artifact |
| G2 | bounded selected state for 4,096 steps | implemented and tested mechanically |
| G3 | non-final one-block query credit from final-only loss | **failed structurally** |
| G4 | three-seed, 600-update MQAR capability above 90%, untrained below 15% | not run |
| G5 | every screen paired with ProductKey and gated-delta rows | invalid as written: static ProductKey is non-episodic and FLA DeltaRule entries are operators, not complete LMs |
| G6 | complete state accounting, including attention KV and convolution caches | implemented and tested mechanically |
| G7 | fp16/fp32 error growth through length 65,536 against direct fp64 source | completed for generic high-retention, selected-block, and maintained DeltaProduct paths; see the dated precision artifact |
| G8 | trained structured-tier chart-switch and orthogonality audit | pending because no checkpoint exists |

## Post-falsification engineering acceptance

These checks were fixed after observing the original failure. They are
engineering acceptance criteria, not untouched prospective scientific
evidence.

Run `temporal_observability_screen.py` in float64 with seed `42`, tokens
`[1,2,3,4,5,6]`, and per-position activation gradients rather than a shared
parameter norm.

- T1, hard one block: coarse route tensors are disconnected and non-final
  fine-read norm is exactly zero. This negative control must remain visible.
- T2, straight-through one block: coarse route tensors are connected, while
  non-final read norm remains exactly zero. This distinguishes route
  differentiability from temporal observability.
- T3, straight-through `selected_block -> attention`: coarse tensors are
  connected and both coarse and fine non-final read norms are strictly
  positive. This establishes a causal path only.
- T4, near-zero readout conditioning: zero-state Jacobians remain finite with
  norm below `10` for selected memory and below `20` for structured memory in
  the fixed unit tests. The readout uses `x / sqrt(1 + ||x||^2)` plus log
  energy; dtype-dependent normalization by machine epsilon is forbidden.

## Prospective G4a: retrieval capability and matched control

This section is frozen before any 600-update v1.4 quality run.

### Common configuration

- task: causal MQAR;
- training length: `512`; evaluation lengths: `512, 2048, 8192`;
- updates: `600`; model seeds: `41, 43, 47`;
- training batch `8`; evaluation batch `8`; `8` evaluation batches per
  length and seed;
- learning rate `3e-3`; weight decay `0.01`;
- fresh deterministic episode on every step;
- AdamW, identical optimizer settings, presented tokens, seed order, and
  evaluation cohorts;
- chunk size `128`; all long evaluation carries complete model state;
- parameter gap threshold: `5%` relative to the candidate;
- every artifact records pre-training and post-training metrics, actual and
  capacity cache bytes, source SHA-256 values, Git commit plus dirty/untracked
  status, environment, wall time, and an evidentiary flag.

Candidate configuration:

```text
model_dim=64
layer_plan=(selected_block, attention)
attention_heads=4, attention_window_size=128
selected_heads=2, blocks=4, slots_per_block=4, value_dim=16
selected_update_rank=1, selected_controller_rank=None
conv_kernel=3, expansion=2, dropout=0
```

Matched common-shell control:

```text
same configuration except layer_plan=(delta_product, delta_product)
delta_heads=4, delta_num_householder=1
```

The current parameter counts are `107,552` candidate versus `112,290`
control, a `4.4053%` gap.

### Routing cohorts

- Q1 label-free estimator: `selected_training_route_mode=straight_through`,
  routing auxiliary coefficient `0.0`.
- Q2 explicitly supervised ablation: identical protocol with routing
  auxiliary coefficient `0.1`. This cohort is labeled supervised routing and
  cannot support a label-free claim.

### Frozen decisions

- Capability passes only if every candidate seed has pre-training accuracy
  below `15%` and post-training exact per-query accuracy at length 512 above
  `90%`.
- Failure of Q1 is a failure of the label-free route, even if Q2 succeeds.
- Length-2,048 and length-8,192 results are reported without moving thresholds
  after inspection.
- A candidate/control quality claim requires the candidate to avoid a paired
  regression larger than two percentage points at every reported length and
  to improve the three-seed mean at length 2,048. Otherwise the result is
  capability-only or negative.
- Parameter-gate failure invalidates the cohort before training.

## Baseline amendment

The mandatory matched row is the DeltaProduct common-shell control above.
Static ProductKey memory remains a useful capacity/static-retrieval reference
but is excluded from episodic MQAR promotion. FLA DeltaRule adapters remain
operator semantic/systems controls. Official Mamba-2 may be reported only as a
separate complete-model cohort when its exact availability probe succeeds; an
unavailable dependency produces an explicit skip and never a fallback.

## Deferred

- Natural-language training and any bits-per-byte claim.
- Fused-kernel or Tensor-Core speed claims.
- State-size quality sweeps until G4a establishes learnability.
- Learned Spin rung use until a trained checkpoint exists and a separate
  held-out protocol is frozen.
