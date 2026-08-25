# State-space model status

**Research author:** Hayden Austin
**Last reconciled:** 2026-08-25

This is the maintained model inventory for `AI_Culture_Mind`. It distinguishes
an implementation from a trained checkpoint, a bounded experiment, a
maintained model, and a promoted model-quality claim. Dated reports remain the
controlling evidence for their exact numbers.

## Status vocabulary

- **Maintained:** current contract, tests, trainer/checkpoint path, and named
  evidence are retained.
- **Experimental:** executable research implementation without a promoted
  general model claim.
- **Historical:** preserved lineage; not the current implementation frontier.
- **Mechanism result:** demonstrates a specified operation or task under a
  bounded protocol; it is not general language quality.
- **Quality result:** completed predictive comparison under the reported data,
  seeds, budget, and matching contract.

## Current inventory

| Family | Status | Checkpoint / training evidence | Current conclusion |
|---|---|---|---|
| [Pure Rotor v2.1](pure_rotor_ssm/CONTRACT.md) | Maintained canonical `Cl(3,0)` family | Versioned checkpoint and five-seed [transport ablation](experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md) | Rotor actions are causally active and improve state-matched prediction over identity, but lose the registered memory and measured-compute gates; quaternion and fixed complex phases are stronger simple structured controls in important views |
| [Pure Spin(8) v1.1](pure_spin8_ssm/CONTRACT.md) | Maintained task-bounded Spin(8) family | Frozen supplied, latent-token, continuous-observation, endpoint-only, calibration, and compiler cohorts | Strong exact/synthetic center-sensitive tracking and shared-action transfer; no natural-language, generic retrieval, or matched modern-SSM superiority |
| [Pure Spin v1.2](pure_spin_ssm_v1_2/README.md) | Implemented frontier model with current CUDA backend | Current tests/backends plus [frontier training result](pure_spin_ssm_v1_2/FRONTIER_TRAINING_RESULTS.md) | Fused Mamba-2 wins all three matched Tiny Shakespeare quality seeds, 2.4942 versus 2.7477 mean bpb; low-level throughput ordering is unresolved at observed repeatability |
| [Pure Exceptional Delta v1.3](pure_f4_delta_ssm_v1_3/README.md) | Experimental Albert/F4/E6 model | Algebra/model suite and small natural-text development cohorts | The exceptional hierarchy is executable, but the fresh [layer-localization result](pure_f4_delta_ssm_v1_3/SHAKESPEARE_LAYER_LOCALIZATION_RESULTS.md) rejects early E6 transport; identity is the supported natural-text reference |
| [Hybrid Memory v1.4/v1.4.5](hybrid_memory_v1_4/README.md) | Active research workspace; no root-level promotion | G-series preregistrations, results, and artifacts inside the active directory | Validated small hybrid causal learner and commissioned synthetic memory; G13 rejects long-context archive promotion; G14 supports decoupled erase/write only on a constructed mechanism task; G15A supports supplied-coordinate shared vector/positive-Spin coupling, G15A-L rejects cosine-map autonomous attribution, and G15A-F restores comparator separation but fails absolute controller-quality gates; no full triality, generic association, natural text, or scaling promotion |
| [Dense SO(8) Cayley scan](pure_rotor_ssm/dense_so8_cayley_scan.py) | Experimental control | Structural tests and one CUDA feasibility smoke in the [design report](experiments/DENSE_SO8_CAYLEY_SCAN_DESIGN.md) | Exact 28-direction chart and bounded scan are implemented; no training, quality, or comparative speed claim |
| [SpinorDeltaLM](../Spin8-SSM-Benchmark/README.md) | Historical isolated benchmark model | Completed short benchmark artifacts under its own directory | Useful controlled historical evidence; not the maintained Pure Rotor/Pure Spin successor |
| [SpinorModel](../SpinorModel/README.md) | Historical prototype | Original tensor-GA baseline and separate overhaul | Implementation provenance only; commands reproduce the historical model, not the current frontier |

## Memory-specific frontier

The repository's strongest general memory evidence currently favors explicit
content addressing and edit laws over richer transport geometry alone.

- Hierarchical selected-block and gathered-memory studies establish bounded
  memory-core quality and systems results, but controller observability and
  end-to-end model integration remain limiting factors.
- Hybrid v1.4.5's default successful small-model path is gated-delta memory
  followed by attention, not the older 24-scalar structured Spin(8) value
  cache.
- G14 shows that independently controlled erase and write can represent a
  constructed accumulation law that tied GDN-v1 cannot. It does not establish
  natural-text or long-context superiority.
- The G15 Spin-Dirac candidate is better posed than the old structured cache:
  it stores content-addressed `8 x 8` fast weights and restricts Spin geometry
  to transport and a fixed Clifford read. Identity, commuting `SO(2)^4`, the
  exact constrained `SU(3)` rank-two torus, full Spin(8), and a broken-coupling
  control are implemented. The primary head-scalar edit law is exactly inner-
  conjugation covariant; channelwise gating is retained only as a named non-
  equivariant ablation. The pre-training integrity artifact passes. In G15A's
  exact SM75/FP32 three-seed primary cohort, full Spin `S` reaches 1.0 symmetry
  macro versus 0.2 for commuting `C` and 0.1 for both `I` and `I+C` in every
  seed; every arm also learns the no-sym delayed-recall control at 1.0 through
  L1024. In the completed prospectively frozen controls, `S+identity-read` ties
  `S` at 1.0 symmetry macro in all seeds, so a fixed Clifford/negative-spin
  read contribution is not supported. `S-broken` scores 0.3/0.2/0.2 versus `S`
  at 1.0, passing the shared-coupling control with 0.7/0.8/0.8 margins; both
  conditional arms retain 1.0 no-sym recall through L1024. Because the task
  supplies exact coordinates and oracle carrier controls, the strongest result
  is shared vector/positive-Spin lift on this designed task, not full three-
  carrier triality, generic association, natural text, or scaling.
- G15A-L is a failed clean-SM75 cohort from commit `0c49f64` and fresh seeds
  2153/2161/2179. `S` mean cosine spans 0.9902--0.9957 at L64,
  0.9803--0.9913 at L256, and 0.9726--0.9862 at L1024. Only seed 2161 L64
  meets the absolute `S` thresholds, and no row meets the 0.05 comparator gate.
  Learned `S-broken` matches `S` within about `6e-8` across all rows; its
  effective positive chart matches within `<4e-7`. Analytically, cosine cancels
  positive scalar `<q,Vk>`, making the vector carrier unobservable and the
  broken signed permutation invertible for `P`. Autonomous attribution fails
  under that observation map, motivating G15A-F.
- G15A-F runs the frozen full-frame repair on clean commit `503fa82`, exact SM75
  seeds 2203/2207/2213. All rank-56 screens pass, with condition ratios
  0.286--0.310, minimum primitive projection residual about 0.598, and 474/784
  exact broken Lie-bracket mismatches at maximum integer residual 4. The quality
  gate fails: `S` mean relative Frobenius errors are 0.0927/0.0912/0.0705 at
  L64, 0.1201/0.1187/0.1036 at L256, and 0.1304/0.1433/0.1392 at L1024; p95
  fails every row and maximum-error gates fail cohort-wide. `S` beats `I`, `C`,
  and `S-broken` by at least 0.05 in every row, so the repaired observation
  separates the shared lift, but controller precision remains unsolved; the
  broken-2x check passes only two rows. Diagnose chart leakage/convergence,
  then prospectively freeze a balanced-primitive curriculum/block-scalar
  optimizer before G15B or natural text. No promotion follows.

The current machine-readable G15-series evidence is the
[primary cohort artifact](hybrid_memory_v1_4/artifacts/g15a_spin_dirac_cohort_sm75_2026-08-25.json),
read together with the frozen
[G15 preregistration/result ledger](hybrid_memory_v1_4/G15_SPIN_DIRAC_RESULTS.md),
the prospectively frozen
[conditional attribution protocol](hybrid_memory_v1_4/G15A_CONDITIONAL_CONTROLS_PROTOCOL_2026-08-25.md)
with its completed
[conditional-control artifact](hybrid_memory_v1_4/artifacts/g15a_conditional_controls_sm75_2026-08-25.json),
the prospectively frozen
[G15A-L protocol](hybrid_memory_v1_4/G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md),
and its failed
[learned-coordinate artifact](hybrid_memory_v1_4/artifacts/g15al_learned_coordinate_cohort_sm75_2026-08-25.json).
The next evidence is the prospectively frozen
[G15A-F protocol](hybrid_memory_v1_4/G15AF_FULL_FRAME_PROTOCOL_2026-08-25.md)
and its failed
[full-frame artifact](hybrid_memory_v1_4/artifacts/g15af_full_frame_cohort_sm75_2026-08-25.json).
Status is not inferred from the presence of passing algebraic unit tests.
Local baseline eligibility is independently controlled by
[SM75_NATIVE_RUNTIME.md](hybrid_memory_v1_4/SM75_NATIVE_RUNTIME.md); native
execution does not itself promote model quality.

See [Programme 03](../research-programs/03-structured-memory-and-retrieval/README.md)
for retrieval/update-law claims and
[Programme 06](../research-programs/06-rotor-noncommutative-state-space-models/README.md)
for trained noncommutative-model claims.

## Strongest established negative results

These are first-class current evidence:

1. Pure Rotor does not win the registered generic associative-recall or
   measured-compute gates.
2. Pure Spin v1.2 loses the matched three-seed Tiny Shakespeare quality gate to
   fused Mamba-2.
3. Dense F4/E6 transport is not supported as the generic natural-text default;
   the fresh localization gate favors identity.
4. Triality does not add ordinary same-state overwrite capacity when routing
   and transport are matched.
5. Vector-only Spin(8) observations cannot recover the hidden lift under the
   documented balanced collision; the Bayes boundary remains attached to the
   claim.
6. Hybrid v1.4.5 has not demonstrated robust factual recall beyond its actual
   attention window under ordinary pretraining.

## Promotion requirements

A new model becomes maintained or gains a broader quality claim only when its
controlling documents provide:

- exact recurrent/parallel/chunk/token semantics where claimed;
- explicit state bytes and used/trainable parameter counts;
- gradient, masking, initialization, optimizer, and checkpoint contracts;
- at least the preregistered independent seeds and capability gates;
- state-, parameter-, token-, and measured-compute matching labelled
  separately;
- actual upstream backend identity, including fused versus fallback paths;
- raw structured artifacts, source hashes, data hashes, and negative controls;
- a nonclaim section preventing synthetic, algebraic, or isolated-kernel
  evidence from becoming a general model claim.

## Historical evidence policy

Files under `experiments`, model-local result directories, and
`Spin-Space-Research/docs` may describe an earlier frontier. They remain valid
records of their own protocols. This page supplies current model status; it
does not alter their observations or hashes.
