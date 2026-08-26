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
| [Hybrid Memory v1.4/v1.4.5](hybrid_memory_v1_4/README.md) | Active research workspace; no root-level promotion | G-series preregistrations, results, and artifacts inside the active directory | Validated small hybrid causal learner and commissioned synthetic memory; G13 rejects long-context archive promotion; G14 is a constructed mechanism task; after the distinct G15A-L/F observation/precision failures and G15A-R repair, G15A-S confirms composition-only signed-dictionary transfer under oracle edit timing. G15B rejects the token/local-convolution commissioned controller; R0 rejects tied delta, while R1/R2 reject scalar symmetric erase. R3's oracle component reset repairs ordinary overwrite by 12.2--12.8 points and reaches 1.0 on its guard, but misses the frozen saturated-baseline improvement and FP32 replay-tolerance gates. R4 then shows that only ambiguous value-plus-tail ownership passes; value-only arms fail and background removal does not harm the tail arms. R5 localizes the useful source to strict history conditional on the full-token transition: its background-free arm passes all performance/bias gates, but frozen replay/runtime numeric gates fail. R5-S then fails all 135 prospective scaled-logit checks despite exact categorical behavior and passing state/read, BPQ, transition, fingerprint, and FP64 contracts. Retained-checkpoint repair is closed; identity transport remains the generic default, with the next fresh model requiring explicit pending-write/commit semantics before any Spin comparison |
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
  broken-2x check passes only two rows. This motivates G15A-R.
- G15A-R passes exact SM75 development and confirmation on clean commit
  `eca70f0`; artifact SHA-256 begins `be004dea`. All five development recipes
  qualify. The 600-step fixed-LR/random control reaches mean error
  0.0155--0.0285, so decay is not proven necessary. Frozen least-intervention
  selection chooses `G-decay/random`, reaching about `1.03e-7`--`1.85e-7`
  development means; staged decay is vastly more precise, while block moments
  and curriculum are unnecessary. Fresh seeds 2251/2267/2273 pass every
  `I/C/S/S-broken` gate at L64/L256/L1024. `S` mean error is
  `1.21e-7`--`1.89e-7`, p95 `<=2.68e-7`, maximum `<=3.27e-7`; margins are
  0.234--0.355 versus `I`, 0.231--0.351 versus `C`, and 0.153--0.234 versus
  broken, with broken-2x passing every row. This establishes composition-only
  minimal-controller shared vector/positive-Spin chart learning under the
  four-probe oracle-frame objective and oracle edit timing. G15B learned
  address/query generic association is next, retaining all four controls and
  the selected recipe; no language, full-triality, or scaling claim follows.
- G15A-S passes its exact SM75 spanning-chart and center-sensitive transfer
  gate from clean commit `4067926`, artifact SHA-256
  `96e939fa4411e305637961941a565ac26da5a4212b47de3fc198687693b5dbcc`,
  fresh seeds 2281/2287/2293, and the retained `I/C/S/S-broken` controls. All
  train/evaluation frame banks have rank 56. On unseen banks, `S` means span
  `6.89e-7`--`1.18e-6` through L1024, p95 is at most `1.47e-6`, maximum at
  most `1.94e-6`; comparator margins span 0.266--0.536 and broken-2x passes
  every row. Structured direct vector/positive errors are at most `2.36e-5`
  and frame maximum at most `3.54e-5`. This extends the composition-
  only claim from a four-probe chart to a learned signed 28-generator
  dictionary, unseen frame banks, and global center words under oracle edit
  timing. It does not establish learned topology, generic association,
  addressing/querying, negative-spin/Clifford utility, language, or full
  triality, and it provides no scaling/efficiency result.
- G15B completes its prospectively frozen exact-SM75 quality cohort from clean
  commit `bd5045a`: three seeds for each `I/C/S` arm, 4,200 updates and 375,360
  scored training decisions per seed/arm. Preflight and integrity pass, but the
  adjudication fails. Identity reaches about 0.972 mean MQAR and 0.975 selective
  recall, yet overwrite remains 0.768--0.833. `C` is worse; `S` is inferior to
  identity in every MQAR/overwrite/selective mean cell and meets noninferiority
  only on needle cells. Address top-1 is near perfect, while overwrite erase
  recall stays about 0.5 and write F1 is below gate. The frozen erase target is
  collision history, but the controller sees only token/local-convolution
  features: a temporal-observability mismatch. G15C/external-only is blocked.
  G15A-S remains an intact, separate composition result under oracle edit
  timing. G15B-R0 then rejects naive erase-equals-write delta: exact replay
  shows that the learned controllers use a structured one-token write
  continuation, and tying erase to it degrades every non-needle cell. G15B-R1
  preserves that continuation but rejects erase at every valid write:
  overwrite drops 9.7--11.5 points, all nine non-needle gates fail, and learned
  key prototypes are highly nonorthogonal. G15B-R2 then supplies perfect
  collision timing with the full write tail, but post-same-key-overwrite recall
  falls 10.3--12.1 points. G15B-R3 replaces an oracle per-key component and
  repairs ordinary overwrite by 12.2--12.8 points while reaching 1.0 on its
  guard, but its frozen gate still fails. G15B-R4's exact-replay,
  numerically certified factorial passes only value-plus-tail ownership:
  `VT/BG+` reaches 0.8958/0.9543/0.9536 ordinary overwrite and `VT/BG-`
  0.8998/0.9542/0.9543, while `V/BG+` remains below learned at every length
  and seed 2311 collapses. Background exclusion does not hurt the passing tail
  arms, so the useful association is in ambiguous `t+1` ownership rather than
  shared background. G15B-R5 then isolates a strict-history tail injection
  conditional on the unchanged full-token transition. Its background-free LWW
  arm passes every performance and bias-separation check, with mean overwrite
  0.9424/0.9456/0.9466 and 1.0 constructed-guard accuracy. Formal adjudication
  still fails because R4 BPQ replay reaches `1.40e-7` versus `1e-12`, while
  FP32 no-reset-state/background-read residuals reach
  `2.38e-6`/`3.58e-6` versus `2e-6`; discrete replay and learned logits are
  exact and the FP64 algebra passes. R5-S subsequently fails all 135 frozen
  scaled-logit checks, with ratios `1.171875`--`66.078125`, despite exact
  categorical behavior, BPQ drift below `1.72e-7`, state/read residuals below
  `3e-6`, bit-exact transitions, and FP64 algebra below `5.87e-14`. The
  retained-checkpoint repair route stops. Spin transport remains specialized
  to supplied/coherent-frame tasks; a fresh model must first learn explicit
  pending-write/commit semantics on an exact monolithic residual/read path.
- G16 completed its prospectively frozen one-seed SM75 development shootout.
  Official fused Mamba-2 wins ordinary compression at every context and reaches
  `1.48571` BPRB at L4096 versus v1.4.5 `1.58335`; local semantic GDN2 and
  actual OLMo Hybrid lose. All four arms fail the learned-recall gate. This is
  bounded development evidence, not a model-family promotion or scaling law.

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
The successful first-order repair is controlled by the prospectively frozen
[G15A-R protocol](hybrid_memory_v1_4/G15AR_FIRST_ORDER_PROTOCOL_2026-08-25.md)
and source-bound
[repair artifact](hybrid_memory_v1_4/artifacts/g15ar_first_order_repair_sm75_2026-08-25.json).
The spanning-center successor is controlled by the
[G15A-S protocol](hybrid_memory_v1_4/G15AS_SPANNING_CENTER_PROTOCOL_2026-08-25.md)
and source-bound
[G15A-S artifact](hybrid_memory_v1_4/artifacts/g15as_spanning_center_sm75_2026-08-25.json).
The commissioned-controller successor is controlled by the
[G15B protocol](hybrid_memory_v1_4/G15B_CONTROL_PROTOCOL_2026-08-25.md), with
the failed quality adjudication in the
[G15B result](hybrid_memory_v1_4/G15B_INTERLEAVED_CONTROLLER_RESULTS.md) and
source-bound evidence in the
[G15B artifact](hybrid_memory_v1_4/artifacts/g15b_interleaved_controller_sm75_2026-08-26.json),
SHA-256 `f74d860e30ab40ec747521dfcecd74aac2bb75151206c25b7104d334727429eb`.
The zero-update repair is recorded in the
[G15B-R0 result](hybrid_memory_v1_4/G15BR_CHECKPOINT_REPAIR_RESULTS.md) and
[G15B-R0 artifact](hybrid_memory_v1_4/artifacts/g15br_checkpoint_repair_sm75_2026-08-26.json),
SHA-256 `4d92d6af2fb062cf2baaa035c4e4eff89d494dfcb56b9b666523bbbdbfe3cf9c`.
The event-erase successor is recorded in the
[G15B-R1 result](hybrid_memory_v1_4/G15BR1_EVENT_ERASE_RESULTS.md) and
[G15B-R1 artifact](hybrid_memory_v1_4/artifacts/g15br1_event_erase_sm75_2026-08-26.json),
SHA-256 `c015b128846e4b5c63d927778815a87728a7d613369163b1027ed3dd9f0b2912`.
The collision-only successor is recorded in the
[G15B-R2 result](hybrid_memory_v1_4/G15BR2_COLLISION_ERASE_RESULTS.md) and
[G15B-R2 artifact](hybrid_memory_v1_4/artifacts/g15br2_collision_erase_sm75_2026-08-26.json),
SHA-256 `90652fe7034e5901b968eb5d139f02eb8bc714b0417c0889e16a2fdd6b7cf924`.
The component-replacement successor is recorded in the
[G15B-R3 result](hybrid_memory_v1_4/G15BR3_LOGICAL_COMPONENT_RESULTS.md) and
[G15B-R3 artifact](hybrid_memory_v1_4/artifacts/g15br3_logical_component_sm75_2026-08-26.json),
SHA-256 `0fe54b8ce38868d67a7ecb0cb888f2279d8809c2bbaf3ccbda678326ff808959`.
The ownership/background factorial is recorded in the
[G15B-R4 result](hybrid_memory_v1_4/G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md) and
[G15B-R4 artifact](hybrid_memory_v1_4/artifacts/g15br4_ownership_background_sm75_2026-08-26.json),
SHA-256 `921d45e3c492e172fae62064120e9e051dca2965bacc44891268b135d8cef26e`.
The causal tail-source diagnostic is recorded in the
[G15B-R5 result](hybrid_memory_v1_4/G15BR5_CAUSAL_TAIL_SOURCE_RESULTS.md) and
[G15B-R5 artifact](hybrid_memory_v1_4/artifacts/g15br5_causal_tail_source_sm75_2026-08-26.json),
SHA-256 `ba627fe34e8dd29458fc1321b52c98242838c3b56e2abdc7e44c749f50aaa313`.
The prospective numerical follow-up is recorded in the
[G15B-R5-S result](hybrid_memory_v1_4/G15BR5S_NUMERICAL_RATIFICATION_RESULTS.md)
and
[G15B-R5-S artifact](hybrid_memory_v1_4/artifacts/g15br5s_numerical_ratification_sm75_2026-08-26.json),
SHA-256 `3ac514e16e6fa1c720d5ef4244525f5d0f08c233634648e59181c6acfccc3a00`.
The four-arm model harness is frozen separately in the
[G16 protocol](hybrid_memory_v1_4/G16_SM75_FRONTIER_SHOOTOUT_PROTOCOL_2026-08-25.md),
with exact source/runtime evidence in the
[G16 qualification artifact](hybrid_memory_v1_4/artifacts/g16_runtime_qualification_sm75_2026-08-25.json)
and the completed development ordering in the
[G16 result](hybrid_memory_v1_4/G16_SM75_FRONTIER_SHOOTOUT_RESULTS.md).
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
