# Hybrid Memory v1.4/v1.4.5: 2026 frontier review

**Review date:** 2026-08-25
**Disposition:** v1.4.5 is a validated small hybrid causal learner and a
validated commissioned associative-memory model. It is not yet a validated
long-term natural-text memory. The next model-quality experiment should change
the memory edit law before changing tokenizer, optimizer, geometry, or context
length again.

## Outcome first

The present problem is not that the model cannot learn. It is more specific:

1. v1.4.5 learns ordinary TinyStories next-token prediction, and disabling its
   Gated Delta layer makes validation much worse;
2. it learns externally supervised delayed bindings robustly through 512-token
   synthetic sequences;
3. exact-target context training through 4,096 tokens modestly improves
   ordinary loss but does not create reliable recall beyond attention;
4. its recurrent state is written on nearly every token, while the same scalar
   gate controls both erasure of the old association and injection of the new
   value; and
5. the high global retention floor prevents one catastrophic failure mode but
   does not protect directions selected by the delta write itself.

The G13 state diagnostics make that last point measurable. Curriculum training
uses the recurrent layer strongly: removing it costs `+1.3921` bits per raw
byte at length 4,096. Yet mean per-head write strengths are
`[0.446, 0.768, 0.671, 0.618]`, and effective retention in each newly written
direction is only `[0.553, 0.232, 0.329, 0.382]`. That is a fast continuously
edited working memory, not a protected archive.

The direct 2026 successor is Gated DeltaNet-2 (GDN2), not another retention
floor. GDN2 gives the key-side erase and value-side write independent,
channel-wise gates. The official paper reports that this strict generalization
of KDA and Gated DeltaNet was strongest overall in its matched 1.3B/100B-token
study, especially on multi-key long-context retrieval. Those external results
are evidence for prioritization, not evidence about this 125K-parameter model.
[Paper](https://arxiv.org/abs/2605.22791) and
[official implementation](https://github.com/NVlabs/GatedDeltaNet-2).

## What the full v1.4 record actually establishes

### Rejected routes

- The inherited selected-block design had a structural temporal-observability
  failure: one-block final-only loss could not observe non-final reads, and
  hard routes disconnected controller logits.
- The G4a selected-memory and DeltaProduct common-shell arms both stayed near
  chance. More routing supervision did not solve a missing content-addressed
  association.
- G4b through G4e repeatedly improved means while failing frozen worst-seed
  gates. The sequence of failures localized value interference, unstable
  identity/readout initialization, and independently rotated query/key frames.
- G6 and G7 showed that a learned causal reverse-binding auxiliary was not
  sufficient and that elapsed-update or short-task competence schedules did
  not ensure target-distance retention.
- G9 showed that unrestricted learned global decay can enter a destructive
  seed-dependent basin. A head at retention `0.9004` across roughly 416 filler
  transitions cannot preserve old state.
- G13 rejected the claim that longer ordinary pretraining alone produces the
  desired archive. The 4,096 curriculum won all three paired ordinary-loss
  rows but missed its frozen effect-size gate, while 8,192-byte factual recall
  was tiny and sign-unstable.

### Supported results

- Exact recurrent, parallel, arbitrary-chunk, and token-step semantics; exact
  state-byte accounting; and fp32/fp16 horizon audits are implemented.
- G4f validated commissioned synthetic learning at about 98--99% across three
  fresh seeds after tying query/key geometry and repairing the value/readout
  path. This uses explicit internal supervision.
- G10 validated the retention-safe v1.4.5 external-causal schedule across
  three fresh seeds, with every L96/L512 result above 96%.
- G11 validated ordinary next-byte learning: v1.4.5 reached 1.614 BPC from
  8.028, and its recurrent state carried most of the bounded learned function.
- G12 validated multi-seed ordinary-text improvements from the repository's
  composite optimizer and exact-round-trip 512-token ByteLevel BPE. It did not
  validate factual recall. The BPE arm also saw about 2.3 times more raw bytes
  for the same number of token targets, so its advantage is not tokenizer-only.
- G13 validated finite training through `256 -> 512 -> 1024 -> 2048 -> 4096`
  with the exact same 4,096 target token IDs per paired update. It rejected the
  requested long-memory promotion.

The old `ssm-research-in-ox-alpha.json` session correctly emphasized affine
scan discipline, hybrid attention, state accounting, negative controls, and
observability. Its early claims that hybrid attention, Gated Delta memory,
long-context runs, and upstream comparisons were still missing are now
historical: those items have been implemented and tested. Its central warning
survives—memory quality depends on learned write/read/forget control, not merely
on having a large recurrent state.

## Lessons from the rest of the repository

| Lineage | What survived | What it rules out for v1.4 |
|---|---|---|
| Pure Rotor v2.1 | exact associative transport, streaming semantics, negative controls | geometry alone is not memory; quaternion/commuting controls did better and compute-matched identity won |
| Pure Spin v1.2 / Spin-Delta | recurrent capacity can be high while autonomous event/address inference is the bottleneck; temporal observability must be audited | do not add another router or curriculum until its causal grammar and loss path are explicit |
| Pure F4 v1.3 | exceptional structure can be localized and cleanly ablated | a one-seed geometric gain that fails a fresh frozen gate stays rejected; identity is the natural-text reference |
| Pure Spin(8) experiments | exact algebra-matched actions can dominate on injective synthetic identification | task-aligned synthetic success is not generic language-model superiority |
| v1.4 DeltaProduct control | multiple delta updates and precision contracts are already present | “add DeltaProduct” is not a new frontier move; it has not solved autonomous ordinary-text recall here |

The common lesson is architectural harmony: the state law, observation path,
training signal, data distance, tokenizer exposure, optimizer geometry, and
kernel must all serve the same capability. Fixing only one of them repeatedly
produced a good mean and a failed worst seed.

## What production hybrids reveal that v1.4.5 does not test

The local model is a two-block microarchitecture (`gated_delta -> attention`).
Modern successful hybrids are deep schedules, not one recurrent layer plus one
attention layer:

- Qwen3-Next uses 48 layers in a repeated 3 Gated DeltaNet to 1 attention
  pattern and declares 262,144-token native context.
  [Official model card](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct)
- Kimi Linear uses 20 KDA and 7 full-attention mixers. Its paper reports a 1M
  context production model and attributes the finite-memory improvement to
  channel-wise decay and a DPLR transition.
  [Paper](https://arxiv.org/abs/2510.26692)
- Mamba-3 adds an exponential-trapezoidal recurrence, complex state rotation,
  and MIMO updates; its paper reports better state tracking and a hybrid 5:1
  linear-to-NoPE-attention schedule.
  [Paper](https://arxiv.org/abs/2603.15569)
- Falcon-H1 combines attention and Mamba heads within hybrid blocks, while
  Nemotron-H interleaves Mamba-2, attention, and MLP blocks.
  [Falcon-H1 model](https://huggingface.co/tiiuae/Falcon-H1-0.5B-Base) and
  [Nemotron-H model](https://huggingface.co/nvidia/Nemotron-H-4B-Base-8K)

Therefore the current experiment cannot answer whether the right recurrent
law works when repeated, whether attention should be periodic or parallel,
or whether the recurrent layers should use a different positional convention.

## 2026 method map and relevance

### Tier A: direct falsifiers of the measured failure

1. **GDN2: decouple erase and write.** This directly addresses the scalar
   coupling measured in G13 and is the first experiment to run.
2. **KDA: channel-wise decay with tied erase/write.** This is the necessary
   control. It distinguishes the benefit of finer decay from the benefit of
   erase/write separation. Kimi Linear supplies the large-scale external case.
3. **Erase-then-Delta Attention (EDA): independent erase address.** GDN2
   separates how much to erase/write; EDA also separates where stale memory is
   erased. Use it if GDN2 writes correctly but stale associations remain.
   [Paper](https://arxiv.org/abs/2606.26560)
4. **Q-Delta: query-aware state evolution.** Use this if diagnostics show that
   the query carries address information the key-only write law never learns.
   Its query participates in the corrective update, rather than only reading.
   [Paper](https://arxiv.org/abs/2606.08804)

### Tier B: stronger alternative memory estimators and controls

5. **Mamba-3 SISO/MIMO** is the primary non-delta state-tracking control. Its
   complex rotation and MIMO state test whether the problem is associative
   edit control or more general finite-state dynamics.
6. **Gated KalmaNet (GKA)** maintains a ridge-regression/Kalman-style memory
   rather than taking one delta correction. It is especially relevant if
   repeated correlated keys make the v1.4 state ill-conditioned.
   [Paper](https://arxiv.org/abs/2511.21016)
7. **Kalman Linear Attention (KLA)** explicitly carries uncertainty and uses an
   associative information-form recursion. Its present public evidence is much
   smaller than production GDN/KDA evidence, so it is a research control rather
   than the first pivot. [Paper](https://arxiv.org/abs/2602.10743)
8. **Cross-Layer Value Routing (CLVR)** is a low-cost depth experiment after a
   multi-layer stack exists. A 2026 350M/15B-token single-run study found modest
   GDN/DeltaNet gains from aligned value routing, while routing delta error did
   not help. [Study](https://arxiv.org/abs/2607.07953)

### Tier C: a second memory timescale, not a first-line replacement

9. **TTT and Titans-style learned memory** can supply a slower archive whose
   state is itself an online-trained model. They are relevant only after an
   admission/consolidation objective is defined; otherwise they create a more
   expensive version of the same uncontrolled-write problem.
   [TTT](https://arxiv.org/abs/2407.04620) and
   [Titans](https://arxiv.org/abs/2501.00663)
10. **State composition and priming** may be more practical than asking one
    sequential state to survive arbitrary overwrite. Hybrid Model Factory maps
    pretrained attention weights into SSM projections and experimentally
    composes independently processed chunk states. This is a high-value route
    for later scale, but its published examples use 8B models and much larger
    hardware. [Priming toolkit](https://github.com/awslabs/hybrid-model-factory)
    and [state-composition guide](https://github.com/awslabs/hybrid-model-factory/blob/main/docs/StateComposition.md)
11. **PD-SSM / structured finite-state dynamics** are relevant to the repo's
    group/state-tracking experiments, but should enter natural text only after
    a generic GDN2/Mamba-3 control wins. [Paper](https://arxiv.org/abs/2509.22284)
12. **StateFlow** is a later systems move for 256K training. It propagates
    boundary states and gradients across sequence-pipeline chunks; it does not
    repair a bad memory update law. [Paper](https://arxiv.org/abs/2608.06838)

## What was still missing from view

1. **Erase/write admission precision.** Existing diagnostics report average
   write and retention, not whether writes, erasures, and queries occur on the
   correct semantic events. Future artifacts need event-conditioned gate
   precision/recall and stale-versus-live association probes.
2. **Collision and occupancy curves.** State bytes are exact, but effective
   associations per head, key correlation, rank, condition number, and error as
   occupancy rises are not yet mapped.
3. **A true archive objective.** Ordinary next-token loss rewards recent local
   prediction far more often than rare delayed recall. Longer sequences do not
   change that imbalance. Commissioned delayed-query or contrastive
   preservation events must be mixed into ordinary pretraining without using
   internal oracle controls.
4. **Depth/layout scaling.** One recurrent layer cannot stand in for the 3:1,
   5:1, interleaved, or parallel hybrid layouts used by current systems.
5. **Production block parity.** v1.4.5 uses a common model-dimension local
   convolution and optional identity readout. Production GDN2/KDA use separate
   q/k/v projections and convolutions plus a gated RMS-normalized output. Gate
   law and block shell must be ablated separately.
6. **Data-scale generalization.** The repo has strong controlled evidence at
   roughly 125K parameters and millions of tokens/bytes; the papers operate at
   hundreds of millions to billions of parameters and tens to hundreds of
   billions of tokens. Neither scale can be extrapolated to the other.
7. **Joint optimizer/learning-rate tuning.** The repository composite optimizer
   already uses Muon for large hidden matrices and Adam-like control updates.
   A 2026 matched study found Muon improvements in all its tested architecture
   pairs, but also different optimal learning rates. An optimizer-only swap is
   therefore not a clean causal test.
8. **Pretrained initialization.** From-scratch microtraining may spend most of
   its budget learning lexical and local structure. Priming from an attention
   teacher is a distinct hypothesis about learnability, not merely a scaling
   trick.
9. **Two-timescale consolidation.** The current state is a single fast memory.
   A protected slow state needs a measurable admission rule, consolidation
   cadence, and interference test before Spin/F4/Rotor structure is reintroduced.
10. **Kernel/hardware separation.** The RTX 2070 SUPER is SM75. Modern fused
    kernels target newer toolchains and GPUs. Semantic quality gates should run
    first; kernel availability and throughput are a separate systems track.

## New implementation and G14 result

The repository now contains a semantic `gated_delta_v2` common-shell layer. It
implements

`S_t = (I - k_t (b_t * k_t)^T) D_t S_(t-1) + k_t (w_t * v_t)^T`.

Tests establish:

- recurrent/parallel/arbitrary-chunk equality;
- exact reduction to Gated DeltaNet v1 when decay, erase, and write collapse
  to the corresponding scalars;
- independent no-erase/write and erase/no-write operations;
- finite gradients through query, key, value, erase, write, decay, output gate,
  and output projection; and
- common-shell streaming and state-byte accounting.

G14 was frozen before its learning run. Its task asks a fixed address to retain
the union of eight distinct value features, scores the raw state, and supplies
no decoder. For tied GDN-v1, the final nonnegative coefficients sum to at most
one, so the minimum possible length-eight mean squared error against eight ones
is `49/64 = 0.765625`. GDN2 can represent the target with zero erase and unit
write on the active channel.

| Arm | Parameters | Held-out MSE, three seeds | Bit accuracy | Learned erase | Active write |
|---|---:|---:|---:|---:|---:|
| tied GDN-v1 | 10 | 0.770998--0.771060 | 0% | 0.413--0.414 | same gate |
| decoupled GDN2 | 90 | 0.000181--0.000292 | 100% | 0.0042--0.0051 | 0.9977 |

Every preregistered G14 gate passed. This proves a representational separation
for the controlled state law and shows that AdamW can learn the desired gate
policy. It does **not** prove natural-text recall, parameter efficiency, or a
full-model improvement. The parameter mismatch is explicit.

## Actual maintained-code probes

The Hugging Face workflow inspected official model metadata/configuration
and later downloaded the actual 130M/187M Mamba weight sets selected for local
qualification. Transformers 5.9.0 tiny random
instances of Qwen3-Next GDN, Mamba-2, Falcon-H1, and Nemotron-H all completed
finite forward/backward on the local CUDA runtime. These remain semantic
availability probes, not pretrained results.

The stricter WSL SM75 qualification binds every passing package to its source
checkout. Actual source-current Mamba-3 SISO completed training forward/
backward with all parameter gradients; MIMO failed on an SM80 TileLang MMA
requirement. Actual FLA GDN2 recurrent did not return recurrent-core parameter
gradients, and chunk backward exceeded the bounded attempt, so neither is a
local training baseline. The source-built Turing FlashAttention extension
passed frozen FP16 output/gradient error gates. Actual Mamba-3 187M weights
produced finite logits/loss on the RTX 2070 SUPER. See
[`SM75_NATIVE_RUNTIME.md`](SM75_NATIVE_RUNTIME.md). None of these execution
boundaries is a model-quality result.

## Ranked next moves

### 1. Run G15A Spin-Dirac mechanism gate

The Spin path is now the active preregistered transport study, but it is built
over content addressing and independent edit controls rather than the old
transport-only cache. Its training-dtype state-growth, mapped optimizer/SGD
covariance, and delayed-descent artifact passes. Run identity, identity+
Clifford, fixed `SO(2)^4`, and full Spin(8) on a symmetry-visible task and a
no-symmetry control. The exact constrained `SU(3)` torus is an additional arm.
Full Spin cannot earn a triality claim without beating the broken-coupling
control. See [`G15_SPIN_DIRAC_RESULTS.md`](G15_SPIN_DIRAC_RESULTS.md).

### 2. G16: matched gate-law cohort

Freeze three common-shell arms:

- v1.4.5 GDN: scalar decay, scalar tied erase/write;
- KDA: channel decay, scalar tied erase/write;
- GDN2: channel decay, channel erase, channel write.

Match recurrent state scalars exactly. Enumerate width/FFN choices to match
parameters and measured CUDA update time within a frozen tolerance. Run fresh
seeds on both ordinary next-token data and a causal delayed-binding mixture.
This is the experiment that decides whether GDN2 becomes the v1.4.6 default.

### 3. Tune optimizer with architecture, not before it

For each surviving gate law, compare AdamW against the existing composite
Muon/control optimizer with a small frozen learning-rate grid. Equalize scored
raw bytes and exact token targets. Do not infer an optimizer verdict from a
single shared learning rate.

### 4. Scale depth before context

After a layer-local win, compare periodic schedules such as
`GDN2-GDN2-GDN2-attention` against the current two-layer model at matched
parameters/compute. Only then resume the exact-target context staircase
`256 -> 512 -> 1024 -> 2048 -> 4096` and test at 8,192+ beyond the attention
window.

### 5. Add memory-health evaluation

Every checkpoint should report:

- event-conditioned erase/write/admission calibration;
- live, stale, overwritten, and contradictory-key recall;
- state rank, singular spectrum, key correlation, and occupancy curves;
- recall distance in both tokens and represented raw bytes;
- recurrent and attention ablations;
- multi-seed ordinary-text loss and parameter/compute/state matching.

### 6. Escalation rules

- If KDA matches GDN2, the gain is channel decay, not decoupled edit control.
- If GDN2 wins synthetic memory but not ordinary text, fix the objective/data
  mixture or use priming; do not add geometry.
- If GDN2 writes well but cannot delete stale bindings, test EDA.
- If queries identify the needed edit better than keys, test Q-Delta.
- If all delta laws fail state tracking, test Mamba-3 and GKA/KLA.
- Spin geometry is tested only as matched transport over the repaired edit law;
  moving `G2` frames and F4/E6 text transport still wait for fixed-transport
  falsifiers.

## Explicit nonclaims

- GDN2 is not promoted as the v1.4 default.
- G14 is not a language-model or long-context result.
- G15A passes its three-seed oracle-coordinate mechanism gate, but conditional
  coupling/readout attribution, generic association, natural text, and scaling
  remain open; the result does not promote Spin to the default model.
- Tiny random upstream probes are not pretrained evaluations. Actual Mamba
  checkpoint finiteness is not a quality or throughput ranking.
- External paper results are not reproduced locally.
- No current result supports a `10,000x` optimizer claim, a tokenizer cure, or
  generic superiority over Mamba-3, Kimi Linear, Qwen3-Next, GKA, or attention.

## Evidence added by this review

- [`G14_PREREGISTRATION.md`](G14_PREREGISTRATION.md)
- [`gated_delta_v2.py`](gated_delta_v2.py)
- [`g14_gate_law_screen.py`](g14_gate_law_screen.py)
- [`artifacts/g14_gate_law_screen_2026-08-25.json`](artifacts/g14_gate_law_screen_2026-08-25.json)
- [`modern_ssm_probe.py`](modern_ssm_probe.py)
- [`artifacts/modern_ssm_transformers_probe_2026-08-25.json`](artifacts/modern_ssm_transformers_probe_2026-08-25.json)
- [`artifacts/modern_ssm_fla_probe_2026-08-25.json`](artifacts/modern_ssm_fla_probe_2026-08-25.json)
- [`artifacts/modern_ssm_hf_inventory_2026-08-25.json`](artifacts/modern_ssm_hf_inventory_2026-08-25.json)
- [`SPIN_TORUS_RESEARCH.md`](SPIN_TORUS_RESEARCH.md)
- [`G15_SPIN_DIRAC_PREREGISTRATION.md`](G15_SPIN_DIRAC_PREREGISTRATION.md)
- [`G15_SPIN_DIRAC_AMENDMENT_2026-08-25.md`](G15_SPIN_DIRAC_AMENDMENT_2026-08-25.md)
- [`G15_SPIN_DIRAC_EDIT_LAW_AMENDMENT_2026-08-25.md`](G15_SPIN_DIRAC_EDIT_LAW_AMENDMENT_2026-08-25.md)
- [`G15_SPIN_DIRAC_RESULTS.md`](G15_SPIN_DIRAC_RESULTS.md)
- [`SM75_NATIVE_RUNTIME.md`](SM75_NATIVE_RUNTIME.md)
- [`RESEARCH_LOG_2026-08-25.md`](RESEARCH_LOG_2026-08-25.md)
