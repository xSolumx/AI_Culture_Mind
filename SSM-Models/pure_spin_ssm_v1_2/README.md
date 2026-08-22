# Pure Spin SSM v1.2

This folder is the isolated implementation and evidence boundary for Pure Spin
SSM v1.2. It does not inherit empirical claims from older rotor, Spin(8), or
synthetic-memory experiments.

The current local execution contract is documented in the
[`hardware profile`](LOCAL_HARDWARE_PROFILE_2026-08-22.md) and
[`storage audit`](STORAGE_AUDIT_2026-08-22.md): WSL is authoritative, CUDA
12.6 targets native `sm_75`, source/build/environments stay on WSL ext4, and
only large weights plus download/dataset caches belong on E:.
The [`pre-BIOS handoff`](PRE_BIOS_HANDOFF_2026-08-22.md) records the repaired
WSL user session, frozen environment, and mandatory post-firmware smoke checks.
The [`post-BIOS validation`](POST_BIOS_VALIDATION_2026-08-22.md) records the
accepted F14a, Secure Boot, DDR4-3200, systemd, and CUDA readiness state.

## Architecture

Each causal block contains:

1. RMS normalization and a gated input projection;
2. causal depthwise convolution for local token mixing;
3. the maintained bounded Spin(8) recurrence with shared `8v/8+/8-` triality
   action and a fixed-size recurrent cache;
4. a selectable exact nested-group ladder
   `Spin(3) -> Spin(4) -> Spin(6) -> Spin(8)`;
5. raw CUDA controller-fused and GEMM-plus-factorized training lowerings;
6. gated residual readout and a SwiGLU channel mixer.

The maintained readout remains the normalized 48-dimensional triality
direction. An optional `triality_invariants` control restores sector energies
and the cubic Spin(8) invariant, but its prospectively frozen three-seed gate
improved all seeds by only 0.00661 mean bpb, below the 0.0100 promotion
threshold. See
[`TRIALITY_INVARIANT_READOUT_RESULTS.md`](TRIALITY_INVARIANT_READOUT_RESULTS.md).
The subsequent Schur-legal `orthogonal_query` control rotates the two-copy
multiplicity axis without disturbing the shared Spin(8) action. It also failed
promotion in its original run (1/3 wins; +0.00084 mean bpb), so
`multiplicity_router=none` remains the default. A later initialization audit
found that this was not the claimed exactly paired identity-start ablation; the
artifact is retained but is no longer treated as a strict falsification. See
[`COUPLED_ISOTYPIC_RESULTS.md`](COUPLED_ISOTYPIC_RESULTS.md).

The stronger recurrence-level candidate is now fully compiled. It uses one
shared Spin action and a contractive `SO(2)` multiplicity action, with an exact
two-sided affine prefix algebra and full raw-CUDA backward. Its valid paired
gate won 2/3 seeds but regressed by 0.00214 mean bpb, so recurrent mixing is not
promoted. The compiler remains supported research machinery; the maintained
architecture remains the independent-action `raw_cuda_hybrid` model.
The shared-action identity model was then tested separately as a compression
candidate. It lost all three seeds by 0.02415 mean bpb and failed quality
non-inferiority, showing that the independent channel controllers are useful at
this scale. See
[`SHARED_ACTION_COMPRESSION_RESULTS.md`](SHARED_ACTION_COMPRESSION_RESULTS.md).

The resulting frontier preserves both independent Spin controllers and inserts
an identity-start `SO(2)` mixer after their tokenwise actions. Its exact closure
is a 16-dimensional block-affine monoid per triality sector, while the raw CUDA
streaming path evaluates the structured factors directly with the original
48-scalar cache. This is an experimental control pending its frozen Shakespeare
gate; it is not the maintained default. See
[`INDEPENDENT_BLOCK_RECURRENCE_PREREGISTRATION.md`](INDEPENDENT_BLOCK_RECURRENCE_PREREGISTRATION.md).
Its free-angle gate subsequently won 1/3 seeds and regressed by 0.01565 mean
bpb, so it remains research-only. See
[`INDEPENDENT_BLOCK_RECURRENCE_RESULTS.md`](INDEPENDENT_BLOCK_RECURRENCE_RESULTS.md).
The prospective successor scales that coupling by the local retention-derived
step `1-geometric_mean(s)`, matching the first-order continuous-time
discretization rather than applying a full rotation at every token. It remains
an unpromoted control pending the frozen gate in
[`RETENTION_SCALED_BLOCK_PREREGISTRATION.md`](RETENTION_SCALED_BLOCK_PREREGISTRATION.md).
It has now passed quality with 3/3 wins and +0.01427 mean bpb; default promotion
remains conditional on systems evidence. Its first packed-warp speed gate
failed at 0.7642x maintained throughput. The mathematically equivalent
isotypic-forward rescue improved this to 0.8049x but also failed its frozen
0.90x boundary. A guarded analytic reconstruction removes one ordinary-path
backward replay, but its 0.7879x gate also failed. The model remains a
quality-positive research control, not the default. See
[`RETENTION_SCALED_BLOCK_RESULTS.md`](RETENTION_SCALED_BLOCK_RESULTS.md).

The next model-level candidate steps away from cross-channel kernel rescue. It
keeps the maintained independent Spin actions but removes an accidental
timescale tying: `8v`, `8+`, and `8-` receive centered, independently learned
retention offsets, which are legal because the sectors are inequivalent
irreducible Spin(8) modules. The residual is exactly zero-start, all common
parameters and initial logits are bitwise paired, and the semantic plus raw
CUDA paths pass 69 WSL/cu126 tests. Its frozen gate is documented in
[`ISOTYPIC_RETENTION_PREREGISTRATION.md`](ISOTYPIC_RETENTION_PREREGISTRATION.md).
That dynamic candidate subsequently won 2/3 seeds but improved by only
`+0.00242` mean bpb because seed 281 regressed by `0.03472`; it is not
promoted. See
[`ISOTYPIC_RETENTION_RESULTS.md`](ISOTYPIC_RETENTION_RESULTS.md).
The lower-variance successor factors retention into a token-dependent shared
step and a learned static positive decay spectrum on `8v/8+/8-`. It adds only
24 model parameters and is prospectively frozen in
[`ISOTYPIC_SPECTRUM_PREREGISTRATION.md`](ISOTYPIC_SPECTRUM_PREREGISTRATION.md).
That spectrum lost all three frozen seeds by `0.01364` mean bpb, closing
sector-retention refinement as the immediate successor. The next architecture
separates independent Spin transport heads from addressable delta-memory slots;
see [`SPIN_DELTA_SUCCESSOR_DESIGN.md`](SPIN_DELTA_SUCCESSOR_DESIGN.md). Its
semantic compiler and model path are now implemented with
`recurrence="spin_delta"`: two independent transport heads each carry two
addressable triality slots, with a contractive rank-one erase, routed write,
and bounded read probe. Sequential/parallel output and gradient parity,
baseline embedding, and long-sequence finiteness are tested. It has no quality
or speed promotion yet. Its full-training raw-CUDA lowering reuses the audited
two-copy kernel over a flattened `(batch, transport_head)` grid and passes
semantic plus full-model gradient parity across the Spin(3/4/6/8) ladder. The
frozen natural-data decision is specified in
[`SPIN_DELTA_PREREGISTRATION.md`](SPIN_DELTA_PREREGISTRATION.md).
The corrected frozen cohort subsequently won only 1/3 seeds, regressed by
`0.02521` mean bpb, and exceeded the maximum allowed single-seed regression.
It is not promoted and no speed gate is authorized. See
[`SPIN_DELTA_RESULTS.md`](SPIN_DELTA_RESULTS.md).
The mechanism is being separated from that failed language claim in a frozen
two-key overwrite/retrieval capability gate; see
[`SPIN_DELTA_CAPABILITY_PREREGISTRATION.md`](SPIN_DELTA_CAPABILITY_PREREGISTRATION.md).
That gate found one approximately 97%-accurate, length-stable Spin-Delta seed,
but the other two seeds failed the frozen trained-length capability boundary
and maintained v1.2 had higher mean accuracy. This establishes possible but
non-robust capability and no differential advantage. See
[`SPIN_DELTA_CAPABILITY_RESULTS.md`](SPIN_DELTA_CAPABILITY_RESULTS.md).
The next causal control supplies only exact write/query slot identities while
leaving the Spin-Delta model and CUDA recurrence unchanged; its frozen protocol
is [`SPIN_DELTA_ORACLE_ADDRESS_PREREGISTRATION.md`](SPIN_DELTA_ORACLE_ADDRESS_PREREGISTRATION.md).
That intervention passed every frozen gate: oracle addressing reached
99.3--100% in all seeds and lengths and rescued 16-write accuracy by 13.55
points on average. The recurrence has sufficient overwrite capacity; the
autonomous failure lies in causal write/query event and slot inference. See
[`SPIN_DELTA_ORACLE_ADDRESS_RESULTS.md`](SPIN_DELTA_ORACLE_ADDRESS_RESULTS.md).

The follow-up autonomous causal router learned every synthetic event and slot
perfectly across three seeds and all lengths, but end-to-end retrieval remained
seed-unstable and lost 1.60 points on average at 16 writes. Identification is
therefore no longer the unresolved variable; joint router/recurrence
co-adaptation is. See
[`SPIN_DELTA_CAUSAL_ROUTER_RESULTS.md`](SPIN_DELTA_CAUSAL_ROUTER_RESULTS.md).

Training that perfect router first and freezing it before exposing a pristine
core raised the three-seed mean above 97%, but one seed still missed the 95%
8/16-write threshold and the schedule did not beat joint training. Early
routing noise is therefore not the whole optimization defect. See
[`SPIN_DELTA_PHASED_ROUTER_RESULTS.md`](SPIN_DELTA_PHASED_ROUTER_RESULTS.md).

A subsequent exact-control 3x3 factorial separated initialization from batch
order. Sixteen-write accuracy ranged from 91.41% to 100%; both factors crossed
the preregistered five-point sensitivity threshold, with a strong sign-changing
interaction. The remaining frontier is core training geometry, not routing.
See [`SPIN_DELTA_PERFECT_CONTROL_FACTORIAL_RESULTS.md`](SPIN_DELTA_PERFECT_CONTROL_FACTORIAL_RESULTS.md).
The repository-wide
[`mechanism reuse audit`](SPIN_DELTA_MECHANISM_REUSE_AUDIT.md) then traced the
relevant memory, curriculum, optimizer, representation, readout, and compiler
experiments before selecting another intervention. Its evidence-ranked next
gate is a fresh exact-control short-to-long write curriculum; local
reconstruction credit and coordinate-covariant optimizer controls follow only
if that gate fails. No new architecture claim is attached to the audit. The
paired 3x3 implementation and frozen decision are in
[`SPIN_DELTA_WRITE_CURRICULUM_PREREGISTRATION.md`](SPIN_DELTA_WRITE_CURRICULUM_PREREGISTRATION.md);
all 18 artifacts subsequently completed. The curriculum passed every frozen
condition: its worst 16-write cell rose from 93.51% to 99.37%, its mean paired
change was +1.11 points, and its largest factorial range fell from 6.49 to
0.63 points while using 24.52% fewer training tokens. See
[`SPIN_DELTA_WRITE_CURRICULUM_RESULTS.md`](SPIN_DELTA_WRITE_CURRICULUM_RESULTS.md).
The autonomous transfer gate now freezes one supervised causal router, clones
its untouched core into fixed-depth and curriculum arms, and audits every
phase-B hard control without supplying oracle controls to the model. Its
prospective protocol is
[`SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_PREREGISTRATION.md`](SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_PREREGISTRATION.md).
The first commissioning cohort failed its bitwise post-router replication gate
before summarization; its quality fields are not evidence. The failure and the
single-router-execution repair are documented in
[`SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_COMMISSIONING_FAILURE.md`](SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_COMMISSIONING_FAILURE.md)
and prospectively frozen in
[`SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_V2_PREREGISTRATION.md`](SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_V2_PREREGISTRATION.md).
V2 passed every frozen condition. With perfect learned causal controls, the
curriculum raised worst 16-write accuracy from 78.96% to 97.02%, improved the
nine-cell mean by 4.30 points, and contracted the largest factorial range from
21.00 to 2.83 points. See
[`SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_RESULTS.md`](SPIN_DELTA_ROUTER_CURRICULUM_TRANSFER_RESULTS.md).
The next gate removes router commissioning labels entirely and jointly trains
router plus core from retrieval loss. Its slot-identification decision is
defined modulo the unavoidable global two-slot permutation. See
[`SPIN_DELTA_LABEL_FREE_CURRICULUM_PREREGISTRATION.md`](SPIN_DELTA_LABEL_FREE_CURRICULUM_PREREGISTRATION.md).
The frozen cohort closed negatively: curriculum mean 16-write accuracy was
34.30% versus 62.84% at fixed depth, with 93-point factorial ranges. Query-event
F1 stayed zero in every curriculum cell even when retrieval reached 98.19%,
exposing the internal-query fallback as an optimization bypass rather than
recovering the explicit grammar. See
[`SPIN_DELTA_LABEL_FREE_CURRICULUM_RESULTS.md`](SPIN_DELTA_LABEL_FREE_CURRICULUM_RESULTS.md).
Before another quality gate, the query branch is being audited at one-step
gradient resolution under current hard fallback, soft event continuation, and
immediately authoritative routing. The prospective selector is
[`SPIN_DELTA_QUERY_GRADIENT_TOPOLOGY_PREREGISTRATION.md`](SPIN_DELTA_QUERY_GRADIENT_TOPOLOGY_PREREGISTRATION.md).

The recurrence is not Mamba-2 under different notation. Its state transition is
a selective contractive affine Spin(8) action. Local convolution and SwiGLU are
included because the old pure recurrence lacked the local/channel mixing needed
for a credible language-model comparison.

## Required comparison

`benchmark.py` trains both candidates on identical windows from a pinned Tiny
Shakespeare UTF-8 byte stream. The exact upstream commit and full-file SHA-256
are fixed in `data.py`; deterministic chronological 90/5/5 slices keep
training, validation, and test bytes disjoint. The harness records all split
hashes, parameter counts, bits per byte, throughput, peak CUDA memory, software
versions, and the exact GPU. The default comparison uses a 128-wide Pure Spin
model and a 144-wide Mamba-2 model; it constructs both candidates up front and
refuses the run unless trainable parameter counts differ by at most five
percent. `--dataset wikitext2_legacy` exists only to replay older artifacts.

The baseline imports `mamba_ssm.Mamba2` with `use_mem_eff_path=True`. The run
fails closed when the official fused SSD kernel is unavailable; Transformers'
reference implementation is never relabeled as fused.

Run under WSL/Linux because the official Mamba package requires Linux and an
NVIDIA CUDA environment:

```bash
cd /mnt/c/Users/HaydenLocal/Programming/AI_Culture_Mind/SSM-Models/pure_spin_ssm_v1_2
# One time, from Windows PowerShell:
wsl.exe -d Ubuntu -u root -- bash /mnt/c/Users/HaydenLocal/Programming/AI_Culture_Mind/SSM-Models/pure_spin_ssm_v1_2/bootstrap_cuda126_wsl.sh
# Then inside WSL:
bash install_wsl.sh
source wsl_env.sh
bash run_wsl_tests.sh
python benchmark.py --dataset tiny_shakespeare --offline \
  --steps 300 --batch-size 8 --sequence-length 256 \
  --spin-backend raw_cuda_hybrid --spin-group-schedule 3 4 6 8
```

The validated local tuple is Python 3.10, Torch 2.10.0+cu126, Triton 3.6.0,
official `mamba_ssm` 2.3.2.post1, and official `causal-conv1d` 1.7.0.
`install_wsl.sh` installs and probes that exact tuple. Its wheel URLs encode the
CUDA, Torch, ABI, Python, OS, and architecture match explicitly; `--no-deps`
prevents backend packages not exercised by this benchmark from replacing the
pinned runtime. The install probe then imports the exact fused SSD symbol.
`wsl_env.sh` keeps the native venv and latency-sensitive compiler caches on
WSL ext4 for compiled-wheel, symlink, and small-file performance. It fails
closed unless E: is mounted read/write. Dataset, Hub, and pip download caches
default to `E:\AI_Culture_Mind_Large\pure_spin_ssm_v1_2`; Torch-extension,
Triton, Inductor, and CUDA compilation caches stay under
`/home/local/.cache/pure_spin_ssm_v1_2`. This prevents new large downloads
from consuming C: without slowing compiler metadata traffic. Override only
the task-specific `PURE_SPIN_V12_*` variables documented in that script.

## Claim ledger

- Algebraic Spin(8) action, bounded recurrence, and scan identities are inherited
  only from their maintained exact/unit-tested modules.
- Shape, causality, gradient-finiteness, and fallback-refusal are unit tests.
- Tiny Shakespeare losses and throughput are empirical properties of a
  recorded run; the preserved WikiText tables are historical results.
- Optional mechanisms remain controls when they fail their frozen promotion
  threshold, even if their mean effect has the favorable sign.
- An artifact whose commissioning audit violates its pairing contract is
  labeled invalid or non-decisive; a negative outcome is not retroactively
  upgraded into a falsification.
- No quality, speed, Tensor-Core, or scaling advantage is claimed before a
  complete artifact exists for both candidates on the same environment.
- Parameter matching is measured before training and the harness fails closed
  when the raw trainable counts differ by more than five percent.
- Falcon-Mamba-7B, Mamba-3 SISO/MIMO, GKA, GDN, and Jamba source pins and
  hardware-feasibility boundaries live in
  [`external_baselines/`](external_baselines/). Metadata auditing is not a
  performance result, and pretrained-system losses are not comparable to this
  small from-scratch byte-LM loss.

The current three-seed, 300-step Tiny Shakespeare result does not show a v1.2
quality win: official fused Mamba-2 wins all three seeds and averages 2.4942
versus 2.7477 bits/byte, while using 22% less peak allocated CUDA memory.
Post-F14a order-balanced cu126 timing reverses ordering across identical
repeats (Mamba +2.91%, then Spin +1.58%); the compatibility-checked verdict is
that throughput ordering is unresolved at observed repeatability. See
[`FRONTIER_TRAINING_RESULTS.md`](FRONTIER_TRAINING_RESULTS.md) for the complete
scope and provenance.

## Raw CUDA comparison

`csrc/spin_scan_cuda.cu` contains the historical materialized-action forward
kernel plus two full FP32 training schedules. The fastest schedule uses a dense
GEMM controller followed by a register-resident raw CUDA ordered-factor
recurrence. Exact backward, current-stream behavior, subgroup restriction, and
full-model gradient parity are maintained tests.

The first RTX 2070 SUPER comparison is reported in
[`RAW_CUDA_RESULTS.md`](RAW_CUDA_RESULTS.md): after current-stream integration
and CUDA-event timing, raw CUDA took 145.1 microseconds versus Triton's 175.5
microseconds in FP32, and 121.6 versus 183.2 microseconds in FP16, on the
recorded shape. It is a one-shape result, not a general speed claim, and the
scalar 8-by-8 kernel is not a Tensor-Core implementation.

See [`FRONTIER_TRAINING_RESULTS.md`](FRONTIER_TRAINING_RESULTS.md) for the
guarded-backward kernel result, the separately scoped steady-step and
natural-data evidence, and the algebra-to-silicon design. See
[`CHUNK_PARALLEL_COMPILER_RESULTS.md`](CHUNK_PARALLEL_COMPILER_RESULTS.md) for
the continuous associative chunk compiler, full-gradient oracle, CUDA
composition benchmark, and the hybrid isotypic-forward/packed-backward
schedule. The latter is the recommended raw training backend and measured
1.54% faster than the packed backend on order-balanced complete steps. See
[`REUSE_ATLAS.md`](REUSE_ATLAS.md) for the repository-wide component audit
and the exact mechanism/claim boundary for every reused subsystem.

The chunk report also records the algebra-selection audit prompted by the
quaternion, Spin(9), and benchmark results. Quaternion conjugation is retained
as the compact `SO(3)` rotation law; center-sensitive quaternion left action
is a valid `Spin(3)` control, but is not representation-equivalent to the
triality state. Spin(9)'s coupled repeated-`V5` block motivates Schur-legal
multiplicity mixing only where equivalent copies are present. Neither result
is converted into an untested language-model claim.

The original three-seed WikiText result is retained as historical evidence
in [`NATURAL_DATA_RESULTS.md`](NATURAL_DATA_RESULTS.md). The pre-reconstruction
frontier backend cut the mean Mamba-2 throughput lead from 4.87x to 1.091x.
That historical timing is not the current result: the post-F14a steady-step
repeats are unresolved, while the current three-seed Shakespeare quality and
peak-memory advantages belong to Mamba-2. Pure Spin's smaller streaming state
remains a design advantage; a complete incremental convolution wrapper is
still open.
