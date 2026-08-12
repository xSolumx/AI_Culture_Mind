# Reproducibility

## Environment

Use Python 3.11 or newer. For the complete historical suite:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[full]"
```

CUDA is optional. Hardware-specific tests skip when their backend is absent.
Exact SymPy certificates and CPU recurrence checks do not require CUDA.

## Test suite

```bash
python -m unittest discover -s tests -p "test_*.py"
python tools/verify_artifact_manifest.py
```

As of 2026-08-06, the current suite passes 188 tests. The recorded full run was
restricted to six logical processors, completed in 375.8 seconds including the
resource supervisor, and peaked at 4.074 GiB of process-tree resident memory.
The edge-theorem unit test is a lightweight
artifact verifier: it reconstructs the stored polynomials and Bernstein arrays,
directly compares both stored coefficient maps, requires complete equality
between stored and freshly recomputed symmetry/divisibility records, and
recomputes all 256 signed holdouts. It deliberately does not rerun the
interpolation grids.

The global five-probe unit test independently regenerates its integral
triality closure, exact generator annihilators, `su(2)` commutators, and
withheld-probe motions. It also checks that modifying a stored rank causes the
verifier to reject the report.

The coordinate-geometry test checks all 52,752 multiview coordinate sensors,
recomputes exact rational Lie ranks for all 141 distinct closures, and verifies
the `SU(3) -> SU(2) -> trivial` representative chain. The generic SchurScan
test independently compares its staged scan and finite homogeneous lift with
sequential recurrence, checks noncommutative irregular-length order and
gradients, exercises a length-2,048 contract, and checks an SO(3) cross-product
control.

## Complex- and quaternionic-type Schur blocks

The division-algebra extension uses exact integer centralizer ranks and
float64 implementation falsifiers. It does not require CUDA.

```powershell
$env:PYTHONPATH = "src"
python -m division_schur_scan `
  --output artifacts/division_schur_scan_20260811.json
python -m pytest tests/test_division_schur_scan.py -q
```

The test checks the complete canonical \(\mathbb C\)- and \(\mathbb H\)-type
commutants, chronological quaternion multiplication, dense-real/factored
agreement, associative scan parity, and gradient parity. It does not claim
automatic decomposition of arbitrary representations or model superiority.
See the [result note](experiments/DIVISION_SCHUR_SCAN_RESULTS.md).

## Exact Schur-type detection

The detector consumes exact rational representation generators. Complete
reducibility is a required logical input; the replay also checks split,
repeated, and missing-assumption rejection controls.

```powershell
$env:PYTHONPATH = "src"
python -m schur_type_detector `
  --output artifacts/schur_type_detection_20260811.json
python -m pytest tests/test_schur_type_detector.py -q
```

The tests reconstruct every displayed multiplication product and repeat all
three positive types after non-orthogonal rational basis changes. See the
[result note](experiments/SCHUR_TYPE_DETECTION_RESULTS.md).

## Exact reducible isotypic decomposition

The reducible compiler uses exact rational commutants, minimal polynomials,
Chinese-remainder idempotents, irreducible Schur certificates, and intertwiner
nullspaces. Its stored artifact includes real, complex, quaternionic, mixed,
rotor, and Spin(9) controls plus refusal gates.

```powershell
$env:PYTHONPATH = "src"
python -m reducible_isotypic_decomposition `
  --output artifacts/reducible_isotypic_decomposition_20260811.json
python -m pytest tests/test_reducible_isotypic_decomposition.py -q
```

The result is exact whenever certified, but the bounded rational idempotent
search is not claimed to terminate for every real decomposition. It refuses
inputs needing unsupported scalar extensions rather than inferring
irreducibility. See the
[result note](experiments/REDUCIBLE_ISOTYPIC_DECOMPOSITION_RESULTS.md).

## Exact algebraic isotypic decomposition

The field-aware path declares the positive real embedding of
\(\mathbb Q(\sqrt2)\), rejects entries outside that field, and uses sparse
exact domain matrices for algebraic nullspaces. The artifact contains a
genuine rational obstruction that splits only after scalar extension, a
negative-square nonsplit control, dense algebraic real/complex/quaternionic
conjugacies, and direct native Spin(9) slice and quotient decompositions.

```powershell
$env:PYTHONPATH = "src"
python -m algebraic_isotypic_decomposition `
  --output artifacts/algebraic_isotypic_decomposition_20260811.json
python -m pytest tests/test_algebraic_isotypic_decomposition.py `
  tests/test_schur_type_detector.py `
  tests/test_reducible_isotypic_decomposition.py -q
```

The old rational code path and artifact serialization remain regression
fixtures. This replay does not claim automatic field discovery, arbitrary
number fields, or noisy decomposition. See the
[design note](ALGEBRAIC_EXTENSION_DESIGN.md) and
[result note](experiments/ALGEBRAIC_ISOTYPIC_DECOMPOSITION_RESULTS.md).

## Clifford signature extension

This exact audit constructs \(\mathrm{Cl}(1,4)\) from five maintained Spin(9)
Clifford involutions, checks all 32 blade images, classifies its volume sectors
and even algebra by exact commutants/intertwiners, embeds
\(\mathrm{Cl}(3,0)\), and repeats the Spin(8) triality Hom table after dense
\(\mathbb Q(\sqrt2)\) conjugacies.

```powershell
$env:PYTHONPATH = "src"
python -m clifford_signature_extension `
  --output artifacts/clifford_signature_extension_20260811.json
python -m pytest tests/test_clifford_signature_extension.py -q
```

The exact audit is intentionally slower than the small compiler controls
because it recomputes several complete commutant and cross-intertwiner spaces.
It is a representation certificate, not a trained-model or determinant-
positivity replay. See the
[theorem narrative](manuscripts/CLIFFORD_SIGNATURE_EXTENSION.md).

## Concrete Spin(9) slice-to-isotypic bridge

The bridge reconstructs the actual Cayley-null normal-slice matrices over
\(\mathbb Q(\sqrt2)\), solves the full exact intertwiner space, aligns the
supported local-Hessian \(\operatorname{Sym}_0(3)\) basis, and only then hands
the rationalized \(V_1\oplus2V_5\) action to the reducible compiler.

```powershell
$env:PYTHONPATH = "src"
python -m spin9_slice_isotypic_bridge `
  --output artifacts/spin9_slice_isotypic_bridge_20260811.json
python -m pytest tests/test_spin9_grassmann_slice.py `
  tests/test_spin9_slice_isotypic_bridge.py `
  tests/test_reducible_isotypic_decomposition.py -q
```

The exact certificate is local to the Cayley-null orbit. It does not certify
the global quotient or the open finite-radius coupled determinant inequality.
See the
[result note](experiments/SPIN9_SLICE_ISOTYPIC_BRIDGE_RESULTS.md).

## Intertwiner SchurScan benchmark

The benchmark records eager tensor-program behavior; it does not claim a fused
production kernel. CPU execution is capped at six threads.

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q `
  tests/test_intertwiner_schurscan.py `
  tests/test_benchmark_intertwiner_schurscan.py

python -m benchmark_intertwiner_schurscan `
  --device cuda --dtype float32 --batch 8 `
  --lengths 16 32 64 128 256 512 1024 2048 4096 `
  --warmup 5 --repeats 15 --backward-max-length 256 `
  --lift-max-length 32 --threads 6 `
  --output artifacts/intertwiner_schurscan_cuda_replay.json
```

Canonical 2026-08-07 hardware results and checksums are listed in
[`INTERTWINER_SCHURSCAN_BENCHMARK_RESULTS.md`](experiments/INTERTWINER_SCHURSCAN_BENCHMARK_RESULTS.md).

## Intertwiner SchurScan equivariant identification

This deterministic float64 gate compares the complete one-dimensional
intertwiner family with unrestricted bilinear, group-augmented bilinear, and
additive fits on identical triangular recurrence endpoints.

```powershell
$env:PYTHONPATH = "src"
python -m intertwiner_schurscan_equivariant_identification `
  --output artifacts/intertwiner_schurscan_equivariant_identification_20260810.json
python -m pytest -q `
  tests/test_intertwiner_schurscan_equivariant_identification.py
```

The frozen protocol, ten-seed Spin(8)/SO(3) results, claim boundary, and
canonical checksum are recorded in
[`INTERTWINER_SCHURSCAN_EQUIVARIANT_IDENTIFICATION_RESULTS.md`](experiments/INTERTWINER_SCHURSCAN_EQUIVARIANT_IDENTIFICATION_RESULTS.md).

## Structured memory-scanner benchmark

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q `
  tests/test_benchmark_schurscan_memory_scanners.py `
  tests/test_spin8_triality_direct_memory_equivalence.py
python -m benchmark_schurscan_memory_scanners `
  --device cuda --dtype float32 --batch 2 `
  --lengths 64 256 1024 2048 4096 `
  --warmup 5 --repeats 15 --threads 6 `
  --output artifacts/schurscan_memory_scanners_cuda_optimized_20260810.json

python -m benchmark_schurscan_memory_scanners `
  --device cpu --dtype float64 --batch 1 `
  --lengths 64 256 1024 2048 `
  --warmup 5 --repeats 15 --threads 6 `
  --output artifacts/schurscan_memory_scanners_cpu_optimized_20260810.json
```

This benchmark compares eager prefix programs for the identical 64-scalar
slot recurrence. It is not a fused modern-delta comparison. Full CPU/CUDA
phase-1 results are preserved in
[`SCHURSCAN_MEMORY_SCANNER_BENCHMARK_RESULTS.md`](experiments/SCHURSCAN_MEMORY_SCANNER_BENCHMARK_RESULTS.md).
The local `9 x 9` homogeneous optimization, reversed-order replications,
whole-workspace method audit, and current hashes are in
[`SCHURSCAN_MEMORY_SCANNER_OPTIMIZATION_RESULTS.md`](experiments/SCHURSCAN_MEMORY_SCANNER_OPTIMIZATION_RESULTS.md).

## Matched learned-retrieval campaign

The quality campaign is float64 and uses CUDA only for training the two matched
address encoders. Evaluation and exact scan parity are canonical CPU operations.
Seeds are written independently so an interrupted cohort retains complete raw
rows.

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q `
  tests/test_schurscan_delta_memory.py `
  tests/test_benchmark_matched_memory_cores.py `
  tests/test_matched_learned_retrieval.py `
  tests/test_matched_retrieval_campaign.py

foreach ($researchSeed in 0..9) {
  python -m matched_learned_retrieval `
    --seeds $researchSeed --device cuda `
    --output "artifacts/matched_learned_retrieval_task_a_seed$researchSeed.json"
}

python -m aggregate_matched_learned_retrieval `
  artifacts/matched_learned_retrieval_task_a_seed0.json `
  artifacts/matched_learned_retrieval_task_a_seed1.json `
  artifacts/matched_learned_retrieval_task_a_seed2.json `
  artifacts/matched_learned_retrieval_task_a_seed3.json `
  artifacts/matched_learned_retrieval_task_a_seed4.json `
  artifacts/matched_learned_retrieval_task_a_seed5.json `
  artifacts/matched_learned_retrieval_task_a_seed6.json `
  artifacts/matched_learned_retrieval_task_a_seed7.json `
  artifacts/matched_learned_retrieval_task_a_seed8.json `
  artifacts/matched_learned_retrieval_task_a_seed9.json `
  --output artifacts/matched_learned_retrieval_task_a_seeds0_9.json

foreach ($run in 1..3) {
  python -m benchmark_matched_memory_cores `
    --device cuda --dtype float32 --batch 2 `
    --lengths 64 256 1024 2048 4096 `
    --warmup 5 --repeats 10 --backward-repeats 5 `
    --replications 5 --tuning-repeats 7 `
    --output "artifacts/matched_memory_cores_tuning_run$run.json"
}

python -m select_matched_memory_core_implementations `
  --inputs artifacts/matched_memory_cores_tuning_run1.json `
    artifacts/matched_memory_cores_tuning_run2.json `
    artifacts/matched_memory_cores_tuning_run3.json `
  --output artifacts/matched_memory_core_frozen_selection.json

foreach ($run in 1..3) {
  python -m benchmark_matched_memory_cores `
    --device cuda --dtype float32 --batch 2 `
    --lengths 64 256 1024 2048 4096 `
    --warmup 5 --repeats 10 --backward-repeats 5 `
    --replications 5 `
    --selection-config artifacts/matched_memory_core_frozen_selection.json `
    --output "artifacts/matched_memory_cores_frozen_run$run.json"
}

python -m aggregate_matched_memory_core_benchmarks `
  --inputs artifacts/matched_memory_cores_frozen_run1.json `
    artifacts/matched_memory_cores_frozen_run2.json `
    artifacts/matched_memory_cores_frozen_run3.json `
  --output artifacts/matched_memory_cores_frozen_aggregate.json

python -m analyze_matched_retrieval_campaign `
  --task-a artifacts/matched_learned_retrieval_task_a_seeds0_9.json `
  --performance artifacts/matched_memory_cores_cuda_rtx2070s_20260810.json `
  --blind-action artifacts/spin8_blind_alias_action_seeds0_9.json `
  --identification artifacts/intertwiner_schurscan_equivariant_identification_20260810.json `
  --output artifacts/matched_retrieval_campaign_synthesis_20260810.json
```

### Task B action replay and prospective close

The first command family is a strict replay attempt. Its frozen failure is an
expected recorded result: the historical source artifact retained metrics but
not the underidentified learned parameters.

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q `
  tests/test_task_b_delta_action_replay.py `
  tests/test_task_b_paired_action_replication.py `
  tests/test_matched_retrieval_campaign.py

foreach ($researchSeed in 0..9) {
  python -m task_b_delta_action_replay `
    --seed $researchSeed `
    --output "artifacts/task_b_delta_action_replay_seed$researchSeed.json"
}

$replayInputs = 0..9 | ForEach-Object {
  "artifacts/task_b_delta_action_replay_seed$_.json"
}
python -m aggregate_task_b_delta_action_replay `
  @replayInputs --expected-seeds 0 1 2 3 4 5 6 7 8 9 `
  --output artifacts/task_b_delta_action_replay_seeds0_9.json

foreach ($researchSeed in 20..29) {
  python -m task_b_paired_action_replication `
    --seed $researchSeed `
    --output "artifacts/task_b_paired_action_replication_seed$researchSeed.json"
}

$replicationInputs = 20..29 | ForEach-Object {
  "artifacts/task_b_paired_action_replication_seed$_.json"
}
python -m aggregate_task_b_paired_action_replication `
  @replicationInputs --expected-seeds 20 21 22 23 24 25 26 27 28 29 `
  --output artifacts/task_b_paired_action_replication_seeds20_29.json

python -m task_b_paired_action_replication `
  --verify artifacts/task_b_paired_action_replication_seed20.json

python -m analyze_matched_retrieval_campaign `
  --task-a artifacts/matched_learned_retrieval_task_a_seeds0_9.json `
  --performance artifacts/matched_memory_cores_cuda_rtx2070s_20260810.json `
  --blind-action artifacts/spin8_blind_alias_action_seeds0_9.json `
  --identification artifacts/intertwiner_schurscan_equivariant_identification_20260810.json `
  --task-b-replication artifacts/task_b_paired_action_replication_seeds20_29.json `
  --output artifacts/matched_retrieval_campaign_synthesis_task_b_closed_20260810.json
```

The prospective seeds `20`--`29` cohort retains both learned action families,
router weights, coordinates, and training reports. It passes the frozen
representation-prior decision `10/10`, exact direct/delta parity `10/10`, and
retained-parameter verification `10/10`. See
[`TASK_B_DELTA_ACTION_REPLAY_RESULTS.md`](experiments/TASK_B_DELTA_ACTION_REPLAY_RESULTS.md)
and
[`TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md`](experiments/TASK_B_PAIRED_ACTION_REPLICATION_RESULTS.md).

The official fused tier requires Linux, PyTorch >=2.7, and Triton >=3.3. The
maintained run used an isolated WSL environment:

```bash
python3 -m venv ~/.venvs/schurscan-fla
~/.venvs/schurscan-fla/bin/python -m pip install \
  'flash-linear-attention[cuda]==0.5.2'

for run in 1 2 3; do
  PYTHONPATH=src ~/.venvs/schurscan-fla/bin/python \
    src/benchmark_fla_delta_rule.py \
    --batch 2 --lengths 64 256 1024 2048 4096 \
    --warmup 5 --repeats 10 --backward-repeats 5 \
    --timing-blocks 5 --tuning-repeats 7 \
    --output "artifacts/fla_delta_tuning_run${run}.json"
done

PYTHONPATH=src python src/select_fla_delta_rule_implementations.py \
  --inputs artifacts/fla_delta_tuning_run1.json \
    artifacts/fla_delta_tuning_run2.json \
    artifacts/fla_delta_tuning_run3.json \
  --output artifacts/fla_delta_frozen_selection.json

for run in 1 2 3; do
  PYTHONPATH=src ~/.venvs/schurscan-fla/bin/python \
    src/benchmark_fla_delta_rule.py \
    --batch 2 --lengths 64 256 1024 2048 4096 \
    --warmup 5 --repeats 10 --backward-repeats 5 --timing-blocks 5 \
    --selection-config artifacts/fla_delta_frozen_selection.json \
    --output "artifacts/fla_delta_frozen_run${run}.json"
done

PYTHONPATH=src python src/aggregate_fla_delta_rule_benchmarks.py \
  --inputs artifacts/fla_delta_frozen_run1.json \
    artifacts/fla_delta_frozen_run2.json \
    artifacts/fla_delta_frozen_run3.json \
  --output artifacts/fla_delta_frozen_aggregate.json
```

## Spin(9) full-\(V_5\) determinant certificate

The reconstruction stage recovers all 631 invariant numerator coefficients
and checks them at an unused prime. The certificate stage performs the
twenty-prime, two-embedding characteristic-zero lift and the strict six-cell
Bernstein atlas:

```powershell
$env:PYTHONPATH = "src"
python -m spin9_v5_cartan_reconstruction `
  --output artifacts/spin9_v5_cartan_reconstruction_20260811.json
python -m spin9_v5_cartan_certificate `
  --output artifacts/spin9_v5_cartan_certificate_20260811.json
python -m pytest tests/test_spin9_v5_cartan_certificate.py -q
```

The full identity replay performs 347,200 modular \(36\times36\) determinant
evaluations. It took about eight minutes on the recorded Windows workstation;
the reconstruction alone took about forty seconds. The certificate is exact,
not a floating-point screen. Its theorem domain is every pure \(V_5\) graph
over the Cayley-null plane, not the coupled \(V_1\oplus V_5\) slice or global
Grassmann quotient. See
[the theorem note](manuscripts/SPIN9_V5_CARTAN_CERTIFICATE.md).

## Spin(9) coupled \(V_1\oplus V_5\) reconstruction

The next normal-slice layer reconstructs 18,600 weighted-degree-84
coefficients. Four workers parallelize independent prime fields; CRT recovery
and the unused-prime conjugate checks remain deterministic:

```powershell
$env:PYTHONPATH = "src"
python -m spin9_v1_v5_reconstruction `
  --workers 4 --quiet `
  --output artifacts/spin9_v1_v5_reconstruction_20260811.json
python -m spin9_v1_v5_screen `
  --output artifacts/spin9_v1_v5_screen_20260811.json
python -m spin9_v1_v5_boundary_char0 `
  --workers 4 `
  --output artifacts/spin9_v1_v5_boundary_char0_20260811.json
python -m spin9_v1_v5_blowup `
  --output artifacts/spin9_v1_v5_blowup_20260811.json
# Generate the two compact and eight required local reports exactly as listed
# in docs/manuscripts/SPIN9_V1_V5_RECONSTRUCTION.md, then assemble them:
python -m spin9_v1_v5_global `
  --assemble-report-dir runtime/spin9-v1-v5-global `
  --output artifacts/spin9_v1_v5_global_20260812.json
python -m spin9_v1_v5_char0 `
  --workers 4 `
  --output artifacts/spin9_v1_v5_char0_20260812.json
python -m spin9_v1_v5_theorem `
  --output artifacts/spin9_v1_v5_theorem_20260812.json
python -m pytest tests/test_spin9_v1_v5_reconstruction.py `
  tests/test_spin9_v1_v5_boundary_char0.py `
  tests/test_spin9_v1_v5_blowup.py `
  tests/test_spin9_v1_v5_global.py `
  tests/test_spin9_v1_v5_char0.py `
  tests/test_spin9_v1_v5_theorem.py -q
```

The recorded reconstruction took 394 seconds. The test replays the unused
prime under both square roots of \(2\), so it takes roughly two and a half
minutes. The artifact proves a modular reconstruction and exact agreement with
the pure-\(V_1\) and pure-\(V_5\) boundaries. The raw boundary command constructs
the limiting information matrices directly over \(\mathbb Q(\sqrt2)\) and checks
2,701 degree-determining Newton nodes per family; four workers took 176 seconds.
It does not load the modular coefficient table. The blow-up command then
extracts both degree-28 rational functions, replays strict Bernstein atlases on
all eight sign charts in roughly nine seconds, and chains them to the raw
identities. Together they prove both complete blow-up planes below \(26/25\).
The promoted follow-on retains every failed uniform-atlas cell and proves an
exact handoff to family-A or family-B core/low-\(q\) charts. Its compact layer
has 312 positive leaves and eight handoff boxes; the eight local charts have
29 positive leaves. The separate 22-prime lift checks 13,914,692 raw
determinants under both square-root embeddings and uses a 175-digit prime
product exceeding twice the exact residual coefficient bound. The combined
artifact proves the \(21/20\) bound on the complete coupled finite-radius
slice. It does not prove exact candidate optimality or the unrestricted
quotient. See the
[reconstruction note](manuscripts/SPIN9_V1_V5_RECONSTRUCTION.md).
The screen is a deterministic float64 falsifier, not part of the exact proof
layer.

## Hierarchical routing, Spin(9), and transported FLA

The same-router quality campaign is CPU-canonical after the existing
384-parameter slot router is trained. Write each seed in its own process:

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests `
  -p "test_hierarchical_matched_retrieval.py" -v
python -m unittest discover -s tests `
  -p "test_schurscan_comoving_delta.py" -v
python -m spin9_clifford_memory `
  --output artifacts/spin9_clifford_memory_boundary_20260810.json

foreach ($researchSeed in 0..9) {
  python -m hierarchical_matched_retrieval `
    --seeds $researchSeed `
    --output "artifacts/hierarchical_matched_retrieval_seed$researchSeed.json"
}

$hierarchicalInputs = 0..9 | ForEach-Object {
  "artifacts/hierarchical_matched_retrieval_seed$_.json"
}
python -m aggregate_hierarchical_matched_retrieval `
  @hierarchicalInputs --expected-seeds 0 1 2 3 4 5 6 7 8 9 `
  --output artifacts/hierarchical_matched_retrieval_seeds0_9.json
```

The full transported fused tier uses the isolated WSL FLA environment. Three
tuning processes are aggregated before any measurement process starts:

```bash
for run in 1 2 3; do
  PYTHONPATH=src ~/.venvs/schurscan-fla/bin/python \
    src/benchmark_comoving_fla_delta_rule.py \
    --batch 16 --lengths 256 1024 4096 \
    --warmup 6 --repeats 1 --backward-repeats 1 \
    --timing-blocks 1 --tuning-repeats 5 \
    --output "artifacts/comoving_fla_tuning_run${run}_20260810.json"
done

PYTHONPATH=src ~/.venvs/schurscan-fla/bin/python \
  src/select_comoving_fla_implementations.py \
  --inputs artifacts/comoving_fla_tuning_run1_20260810.json \
    artifacts/comoving_fla_tuning_run2_20260810.json \
    artifacts/comoving_fla_tuning_run3_20260810.json \
  --output artifacts/comoving_fla_frozen_selection_rtx2070s_20260810.json

for run in 1 2 3; do
  PYTHONPATH=src ~/.venvs/schurscan-fla/bin/python \
    src/benchmark_comoving_fla_delta_rule.py \
    --batch 16 --lengths 256 1024 4096 \
    --warmup 20 --repeats 100 --backward-repeats 100 \
    --timing-blocks 1 --tuning-repeats 0 \
    --selection-config \
      artifacts/comoving_fla_frozen_selection_rtx2070s_20260810.json \
    --output "artifacts/comoving_fla_frozen_run${run}_20260810.json"
done

PYTHONPATH=src ~/.venvs/schurscan-fla/bin/python \
  src/aggregate_comoving_fla_benchmarks.py \
  --inputs artifacts/comoving_fla_frozen_run1_20260810.json \
    artifacts/comoving_fla_frozen_run2_20260810.json \
    artifacts/comoving_fla_frozen_run3_20260810.json \
  --output artifacts/comoving_fla_frozen_aggregate_20260810.json
```

The algebra, routing failure frontier, timing statistics, state mismatch, and
interpretation boundary are recorded in
[`SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md).

## Large-slot semantic hierarchy and fused physical gather

The later quality cohort is CPU `float64`. Run the development seed separately;
only frozen seeds `30`--`39` enter the aggregate:

```powershell
$env:PYTHONPATH = "src"
python src/large_slot_semantic_hierarchy.py --seed 103 `
  --output artifacts/large_slot_semantic_hierarchy_dev103.json

foreach ($researchSeed in 30..39) {
  python src/large_slot_semantic_hierarchy.py `
    --seed $researchSeed `
    --output "artifacts/large_slot_semantic_hierarchy_seed$researchSeed.json"
}

$semanticInputs = 30..39 | ForEach-Object {
  "artifacts/large_slot_semantic_hierarchy_seed$_.json"
}
python src/aggregate_large_slot_semantic_hierarchy.py `
  @semanticInputs --expected-seeds 30 31 32 33 34 35 36 37 38 39 `
  --output artifacts/large_slot_semantic_hierarchy_seeds30_39.json

python src/large_slot_semantic_hierarchy.py `
  --verify artifacts/large_slot_semantic_hierarchy_seed30.json
```

The eager CUDA gather uses the system CUDA-enabled PyTorch environment. Each
measurement process uses the frozen default grid and retains 25 raw timing
blocks per variant:

```powershell
foreach ($run in 1..3) {
  python src/benchmark_gathered_block_memory.py `
    --device cuda `
    --output "artifacts/gathered_block_memory_cuda_run${run}_20260810.json"
}
python src/aggregate_gathered_block_memory_benchmarks.py `
  artifacts/gathered_block_memory_cuda_run1_20260810.json `
  artifacts/gathered_block_memory_cuda_run2_20260810.json `
  artifacts/gathered_block_memory_cuda_run3_20260810.json `
  --output artifacts/gathered_block_memory_cuda_aggregate_20260810.json
```

The fused inference tier additionally requires the exact optional dependency
used by the frozen protocol:

```powershell
python -m pip install triton-windows==3.7.1.post27
$env:PYTHONPATH = "src"

foreach ($run in 1..3) {
  python src/benchmark_fused_gathered_block_memory.py `
    --output "artifacts/fused_gathered_block_memory_cuda_run${run}_20260810.json"
}
python src/aggregate_fused_gathered_block_memory_benchmarks.py `
  artifacts/fused_gathered_block_memory_cuda_run1_20260810.json `
  artifacts/fused_gathered_block_memory_cuda_run2_20260810.json `
  artifacts/fused_gathered_block_memory_cuda_run3_20260810.json `
  --output artifacts/fused_gathered_block_memory_cuda_aggregate_20260810.json

python src/fused_gathered_recurrent_diagnostic.py `
  --output artifacts/fused_gathered_block_memory_recurrent_diagnostic_20260810.json
```

Run the classified implementation/aggregation tests with:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q `
  tests/test_large_slot_semantic_hierarchy.py `
  tests/test_aggregate_large_slot_semantic_hierarchy.py `
  tests/test_benchmark_gathered_block_memory.py `
  tests/test_aggregate_gathered_block_memory_benchmarks.py `
  tests/test_benchmark_fused_gathered_block_memory.py `
  tests/test_aggregate_fused_gathered_block_memory_benchmarks.py
```

The fused test skips cleanly when CUDA or `triton-windows` is unavailable; a
CPU-only skip is not a fused-kernel replay. Exact metrics and limitations are in
[`LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md`](experiments/LARGE_SLOT_SEMANTIC_HIERARCHY_RESULTS.md).

The frozen cells, paired verdicts, and CUDA timing boundary are recorded in
[`MATCHED_LEARNED_RETRIEVAL_RESULTS.md`](experiments/MATCHED_LEARNED_RETRIEVAL_RESULTS.md).
The formerly open Task B delta-action row is closed prospectively by the paired
replication linked above; the failed strict historical replay remains preserved
as a separate result.

The continuous-orbit test recomputes exact invariant tangent ranks, action
ranks, stabilizer brackets/Killing forms, and one globally free closure for
every mixed five-probe allocation. The compact principal-orbit theorem is the
mathematical inference layer connecting those exact certificates to the global
generic and universal claims.

## Exact Dirac-star replay

```bash
python -m spin8_dirac_star \
  --output artifacts/spin8_dirac_star_replay.json
```

The replay is intentionally expensive. It reconstructs two exact rational
coefficient maps, compares them, evaluates exact Bernstein coefficients, and
checks 32 signed off-grid determinants.

Acceptance conditions are frozen in
[SPIN8_DIRAC_STAR_PREREGISTRATION.md](experiments/SPIN8_DIRAC_STAR_PREREGISTRATION.md).

## Publication theorem extensions

The Cayley design-criterion laws, the forced-factor reduction of the signed
star certificate, and an independent FLINT arithmetic replay are reproduced
with:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_cayley_blocks
python -m spin8_cayley_flag `
  --output artifacts/spin8_cayley_flag_replay.json
python -m spin8_cayley_criteria `
  --output artifacts/spin8_cayley_criteria_replay.json
python -m spin8_dirac_star_structure `
  --output artifacts/spin8_dirac_star_structure_replay.json
python -m spin8_dirac_star_foundations `
  --output artifacts/spin8_dirac_star_foundations_replay.json
python -m spin8_publication_flint_crosscheck `
  --threads 6 `
  --output artifacts/spin8_publication_flint_crosscheck_replay.json
python -m unittest discover -s tests `
  -p "test_spin8_publication_theorems.py" -v
```

The FLINT pass repeats the rational polynomial divisions, derivative
identities, endpoint eigenvalue slopes, and all 1,907 reduced Bernstein
coefficients. It deliberately accepts the maintained rational coefficient
maps as input; the full star replay above remains the independent check that
regenerates those maps from exact determinant samples.

## Exact Cayley-null edge-family replay

```bash
python -m spin8_dirac_edge \
  --output artifacts/spin8_dirac_edge_replay.json
```

This proof-bearing replay constructs the symbolic boundary-nullspace
certificate, derives the Walsh symmetry restriction, reconstructs two exact
coefficient maps on disjoint five-node grids, checks all 256 off-grid signed
determinants, and verifies the native Bernstein certificate. Acceptance
conditions are recorded in
[SPIN8_DIRAC_EDGE_PREREGISTRATION.md](experiments/SPIN8_DIRAC_EDGE_PREREGISTRATION.md).

The exact conditional-decorrelation counterexample has a faster replay:

```bash
python -m spin8_conditional_counterexample \
  --output artifacts/spin8_conditional_counterexample_replay.json
```

## Hardware-tuned variable-Cayley determinant replay

The final one-edge determinant certificate is intentionally staged so the
large SymPy polynomial and the million-control integer tensors never coexist.
On the reference workstation (8-core i7-9700K, 24 GB RAM, RTX 2070 SUPER), use
FLINT-backed exact arithmetic for the CPU stages. CUDA is used only for the
separate falsifier and never supplies proof signs.

```powershell
$env:SYMPY_GROUND_TYPES = "flint"
$env:PYTHONPATH = "src"
python -m spin8_dirac_one_edge_positivity determinant `
  --reconstruction artifacts/spin8_dirac_one_edge_exact_20260804.json `
  --output build/one_edge_positivity/determinant.json
python -m spin8_dirac_one_edge_positivity lower `
  --cache build/one_edge_positivity/determinant.json `
  --output build/one_edge_positivity/lower.json
python -m spin8_dirac_one_edge_positivity upper `
  --cache build/one_edge_positivity/determinant.json `
  --output build/one_edge_positivity/upper.json
python -m spin8_dirac_one_edge_positivity boundary `
  --reconstruction artifacts/spin8_dirac_one_edge_exact_20260804.json `
  --output build/one_edge_positivity/boundary.json
python -m spin8_dirac_one_edge_holdouts `
  --reconstruction artifacts/spin8_dirac_one_edge_exact_20260804.json `
  --workers 4 `
  --output build/one_edge_positivity/holdouts.json
python -m spin8_dirac_one_edge_positivity assemble `
  --reconstruction artifacts/spin8_dirac_one_edge_exact_20260804.json `
  --cache build/one_edge_positivity/determinant.json `
  --lower build/one_edge_positivity/lower.json `
  --upper build/one_edge_positivity/upper.json `
  --boundary build/one_edge_positivity/boundary.json `
  --lower-order artifacts/spin8_dirac_one_edge_positivity_20260804.json `
  --holdouts build/one_edge_positivity/holdouts.json `
  --output artifacts/spin8_dirac_one_edge_positivity_replay.json
```

Install `python-flint` to enable the optimized exact backend. The committed
2026-08-06 artifact records the completed exact theorem. The 10 MB published
determinant cache may replace the first stage when replaying later stages, but
its SHA-256 link to the reconstruction must still pass.

The complete equality-set audit reuses those proof objects and reconstructs
the exact zero-control supports under the same six-core limit:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_dirac_one_edge_equality `
  --reconstruction artifacts/spin8_dirac_one_edge_exact_20260804.json `
  --cache artifacts/spin8_dirac_one_edge_determinant_cache_20260806.json `
  --assembled artifacts/spin8_dirac_one_edge_duffy_20260806.json `
  --workers 6 `
  --output artifacts/spin8_dirac_one_edge_equality_replay.json
```

The exact two-edge endpoint-jet flag law is lightweight to replay from the
published sector maps:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_dirac_two_edge_endpoints `
  --coefficients artifacts/spin8_dirac_two_edge_all_sectors_coefficients_20260806.json `
  --output artifacts/spin8_dirac_two_edge_endpoints_replay.json
```

### Complete endpoint-octet cubic theorem

The final cubic theorem has two replay tiers. The compact checks verify source
hashes, delegated certificates, atlas structure, and stored exact Bernstein
summaries. Add `--recompute-identity` to reconstruct the 1,546,277-term cubic,
the 2,824,946-term quotient, the forced-radical cancellation, and the final
zero-remainder identity in characteristic zero:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_dirac_endpoint_octet_cubic_coarse_atlas `
  --verify artifacts/spin8_dirac_endpoint_octet_cubic_coarse_atlas_20260811.json
python -m spin8_dirac_endpoint_octet_cubic_certificate `
  --verify artifacts/spin8_dirac_endpoint_octet_cubic_certificate_20260811.json `
  --recompute-identity
```

The full quotient replay recomputes every expensive Bernstein transform in the
complete 32-box cover. Boxes `00001` and `00010` are delegated only to their
independent hash-bound exact certificates:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_dirac_endpoint_octet_cubic_coarse_atlas `
  --flint-threads 6 `
  --output artifacts/spin8_dirac_endpoint_octet_cubic_coarse_atlas_replay.json
python -m spin8_dirac_endpoint_octet_cubic_certificate `
  --coarse-report artifacts/spin8_dirac_endpoint_octet_cubic_coarse_atlas_replay.json `
  --output artifacts/spin8_dirac_endpoint_octet_cubic_certificate_replay.json
```

The theorem above covers the cubic principal minor only. Later exact work
reconstructs the fourth-order determinant, proves its \(y=0\) face by a
31-leaf dyadic atlas, and proves its \(y=1\) face from \(Z=X^2\). Run the full
reconstruction and endpoint replay with:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_dirac_endpoint_octet_determinant `
  --flint-threads 6 `
  --output runtime/spin8_dirac_endpoint_octet_determinant_replay.json
python -m spin8_dirac_endpoint_octet_determinant_tangent `
  --flint-threads 6 `
  --output runtime/spin8_dirac_endpoint_octet_determinant_tangent_replay.json
python -m pytest tests/test_spin8_endpoint_octet_determinant.py -q
```

The source command recomputes the 6,082,148-term determinant and all 31 exact
leaf transforms. The test is a compact replay that checks the generic
determinant identity, dependency hashes, complete covers, stored leaf sign
counts, the exact selector rejection witness, and the order-eight squared
tangent certificate. Positivity for \(0<y<1\) remains open; the tangent command
proves only the local exceptional divisor and rejects only the named
degree-matched selector route.

### Endpoint-octet twenty-output replay

Nineteen historical `2026-08-10` watchdog records describe the expensive
blow-up, selected-boundary, boundary-atlas, and failed-symmetry routes behind
the finite endpoint-octet cubic certificate. They are operational evidence,
so they live under `runtime/endpoint-octet/2026-08-10/` rather than the
published `artifacts/` directory. The twentieth formerly unpromoted output is
the complete 32-child `00001` mathematical atlas.

The replay campaign reconstructs all twenty commands and writes fresh outputs
to an isolated runtime tree. It reports strict byte equality separately from
exact mathematical-payload equality. The latter removes only the allow-listed
provenance keys `audit_engine` and `batch_entry_limit` from both JSON trees,
and applies conditional legacy defaults: it may omit `parent_path` and
`post_zero_path` only when their value is the empty list `[]`, and it may omit
`selected_zero_face` only when its value is `null`. Nonempty paths and
substantive face selectors remain comparison failures. Two frozen string
aliases cover the historical
UI-face label and its corresponding nonclaim sentence; only their exact old
and current values are equated. Arbitrary scope or label changes remain
failures. The normalized trees must then agree exactly in every coefficient,
degree, sign count, route, and theorem field. New elapsed-time, RSS, and
return-code records are retained but are not expected to equal historical
measurements. The campaign is checkpointed and may be resumed after
interruption.

```powershell
$env:PYTHONPATH = "src"
python tools/replay_endpoint_octet_runtime_artifacts.py `
  --jobs 2 --workers 6 --memory-gib-per-job 7.4
```

Two concurrent jobs remain below a declared aggregate 14.8-GiB cap. Use
`--jobs 1 --memory-gib-per-job 14.8` when stronger per-process headroom is
preferred. Exact mathematical-payload equality, not runtime reproducibility,
is the acceptance gate; any byte-only schema drift remains a separately
reported negative reproducibility result.

The completed 20-row receipt, exact/byte split, resource measurements, and
historical return-code-120 diagnosis are recorded in the dated
[`Spin(8) endpoint-octet twenty-output replay report`](experiments/SPIN8_ENDPOINT_OCTET_RUNTIME_REPLAY_20260811.md).

### Enforced workstation envelope

For any expensive stage on the reference i7-9700K, use the bounded runner:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_resource_limits --workers 6 --memory-gib 15 -- `
  python -m <module> <arguments>
```

This pins the complete process tree to six logical cores, caps common native
thread pools, selects the FLINT SymPy ground domain, records peak process-tree
RSS, and terminates the stage at 15 GiB. The one-GiB margin keeps the symbolic
process below the requested 16-GiB ceiling despite watchdog sampling latency.

`spin8_flint_crosscheck.py` is the independent arithmetic check. Merely setting
`SYMPY_GROUND_TYPES=flint` improves speed but is not counted as independent
verification.

## Memory benchmark figures

The public-facing atlas is derived from frozen aggregate artifacts and does
not rerun timing or training. Install the optional renderer and regenerate PNG
and SVG outputs with:

```powershell
python -m pip install -e ".[plots]"
python src/plot_memory_benchmark_atlas.py
```

The generator validates the source gates before rendering and writes
`docs/figures/memory_benchmark_atlas/figure_manifest.json` with SHA-256 hashes
for every input artifact and output figure. The explanatory document is
[`MEMORY_BENCHMARK_ATLAS.md`](experiments/MEMORY_BENCHMARK_ATLAS.md).

## Artifact integrity

`ARTIFACTS.sha256` contains one SHA-256 entry per published JSON artifact.
On Linux/macOS:

```bash
sha256sum --check ARTIFACTS.sha256
```

On PowerShell, compare `Get-FileHash -Algorithm SHA256` with the manifest.

`PROVENANCE.json` records the original path, extracted destination, byte size,
and SHA-256 hash for every source-derived file.

## Determinism caveat

Exact rational certificates are deterministic. Some CUDA training experiments
are not bitwise deterministic across devices or PyTorch/CUDA releases. The
documents therefore report dense distributions, per-seed results, gate
hierarchies, and raw artifacts rather than silently treating a seed label as a
bitwise-reproducible checkpoint.

## Deliberate exclusions

The public repository does not include:

- virtual environments and caches;
- transient stdout/stderr logs;
- model and compiler checkpoints (`.pt`, `.msgpack`);
- the unrelated historical 44.8 MB language-model checkpoint;
- unrelated applications from the parent monorepo.

These exclusions remove generated payloads, not scientific claims. Frozen JSON
outputs, preregistrations, corrections, negative results, and theorem
certificates are retained.
