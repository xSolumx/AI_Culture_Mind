# Source index

The source remains flat because the historical harnesses import one another by
module name. Editable installation adds this directory to the Python path.

## Maintained Spin(8) algebra and theorem core

- `spin8_triality.py`: vector and chiral representations, Lie algebra checks,
  triality tensor, affine scan primitives.
- `spin8_triality_lift.py`: triangular triality recurrence and binding.
- `spin8_triality_memory.py`: multiplicity-slot memory contracts.
- `spin8_triality_direct_memory_equivalence.py`: exact dynamic gauge
  equivalence between supplied-key triality-bound and direct addressed slots.
- `spin8_triality_2a5_closure.py`: exact quadratic-field and faithful
  permutation classification of the three common-carrier binary-icosahedral
  triality views, giving a reducible perfect group of order 864,000 with
  independent block-sign center.
- `mixed_monomial_golden_closure.py`: exact algebraic-integer obstruction,
  symmetric length-two minimality audit, and irreducible `7+21` Clifford-
  adjoint certificate proving that adjoining the maintained monomial group to
  any golden triality view gives a topologically dense subgroup of `SO(8)`.
- `mixed_monomial_golden_mixing.py`: exact Sturm and quadratic-field LDL
  certificates for the defining `8`, adjoint `28`, and traceless-symmetric
  `35` contraction gaps of the three fixed symmetric generator measures; this
  is a finite representation band, not a full `L2(SO(8))` spectral gap.
- `mixed_monomial_golden_higher_weight.py`: exact `Lambda^3` and Hodge-split
  `Lambda^4` continuation, the unique monomial-fixed Cayley-form line, and a
  certified `N-H-N` compiled macro distribution that improves the worst
  six-representation gap per macro-step while disclosing its three-letter cost.
- `mixed_monomial_golden_macro_compiler.py`: exact deduplication, multiplicity,
  inverse-symmetry, storage, and weighted-mean contract for the finite `N-H-N`
  matrix dictionaries.
- `benchmark_mixed_monomial_golden_macro.py`: single-thread CPU and synchronized
  CUDA comparison of online three-letter construction against deduplicated and
  direct-labelled compiled lookup; this is empirical workstation evidence.
- `mixed_monomial_golden_chunk_compiler.py`: exact `24x8` stacked local-prefix
  tables emitting all three causal states of every labelled `N-H-N` chunk.
- `benchmark_mixed_monomial_golden_chunk.py`: recurrent endpoint and every-
  prefix comparison through 192 primitive steps on single-thread CPU and
  synchronized CUDA; parallel scan and backward remain separate frontiers.
- `mixed_monomial_golden_parallel_chunk_scan.py`: differentiable maintained-
  order work-efficient and Hillis--Steele controls plus the two-stage compiled
  endpoint scan and parallel local-prefix expansion.
- `benchmark_mixed_monomial_golden_parallel_chunk_scan.py`: eager PyTorch
  forward and initial-state-backward timing/parity across CPU and CUDA; this
  remains the parallel semantic/performance control.
- `mixed_monomial_golden_triton_local_prefix.py`: fused exact-table indexing
  and `24x8` local expansion with incoming-state transpose-matvec backward and
  safe eager fallback for trainable tables.
- `benchmark_mixed_monomial_golden_triton_local_prefix.py`: synchronized CUDA
  comparison against realistic indexed and optimistic pre-gathered eager
  controls, both locally and after the endpoint scan.
- `mixed_monomial_golden_triton_chunk_recurrence.py`: one-program-per-sequence
  register-resident frozen-dictionary recurrence with custom initial-state
  reverse pass; serial in chunk depth and therefore not a parallel scan.
- `benchmark_mixed_monomial_golden_triton_chunk_recurrence.py`: five-way CUDA
  comparison of the fused recurrence, eager recurrence, and parallel controls.
- `spin8_five_probe_identifiability.py`: exact shared-rank and ambiguity gates.
- `spin8_global_probe_certificate.py`: exact integral triality closure proving
  one global five-probe free tuple and a four-probe `su(2)` counterfamily.
- `spin8_coordinate_geometry.py`: exhaustive `F_2^5` classification of all
  coordinate four/five-probe sensors and their exact stabilizer ladder.
- `octonion_operator_groups.py`: exact signed-permutation closure identifying
  the associative left-operator group `2_+^(1+6)`, the non-split signed Fano
  automorphism group `2^3.PSL(2,7)`, and their split perfect
  `2_+^(1+6):PSL(2,7)` extension of order 21,504.
- `spin8_continuous_probe_orbits.py`: invariant/principal-orbit certificate
  proving universal four-probe insufficiency and generic mixed five-probe
  global identifiability.
- `intertwiner_schurscan.py`: generic triangular bilinear scan, finite
  homogeneous lift, ordered linear-work scan, SO(3) cross-product tensor, and
  cyclic-feedback degree obstruction.
- `benchmark_intertwiner_schurscan.py`: correctness-qualified eager CPU/CUDA
  benchmark separating composition work, dependency depth, memory, and wall
  time. It is not a fused-kernel benchmark.
- `intertwiner_schurscan_equivariant_identification.py`: deterministic
  Spin(8)/SO(3) endpoint-identification gate comparing a known intertwiner,
  unrestricted bilinear fits, group augmentation, and an additive control. Its
  SO(3) cross-product row is an experiment control, not a standalone model.
- `benchmark_schurscan_memory_scanners.py`: correctness-qualified eager
  benchmark for structured slot Hillis/work-efficient scans, block-preserving
  local `9 x 9` homogeneous scans, and exact dense affine/homogeneous compilers
  of the same 64-scalar recurrence.
- `spin8_active_sensing.py`: information operator and sensor metrics.
- `spin8_joint_sensor_retraction.py`: joint sensor-family continuation.
- `spin8_cayley_spectrum.py`: exact Cayley spectrum certificates.
- `spin8_cayley_blocks.py`: exact `8 + 8 + 8 + 4` invariant-block mechanism
  behind the balanced Cayley characteristic law and determinant `81/1024`.
- `spin8_five_query_local_geometry.py`: exact 35-dimensional Riemannian
  Hessian, shared-orbit kernel, finite great-circle atlas, and boundary ranks
  for the balanced equal-five-query sensor.
- `spin8_approximate_design_audit.py`: exact separation of equal-five and
  weighted approximate D-optimality, including a rational reweighting
  counterexample and the isotropic global approximate optimum.
- `spin8_d4_24cell_bridge.py`: exact minuscule-weight 24-cell bridge, coloured
  projector geometry, continuous non-vertex tight-frame deformation, and
  spectral packing falsifier plus chordal-bound audit.
- `spin8_flint_crosscheck.py`: independent native-FLINT replay of central
  SymPy matrix and polynomial arithmetic.
- `spin8_resource_limits.py`: six-core affinity, native-thread caps, and a
  process-tree RAM watchdog for staged exact work.
- `spin8_gpu_design_audit.py`, `spin8_gpu_design_cohort.py`: CUDA dense
  interior search, sensitivity mapping, reweighting, and noise profiling;
  numerical falsification only.
- `spin8_dirac_gram.py`: projector geometry, Schur reduction, and falsifiers.
- `spin8_dirac_star.py`: independently replayed rational Bernstein theorem.
- `spin8_conditional_counterexample.py`: exact rational falsifier for naive
  Cholesky decorrelation.
- `spin8_dirac_edge.py`: exact Cayley-null four-correlation theorem with
  symbolic degree, symmetry, and Bernstein certificates.
- `spin8_dirac_one_edge.py`: variable-Cayley one-edge falsifier and exact
  four-sector Walsh audit.
- `spin8_dirac_one_edge_exact.py`: disjoint-grid reconstruction and
  tetrahedral principal-minor certificate utilities.
- `spin8_dirac_one_edge_holdouts.py`: all 256 exact disjoint determinant
  holdouts and a lightweight stored-artifact verifier.
- `spin8_dirac_one_edge_positivity.py`: staged, crash-resilient integer
  Bernstein/Duffy proof of the final determinant gate.
- `spin8_dirac_endpoint_octet_determinant.py`: exact radical-free fourth-order
  Schur determinant reconstruction and both endpoint-face certificates.
- `spin8_dirac_endpoint_octet_determinant_tangent.py`: exact descent
  `t=y^2`, rational rejection of the degree-matched endpoint-selector route,
  and the order-eight squared tangent theorem at the remaining equality corner.
- `spin8_dirac_two_edge.py`: exact common-symmetry and two-anchor sector audit
  for the preregistered `h=0`, residual-`i` bridge.
- `spin8_dirac_two_edge_attack.py`: uniform, boundary-biased, and optimized
  CUDA falsifier for that bridge; never used for proof signs.
- `spin8_dirac_two_edge_degree.py`: 2,736-determinant exact multi-slice degree
  and endpoint-factor audit with disjoint confirmation nodes.
- `spin8_dirac_two_edge_amplitude.py`: both-branch rank and quotient-ring proof
  of the common `(1-c^2)^3` factor in all eight two-edge sectors.
- `spin8_dirac_two_edge_reconstruct.py`: crash-safe exact tensor reconstruction,
  disjoint-grid comparison, holdout replay, and factor certification for the
  first complete two-edge Walsh sector.
- `spin8_dirac_two_edge_shared_reconstruct.py`: preregistered common-grid
  campaign that reuses each exact determinant across all eight Walsh sectors;
  verifies two independent complete coefficient maps, exact holdouts, endpoint
  factors, the one-edge flag bridge, and the even/odd Hadamard block split.
- `spin8_dirac_two_edge_kernel.py`: four-channel boundary-kernel reduction,
  exact local jet and endpoint certificate, exact rejection of the naive
  global quadratic Schur strategy, and float64 CUDA face falsifier.
- `spin8_dirac_two_edge_kernel_flint.py`: independent FLINT replay of the
  rational jet slices and the singular-block factorization.
- `spin8_dirac_two_edge_finite.py`: exact one-squaring radical elimination of
  every finite second-edge margin pair into degree-six and degree-twelve
  polynomial gates, plus a full-six-cube CUDA falsifier.
- `spin8_multiplicity_gauge.py`: exact covariance and `O(m)` gauge theorem for
  repeated probes in the same triality representation.

## Spin(9) Dirac--Clifford sensing

- `spin9_dirac_clifford.py`: nine-involution Clifford system, Hopf identities,
  and exact one/two/three-probe rank witnesses.
- `spin9_three_spinor_conditioning.py`: exact symmetric-curve spectrum,
  determinant, feasible interval, and algebraic curve optimum.
- `spin9_frame_operator.py`: frame-operator factorization, nine-dimensional
  information gauge, approximate-design optimum, and boundary order.
- `spin9_three_spinor_symmetry.py`: exact interior curve stabilizer and
  \(2V_7\oplus2V_5\oplus4V_3\) branching.
- `spin9_grassmann_slice.py`: exact \(V_1\oplus V_5\) normal-slice theorem.
- `spin9_slice_isotypic_bridge.py`: exact \(\mathbb Q(\sqrt2)\) intertwiner
  from the concrete Grassmann slice to rational \(V_1\oplus V_5\)
  coordinates, supported-\(\operatorname{Sym}_0(3)\) basis alignment, and
  certified handoff of \(V_1\oplus2V_5\) to the reducible compiler.
- `spin9_local_hessian.py`: exact quotient projection and coupled
  \(V_5\otimes\mathbb R^2\) Hessian certificate proving strict local
  D-optimality modulo Spin(9).
- `spin9_v5_ray_certificate.py`: exact finite-radius determinant bounds on the
  zero-cubic and axisymmetric graph rays in the Grassmann \(V_5\) slice,
  including the exact Cayley-null radial counterexample.
- `spin9_v5_cartan_reconstruction.py`: 631-coefficient invariant reconstruction
  of the complete \(V_5\cong\operatorname{Sym}_0(3)\) Cartan determinant.
- `spin9_v5_cartan_certificate.py`: deterministic characteristic-zero lift and
  six-cell strict Bernstein atlas proving the \(101/100\) bound for every pure
  \(V_5\) graph over the Cayley-null plane.
- `spin9_v1_v5_reconstruction.py`: 18,600-coefficient modular reconstruction
  of the coupled normal-slice numerator in rational invariants
  \(x=\sqrt2s,p,y\), including both unused-prime square-root embeddings and
  exact pure-\(V_1\)/pure-\(V_5\) boundary identities.
- `spin9_v1_v5_gap.py`: exact three-cube Bernstein-control construction for
  coupled rational gaps, including retained unresolved cells for local-chart
  handoff rather than silent failure.
- `spin9_v1_v5_boundary_char0.py`: raw \(\mathbb Q(\sqrt2)\) information-block
  identity replay on the two exceptional boundary planes using their complete
  degree-72 lower Newton grids; it does not use the modular coefficient table.
- `spin9_v1_v5_blowup.py`: exact coefficient extraction, factorization, and
  eight-chart strict Bernstein certificate proving both exceptional coupled
  boundary planes below \(26/25\), chained to the raw boundary identities.
- `spin9_v1_v5_global.py`: exact compact-plus-local projective atlas proving
  the \(21/20\) gap for the reconstructed rational function on the complete
  coupled orbit domain.
- `spin9_v1_v5_char0.py`: 22-prime, two-embedding graded raw determinant
  identity and explicit coefficient-bound lift to characteristic zero.
- `spin9_v1_v5_theorem.py`: claim-boundary assembler promoting the complete
  finite-radius coupled-slice theorem while keeping candidate optimality, the
  second \(V_5\), and the unrestricted quotient open.
- `spin9_v1_candidate_line.py`: exact rational pullback of the symmetric
  equiangular determinant curve by the complete pure-\(V_1\) graph line, proving global
  candidate optimality there and classifying its four graph preimages.
- `spin9_v1_v5_screen.py`: reproducible float64 multistart, compactified
  random, and projective-boundary falsification screen. It finds no candidate
  counterexample but exactly rejects pointwise monotonicity in the \(V_5\)
  radius; it is numerical evidence only.

## Blind action and addressing line

- `spin8_blind_shared_action.py`
- `spin8_learned_address.py`
- `spin8_continuous_alias.py`
- `spin8_blind_alias_action.py`
- `spin8_masked_completion.py`
- `spin8_triality_identifiability.py`
- `schurscan_delta_memory.py`: exact factored delta recurrence with sequential,
  ordered work-efficient, and real two-level chunkwise scans.
- `matched_learned_retrieval.py`: frozen matched overwrite, long hot/cold stream,
  key corruption, transport, and sample-efficiency campaign.
- `hierarchical_matched_retrieval.py`: same-router dense, two-slot block, and
  hard routing ablation for direct and exact DeltaRule memories.
- `aggregate_hierarchical_matched_retrieval.py`: enforces independent seed
  artifacts before producing the hierarchical quality aggregate.
- `large_slot_semantic_hierarchy.py` and
  `aggregate_large_slot_semantic_hierarchy.py`: 64-slot overlapping-semantic,
  shared-three-view versus independent routing campaign with retained weights.
- `benchmark_gathered_block_memory.py` and its aggregate: actual eager selected-
  block state mutation against masked-full and dense controls.
- `benchmark_fused_gathered_block_memory.py` and its aggregate: one-kernel
  Triton coarse/fine route, gathered write, and gathered read benchmark.
- `fused_gathered_recurrent_diagnostic.py`: post-registered 257-step trajectory
  comparison against the eager gathered recurrence.
- `plot_memory_benchmark_atlas.py`: deterministic publication figures derived
  from frozen quality, fused-gather, matched-core, official-FLA, and co-moving
  aggregates; it performs no retiming and writes a source-hash manifest.
- `spin9_clifford_memory.py`: exact Spin(9) bind/unbind, Hopf coarse-index,
  equivariance, crosstalk, and dynamic direct-gauge boundary.
- `schurscan_comoving_delta.py`: exact invertible-action compilation of
  transported DeltaRule into a standard DeltaRule plus cumulative action and
  inverse/read frame changes.
- `aggregate_matched_learned_retrieval.py`: validates and combines durable
  per-seed campaign artifacts.
- `task_b_delta_action_replay.py` and
  `aggregate_task_b_delta_action_replay.py`: strict historical metric replay
  plus exact direct/delta hard-route controls; the frozen reproduction failure
  is intentional evidence, not a passing rerun.
- `task_b_paired_action_replication.py` and
  `aggregate_task_b_paired_action_replication.py`: prospective paired shared/
  independent action fits with retained parameters, direct/delta sequence and
  scan parity, and independent per-seed verification.
- `benchmark_matched_memory_cores.py`: corrected core and 384-parameter
  encoder-inclusive eager CUDA tier with raw samples, full gradients, tuning,
  and allocated/reserved memory.
- `select_matched_memory_core_implementations.py` and
  `aggregate_matched_memory_core_benchmarks.py`: freeze disjoint tuning choices
  and aggregate independent local measurement processes.
- `benchmark_fla_delta_rule.py`: official FLA compact-WY chunk and fused
  recurrent delta kernels against state-matched local hard-key cores on Linux;
  transport is explicitly disabled because the standard op has no value-action
  input.
- `select_fla_delta_rule_implementations.py` and
  `aggregate_fla_delta_rule_benchmarks.py`: freeze and aggregate the external
  fused tier.
- `benchmark_comoving_fla_delta_rule.py`: stable full-path noncommuting
  transport compiler benchmark, including float32 action prefix/solve around
  the official fp16 FLA operators.
- `select_comoving_fla_implementations.py` and
  `aggregate_comoving_fla_benchmarks.py`: independent-process implementation
  selection and raw-sample aggregation for the transported fused tier.
- `analyze_matched_retrieval_campaign.py`: claim-audited Task A/Task B
  synthesis, including the optional prospective Task B closure artifact.

## Group-action and compiler lineage

- `mechanistic_group_actions.py`, `representation_retraction.py`, and
  `latent_group_discovery.py` contain the A5 mechanism and shared-family
  compiler work.
- `q8_spinor_*` and `spin8_q8_*` contain the center-fidelity, regular-orbit,
  path-section, and observer-transport experiments.
- `spin8_table_blind_*`, `spin8_state_only_*`, and `spin8_finest_congruence_*`
  progressively remove tables, labels, and supplied state cardinality.
- `action_congruence_lattice.py` and the `*_lattice_audit.py` files certify the
  complete recovered congruence structure.

## Recurrence baselines

- `ga_ssm.py`, `GALib.py`, and `rotor_ssm_torch.py` are the maintained GA
  recurrence implementations.
- `recurrence_families_torch.py` and `compare_recurrences.py` implement the
  matched real, complex, quaternion, Householder, and rotor ladder.
- `schur_scan.py` implements the full isotypic multiplicity commutant and
  associative real-type Schur-affine scan.
- `division_schur_scan.py` implements canonical complex- and
  quaternionic-type right-multiplicity blocks, exact commutant audits, and an
  associative differentiable prefix scan with noncommutative order checks.
- `schur_type_detector.py` solves exact rational commutants, detects irreducible
  or declared-field commutants, detects irreducible
  real/complex/quaternionic Schur type under complete reducibility, extracts a
  multiplication basis, and rejects split or repeated controls.
- `exact_real_scalar_field.py` defines the exact ordered coefficient-field
  contract for \(\mathbb Q\) and the positive embedding of
  \(\mathbb Q(\sqrt2)\), including membership, sign, sparse nullspace, rank,
  and projective normalization.
- `reducible_isotypic_decomposition.py` recursively splits exact commutant
  idempotents over the selected field, certifies irreducible Schur leaves,
  groups them by exact intertwiner spaces, exposes aligned isotypic
  coordinates, and verifies corner-algebra and center dimensions.
- `algebraic_isotypic_decomposition.py` runs the genuine quadratic
  split/nonsplit and field-refusal controls, dense algebraic Schur conjugacies,
  and direct native Spin(9) slice/full-quotient compiler audit.
- `clifford_signature_extension.py` certifies the Spin(8) module controls and
  the faithful \(\mathrm{Cl}(1,4)\) image, its even algebra, volume sectors,
  and embedded \(\mathrm{Cl}(3,0)\) action.
- `spin9_slice_isotypic_bridge.py` is the first exact algebraic front end to
  the original rational compiler: it rationalizes one concrete
  \(\mathbb Q(\sqrt2)\) representation and remains an independent closed-form
  comparison for the native field-aware path.
- `GA-SSM-1.py` through `GA-SSM-3.py` are retained research history, not the
  maintained interface.

For scientific interpretation, start with
[`docs/RESEARCH_MAP.md`](../docs/RESEARCH_MAP.md), not with an isolated script.
