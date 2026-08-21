# Pure Spin(8) SSM v1.1 contract

Status: maintained PyTorch model alongside, not in place of, Pure Rotor v2.1.

The lower-dimensional `pure_rotor_ssm` remains the stable `Cl(3,0)` model and
retains its v2.1 checkpoint format. `pure_spin8_ssm` is a new explicit model
family with its own version and checkpoint schema.

## State and group action

The default recurrent state contains all three eight-real irreducible triality
representations:

`h = (h_v, h_+, h_-) in 8v + 8s+ + 8s-`.

One shared 28-coordinate controller constructs one Spin(8) element in the
vector and both chiral representations. The 24-scalar tuple distinguishes all
four central signatures. A single 8D vector or chiral stream is only an
`SO(8)`-type action chart and is not called triality-faithful here.

Two action charts are supported:

- `exponential` (default): the locally faster batched matrix exponential of the
  shared Lie-algebra tangent;
- `factorized`: ordered exact one-plane exponentials. It has no
  Cayley `-1` singularity and represents a `2*pi` central spin rotation exactly;

Both charts produce associative linear operators. Their coordinates are not
numerically identical and checkpoints record the selected chart. On the local
RTX 2070 SUPER, the vectorized factorized constructor is 2.6x slower than the
batched exponential for `(B,L,C)=(8,128,2)`, so it remains an exact-center
alternative rather than the training default.

## Scan and stability

Each channel uses

`h_t = d_t A_t h_(t-1) + (1-d_t) w_t z_t`,

where every selected representation of `A_t` is orthogonal,
`0 < d_t,w_t < 1`, and `||z_t|| < 1`. Therefore, independently for every
channel and triality stream,

`||h_t|| <= max(||h_0||,1)`.

Training defaults to an ordered work-efficient Blelloch-style scan over the
associative affine composition law. Hillis--Steele and recurrent paths are
correctness/streaming alternatives. Padding masks insert exact identity
transitions. The cache is 24 scalars per channel per layer and does not grow
with sequence length.

## Compiled finite-token inference (v1.1)

`discrete_scan.py` provides a separate compiler/runtime boundary for models
whose trained input router has identified a finite action dictionary. The
router is evaluated once per vocabulary item and frozen as a
`(vocabulary, representations, 8, 8)` table. The eager recurrence preserves
action-table and initial-state autograd. The optional CUDA float32 Triton path
keeps one eight-scalar state per `(batch, representation)` in registers and
implements initial-state backward, but deliberately does not implement table
gradients.

`CompiledSpin8TokenTracker` stores the action table, initial state,
representation order, compiler metadata, and package version in a distinct
checkpoint. This is an inference optimization for a frozen finite dictionary;
it is not used silently by the continuous `PureSpin8SSMLayer`, and it is not a
parallel-prefix algorithm.

`transport_only=True` removes damping and drive controllers and scans pure
Spin(8) actions. `normalize_inputs=False` is available when inputs already are
meaningful bivector coordinates. Both choices are serialized; the general
causal block defaults to bounded affine memory with normalized embeddings.

## Triality coupling

The recurrent scan stays affine. After the scan, the fixed invariant octonion
tensor supplies gated equivariant bilinear readout features among `8v`, `8s+`,
and `8s-`. This does not enter the associative composition law or mutate the
streaming cache.

## Checkpoints

`PureSpin8CausalLM.save_checkpoint` writes:

- format version 1;
- model type `pure_spin8_causal_lm`;
- package version;
- the complete serializable config, including representation order and action
  chart;
- CPU state tensors; and
- optional experiment metadata.

Pure Rotor v2.x checkpoints are not loaded into this model, and Pure Spin(8)
checkpoints are not accepted by the lower-dimensional model.

The compiled token tracker uses model type `compiled_spin8_token_tracker` and
format version 1. It cannot be loaded as a `PureSpin8CausalLM` checkpoint.

## Controlled online-identification evidence

The maintained group-action layer is now exercised by a separate continuous
router in `benchmark_pure_spin8_continuous_observation.py`. Three fresh seeds
infer local actions from unique noisy nonlinear observations, compose a held-out
adjacent center relation through L128, and pass the frozen action, signature,
state-match, checkpoint, and split gates. A 24-state independent `SO(8)^3`
control can represent the teacher but is less accurate, including under a
separately frozen local model-update wall allocation.

This evidence validates trainability and the shared-action inductive bias on
the stated synthetic every-prefix task. The router is an experiment, not a new
silent default or checkpoint-format change. See
[`PURE_SPIN8_CONTINUOUS_OBSERVATION_RESULTS.md`](../experiments/PURE_SPIN8_CONTINUOUS_OBSERVATION_RESULTS.md).

A frozen successor exercises the same action layer with endpoint-only loss.
Every L16 sequence provides only its final signed 24-real state; no intermediate
target exists in the batch, and a unit test requires zero loss gradient at every
nonfinal prediction. Three fresh seeds preserve exact L128 relation-row
correctness and a shared-action advantage under both equal updates and a
separately pre-frozen local update-wall allocation. This removes the dense-
prefix-target dependency for this bounded synthetic teacher, not for unsigned,
partial, noninjective, natural, or all-28-coordinate observations. See
[`PURE_SPIN8_ENDPOINT_SUPERVISION_RESULTS.md`](../experiments/PURE_SPIN8_ENDPOINT_SUPERVISION_RESULTS.md).

The partial-readout follow-up certifies the representation boundary rather than
changing the layer. Exact rational generator-probe ranks reach 28 from seven
basis states in each individual 8D view. One signed half-spin endpoint trains
the shared router across all three views, but the overall frozen cohort fails
because center-blind vector supervision does not robustly select an exact lift.
An identical-vector-input/opposite-spinor-target collision proves that a true
vector quotient cannot reveal a balanced hidden lift. See
[`PURE_SPIN8_ENDPOINT_OBSERVABILITY_RESULTS.md`](../experiments/PURE_SPIN8_ENDPOINT_OBSERVABILITY_RESULTS.md).

The frozen calibration successor repairs that specific lift-selection failure
without changing the maintained layer. For unit positive-spinor endpoint `y`,
it supplies the lift-invariant max-coordinate address `argmax |y_j|` and the
single lift-odd bit `sign(y_j)` alongside the final vector endpoint. The
address uses three bits, so the full calibration word has four bits; only one
bit selects the double-cover fiber. Exact geometry bounds the selected
magnitude below by `1/sqrt(8)`. Across untouched seeds 4--6, every seedwise
action, L128, center, relation, capability, and integrity gate passes without
median rescue. This is an experimental supervision interface, not a new model
default or checkpoint format. See
[`PURE_SPIN8_LIFT_BIT_CALIBRATION_RESULTS.md`](../experiments/PURE_SPIN8_LIFT_BIT_CALIBRATION_RESULTS.md).

The exact optimizer-trace audit shows that the independent control's
negative-specific coordinate head is outside that adaptive loss graph: every
weight and bias has zero data gradient and the final block equals its repeated
AdamW decay-only counterfactual exactly. The shared router's one coordinate head
has nonzero gradient in all 28 rows. A same-state shared-latent control with
independently trainable spinor alignments then supplies the stronger falsifier.
Its frozen all-view dominance gate fails on two directly supervised vector-L128
cells, while the maintained alignment wins every action, spinor-L128, and
completely hidden negative-L128 comparison. Full supervision trains both
scrambled alignments and restores negative-view capability. See
[`PURE_SPIN8_LIFT_GRADIENT_IDENTIFIABILITY_RESULTS.md`](../experiments/PURE_SPIN8_LIFT_GRADIENT_IDENTIFIABILITY_RESULTS.md)
and
[`PURE_SPIN8_SCRAMBLED_ALIGNMENT_RESULTS.md`](../experiments/PURE_SPIN8_SCRAMBLED_ALIGNMENT_RESULTS.md).

The follow-up negative-only calibration curve leaves the maintained router and
24-scalar state bitwise unchanged. The exact ordered-probe ranks are
`0,7,13,18,22,25,27,28,28`; explicit rational stabilizer witnesses prove that
fewer than seven probes cannot globally identify the `SO(8)` action, while
seven do and eight are redundant. Rank 28 recovers the aligned action in all
fresh seeds. The frozen uniform effect-size gate still fails in seed 10, so the
result is an exact action-identifiability theorem plus bounded optimization
evidence, not an all-seed task-error phase transition. See
[`PURE_SPIN8_ALIGNMENT_CALIBRATION_RANK_RESULTS.md`](../experiments/PURE_SPIN8_ALIGNMENT_CALIBRATION_RANK_RESULTS.md).

## Claim boundary

This contract establishes an implemented, faithful Spin(8) representation
tuple, legal associative scans, constant cache, center coverage, bounded state,
explicit checkpoint formats, a tested frozen-dictionary CUDA recurrence, and a
replicated noisy continuous-router exercise under both every-prefix and
endpoint-only supervision, including a bounded partial-readout observability
audit with a preserved negative result and a separately frozen adaptive
calibration repair. The matched scrambled-alignment result supports only a
bounded cross-view spinor-transfer claim and explicitly fails universal
all-view dominance. The calibration-rank result identifies the supplied
`SO(8)` action globally from seven ordered probes but does not infer that frame
or its Spin lift from raw inputs. This contract does not by itself establish
language-model superiority, generic triality necessity, natural-data utility,
end-to-end fused action construction, a global optimizer theorem, or the open
unrestricted Dirac--Gram/D-optimality theorem.

## Isotypic-to-silicon compiler v2.1.1

`ScanMode="compiled_recurrent"` is an opt-in continuous-action path. Its typed
plan keeps the exact Schur block, runtime action-sharing contract, precision,
hardware target, and measured dispatch profile separate. It refuses to treat
independently routed channels as isotypic multiplicity. FP32 scalar Triton has
a custom reverse pass for action, scale, drive, and initial state. FP16
Tensor-Core inference is selected only for an exact passing hardware/shape
profile and never for training in v2.1.1.

`SelfCalibratingSpin8SSMLayer` accepts seven ordered vector-probe images and a
separate lift sign. An oriented Hodge cofactor completes the `SO(8)` frame;
adjacent Givens factors lift it into all three maintained triality actions.
The supplied sign leaves `8v` fixed and flips `8s+` and `8s-`. QR projection is
available for noisy differentiable inputs, but it is a local chart rather than
a global continuous section. The caller remains responsible for probe rank,
orientation convention, and a valid `+1/-1` lift sign.

The recorded RTX 2070 SUPER audit finds a `112.62x` full-forward/backward gain
over a sequential eager recurrence oracle, but only a `1.059x` end-to-end
self-calibrating gain because action construction dominates. One of eight
inference cells selects Tensor Cores and records `1.423x`; the other seven
correctly remain scalar. See
[`ISOTYPIC_TO_SILICON_COMPILER_V211.md`](../experiments/ISOTYPIC_TO_SILICON_COMPILER_V211.md).
