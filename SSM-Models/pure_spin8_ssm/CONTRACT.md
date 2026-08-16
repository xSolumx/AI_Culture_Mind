# Pure Spin(8) SSM v1.0 contract

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

## Claim boundary

This contract establishes an implemented, faithful Spin(8) representation
tuple, legal associative scan, constant cache, center coverage, bounded state,
and explicit checkpoint format. It does not by itself establish language-model
superiority, triality-specific capacity, a global optimizer theorem, or the
open unrestricted Dirac--Gram/D-optimality theorem.
