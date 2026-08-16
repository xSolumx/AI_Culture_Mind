# Octonion Multiplication-Operator Scan Protocol

Protocol frozen: **2026-08-16T20:40:20+02:00**

## Question

Can the repository use octonionic token actions without applying a
nonassociative operation inside a parallel prefix tree?

The proposed lift maps each unit octonion `u` to the real linear operator

`L_u : x -> u x`.

The leaf map is octonionic, but scan composition is ordinary associative
matrix/function composition. The target recurrence is the explicitly
parenthesized update

`h_t = u_t h_(t-1)`.

Thus its prefix operator is

`P_t = L_(u_t) ... L_(u_1)`.

The implementation must never simplify this product to
`L_(u_t ... u_1)`: that identity is false outside associative subalgebras and
would erase the associator that motivates the lift.

## Frozen algebra gates

Use the repository's fixed oriented Fano triples

`(123), (145), (176), (246), (257), (347), (365)`.

The implementation passes only if:

1. its complete basis table equals `spin8_triality.py` exactly;
2. left and right matrices reproduce raw octonion multiplication;
3. every tested unit left/right operator is orthogonal with determinant `+1`;
4. the explicit `(e1,e2,e4)` associator is nonzero, while matrix composition
   remains associative;
5. `L_u L_v != L_(uv)` is witnessed explicitly rather than hidden by a
   quaternionic test triple;
6. `L_(-1) = -I8`, so the central sign remains visible; and
7. the 28 by 28 upper-triangle coordinate matrix formed by
   `L_(e_i)` and `[L_(e_i),L_(e_j)]`, `1 <= i < j <= 7`, has exact determinant
   `-2^49`.

Gate 7 certifies that the available one-parameter left actions generate the
full Lie algebra `so(8)`. Because `SO(8)` is connected, their generated
connected operator subgroup is all of `SO(8)`. This is a controllability/
expressivity statement about operator products, not a claim that one
octonion parameterizes an arbitrary `SO(8)` matrix or that the observation is
new mathematics.

## Frozen scan gates

Implement three execution paths:

- raw sequential recurrence with an eight-scalar cache;
- an ordered Hillis--Steele matrix-prefix oracle; and
- the repository's ordered Blelloch-style work-efficient tree, requiring
  fewer than `3P` compositions for padded length `P`.

For non-power-of-two lengths and noncommuting examples, require:

- recurrent/work-efficient/Hillis forward parity;
- first-order gradient parity;
- padding identities and arbitrary split/cache continuation;
- left- and right-action orientation checks; and
- CUDA backward when CUDA is available.

No scan may QR-project intermediate prefixes. QR would make the computed
transition depend on tree association and would invalidate exact scan closure.
Floating-point tree orders need tolerance parity, not bitwise identity.

## Bounded affine stabilization theorem

The trainable layer uses

`h_t = d_t L_(u_t) h_(t-1) + (1-d_t) w_t z_t`,

where `||u_t||=1`, `0 <= d_t,w_t <= 1`, and `||z_t|| <= 1`. Since `L_(u_t)`
is orthogonal,

`||h_t|| <= d_t ||h_(t-1)|| + (1-d_t) w_t`

and therefore

`||h_t|| <= max(||h_0||, 1)`

for every sequence length. The affine maps compose associatively as

`(A2,b2) o (A1,b1) = (A2 A1, A2 b1 + b2)`.

Require recurrent/parallel/gradient/cache parity and a long-sequence numerical
bound. Parallel training may materialize 8 by 8 or homogeneous 9 by 9 prefix
operators; streaming inference must retain only the eight-scalar state per
lane and apply raw parenthesized octonion products.

## Systems measurement

On the named local GPU, compare prebuilt 8 by 8 operator prefixes under
work-efficient and Hillis--Steele trees, and compare the structured eight-
scalar recurrent path. Report construction separately where practical,
composition counts, dtype, batch, lanes, length, warmups, repeats, and timing
dispersion. This is an eager implementation measurement, not a fused-kernel or
production-SSM claim.

## Claim boundary and falsifiers

Success establishes a correct associative operator lift, full `so(8)` Lie
closure, compact streaming, and a bounded affine layer. It does **not**
establish:

- that raw octonion multiplication became associative;
- an eight-scalar cache for arbitrary accumulated `SO(8)` operators;
- triality-specific task quality;
- superiority over Mamba, DeltaNet, attention, or fused orthogonal kernels;
- novelty of classical octonion multiplication-operator theory; or
- the unrestricted Dirac--Gram / global D-optimality theorem.

The architecture is rejected or narrowed if the associator is accidentally
collapsed, exact Lie rank is below 28, scan gradients disagree, the recurrent
cache grows with length, or the stated norm bound fails beyond declared
floating-point tolerance.
