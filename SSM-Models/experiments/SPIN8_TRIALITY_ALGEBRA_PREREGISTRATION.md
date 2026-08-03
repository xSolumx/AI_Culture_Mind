# Spin(8) triality algebra gate: prospective protocol

Status: frozen before the first execution of the Spin(8) construction.

Date: 2026-08-03

## Question

Can the project construct the vector and two real chiral-spinor
representations of `Spin(8)` from one shared 28-coordinate bivector, while
retaining the exact algebraic structure required for a recurrent associative
state-space model?

This is an algebra and implementation gate. Passing it is not evidence of a
learning advantage.

## Construction fixed in advance

1. Build the real octonion algebra from a fixed oriented Fano plane.
2. Form the eight real `8 x 8` left-multiplication maps `rho_i` for the
   octonion basis.
3. Assemble real `16 x 16` Clifford generators

   ```text
   Gamma_i = [[0, rho_i^T], [rho_i, 0]].
   ```

4. Restrict `1/4 [Gamma_i, Gamma_j]` to the two chiral blocks to obtain
   28 generators on each of `S+` and `S-`.
5. Use the standard 28 plane-rotation generators on the vector representation
   `V = R^8`.

No fitted matrices, oracle alignment, post-hoc rescaling, or numerical null
space search is allowed in the algebraic construction.

## Frozen acceptance tests

All algebraic tests use float64. The gate passes only if every item passes.

| Certificate | Acceptance threshold |
|---|---:|
| Octonion basis multiplication preserves basis-vector norm | exact |
| Clifford anticommutator residual | `max_abs <= 1e-12` |
| Chirality operator squares to identity and has two 8D eigenspaces | `max_abs <= 1e-12`, multiplicities `8,8` |
| Generator skew symmetry in `V`, `S+`, and `S-` | `max_abs <= 1e-12` |
| Generator-family linear rank | exactly `28` in every representation |
| `so(8)` commutator residual in all three representations | `max_abs <= 1e-12` |
| Clifford-multiplication equivariance (triality tensor certificate) | `max_abs <= 1e-12` |
| Random exponential orthogonality | `max_abs <= 1e-10` |
| Random exponential determinant | `abs(det - 1) <= 1e-10` |
| A `2 pi` vector-plane rotation | vector `+I`, both spinors `-I`, `max_abs <= 1e-10` |
| Four central elements have the expected distinct `(V,S+,S-)` signatures | `max_abs <= 1e-12` |

The `so(8)` structure constants are fixed by the vector convention

```text
J_ij e_k = delta_jk e_i - delta_ik e_j,  i < j.
```

The spinor generators must satisfy the same commutator table without a fitted
sign or scale.

The triality certificate is the infinitesimal equivariance identity for the
fixed Clifford map `rho(v): S+ -> S-`:

```text
G-_ij rho(v) - rho(v) G+_ij = rho(J_ij v).
```

This is stronger and more precise than claiming the three representations are
ordinary conjugate matrix representations. Triality relates them through an
outer automorphism and the invariant trilinear form induced by `rho`.

## Recurrent implementation gate

After the algebra passes, the same module must implement a shared bivector
controller and recurrent actions on any selected subset of `V`, `S+`, and
`S-`. The recurrent gate requires:

- full-sequence, arbitrary-chunk, and token-at-a-time state parity;
- an explicit affine tuple composition oracle;
- associativity residual `<= 1e-12` in float64 for composed tuples;
- recurrent versus sequential affine-oracle residual `<= 1e-10`;
- finite, nonzero gradients at the zero-bivector identity initialization;
- constant cache size independent of sequence length.

Floating-point prefix trees are not required to be bit-identical to a serial
recurrence. Their measured error must be reported rather than described as
exact.

## Interpretation boundary

A pass establishes that the proposed Spin(8) mechanism exists, is internally
consistent, retains center information differently across triality
representations, and supports constant-state recurrent inference. It does not
establish that triality improves optimization, long-context behavior, or
language modeling. Those claims require the already specified baselines and
multi-seed training gates.

## Chart-topology warning fixed before construction benchmarks

The Cayley transform is not representationally neutral on center-sensitive
tasks. For finite skew `A`,

```text
Cayley(A) = (I - A/2)^(-1) (I + A/2)
```

has no eigenvalue `-1`. Its plane angle is `2 atan(lambda/2)`, which approaches
`pi` only as the tangent magnitude diverges. The exponential can reach `-I`
and the nontrivial central actions at finite tangent norm. Therefore:

- throughput and local-gradient comparisons may compare exponential and
  Cayley directly near identity;
- a center-fidelity failure by Cayley cannot be attributed to optimization
  until its unreachable-boundary obstruction is accounted for;
- Cayley is a valid Q8 generator chart when the learned generator itself needs
  only a quarter-turn and its square supplies `-I`, but it is not a valid exact
  chart for a token whose action is directly `-I`.

Givens products and the exponential do not share this particular exclusion.
