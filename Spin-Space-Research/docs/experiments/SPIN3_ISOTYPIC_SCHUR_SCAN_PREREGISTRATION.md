# Spin(3) isotypic Schur-scan preregistration

Date frozen: 2026-08-03, before executing the audit.

## Question

Does the maintained grade-preserving Cl(3) mixer span the complete space of
linear maps equivariant to the symmetry it claims, and can the missing maps be
incorporated without losing associative training or constant-state streaming?

## Derivation fixed before results

Under proper rotor conjugation,

```text
Cl(3,0) = 1 + 3 + 3 + 1.
```

The scalar and pseudoscalar are equivalent trivial representations. The vector
and Hodge-dual bivector are equivalent standard SO(3) representations. For `C`
input and `D` output channels, Schur's lemma therefore predicts a real linear
commutant of dimension `8*C*D`, excluding bias. `GradeLinear` spans only
`4*C*D`. The complete equivariant bias has dimension `2*D`; `GradeLinear`
contains only `D` scalar biases.

For a general representation

```text
V = direct_sum_lambda (R^m_lambda tensor V_lambda),
```

the proposed token transition is

```text
A_t = direct_sum_lambda (M_t,lambda tensor rho_lambda(g_t)).
```

It is closed under ordered composition because

```text
(M_b tensor rho(g_b)) (M_a tensor rho(g_a))
  = (M_b M_a) tensor rho(g_b g_a).
```

The affine extension composes by ordinary semidirect-product transport.

## Frozen gates

1. A numerical centralizer calculation for generic Cl(3) rotor conjugations
   must find dimension 8 at one channel.
2. The current grade-preserving basis must have rank 4; the proposed isotypic
   basis must have rank 8 and commute with every sampled rotor action.
3. The Hodge-copy swap (scalar with pseudoscalar, vector with dual bivector)
   must be exactly equivariant, exactly expressible by the isotypic layer, and
   have nonzero best-possible residual in the grade-preserving family.
4. A wider two-layer grade-preserving network with more parameters than the
   isotypic layer must still fail the same linear target, ruling out raw
   parameter count as the explanation.
5. Factored affine composition must be associative to float64 tolerance
   `1e-10`; logarithmic-depth prefix scan and recurrent execution must agree to
   `1e-10` on length 17.
6. No language-quality or state-of-the-art claim is permitted. This is a
   theorem/implementation gate for a candidate architecture.

## Prior-art boundary

Isotypic decompositions and multiplicity-space intertwiners are standard in
equivariant neural networks. Dense selective SSMs and noncommutative scans are
also known. The candidate novelty is their explicit combination into a
representation-factored selective affine scan. A novelty claim remains
provisional until a broader paper/code search and task-level baselines are
complete.
