# Spin(3) isotypic Schur-scan results

Date executed: 2026-08-03. The gates were frozen in
`SPIN3_ISOTYPIC_SCHUR_SCAN_PREREGISTRATION.md` before execution. Raw output is
`spin3_isotypic_schur_audit.json`.

## Result

All algebraic, optimization-control, and scan gates passed.

| Diagnostic | Result |
|---|---:|
| numerical Cl(3) Spin(3) centralizer dimension | 8 |
| current grade-preserving basis rank | 4 |
| complete isotypic basis rank | 8 |
| isotypic basis maximum rotor commutator | `2.22e-16` |
| Hodge-copy swap residual in grade family | `1.0` |
| Hodge-copy swap residual in isotypic family | `1.92e-16` |
| Hodge-copy swap equivariance error | `2.22e-16` |
| 4-parameter `GradeLinear` fit MSE | `1.0032208` |
| 16-parameter two-layer grade fit MSE | `1.0032208` |
| 8-parameter isotypic fit MSE | `1.58e-15` |
| affine composition associativity error | `1.11e-16` |
| length-17 parallel/recurrent error | `8.88e-16` max |

The optimization fit used the local RTX 2070 SUPER. Centralizer and scan
certificates used float64 CPU arithmetic.

## Corrected foundation

Under proper rotor conjugation,

```text
Cl(3,0) = 1 + 3 + 3 + 1.
```

The prior `GradeLinear` implementation is equivariant, but it is not the
complete equivariant linear family. It independently mixes each grade across
channels, thereby discarding every scalar/pseudoscalar and
vector/Hodge-bivector intertwiner. For `C` input and `D` output channels it has
`4CD` linear degrees of freedom, whereas the full Spin(3) commutant has `8CD`.
The same omission halves the allowed invariant bias space.

The new `Spin3IsotypicLinear` realizes the complete commutant. The Hodge-copy
swap is an exact witness: it is Spin(3)-equivariant but orthogonal to the entire
grade-preserving subspace. Adding depth or parameters to grade-preserving
linear layers cannot repair this closure property.

## SchurScan construction

For a real-type isotypic state

```text
V = direct_sum_lambda (R^m_lambda tensor V_lambda),
```

define the token action

```text
A_t = direct_sum_lambda (M_t,lambda tensor rho_lambda(g_t)).
```

Ordered composition stays factored:

```text
A_b A_a = direct_sum_lambda (
    M_b,lambda M_a,lambda tensor rho_lambda(g_b g_a)
).
```

The affine extension is the usual semidirect-product law. `schur_scan.py`
implements this without materializing Kronecker matrices and verifies exact
logarithmic-depth scan/recurrent parity. For complex- or quaternionic-type real
irreps, the complete multiplicity algebra is over the corresponding Schur
division algebra rather than ordinary real matrices; that extension is not yet
implemented.

## Claim boundary

This is a genuine correction and constructive extension of the local model.
It is not yet evidence that SchurScan improves language modeling. Isotypic
decomposition and multiplicity-space intertwiners are established equivariant-
network theory; dense selective transitions and associative noncommutative
scans are also established. The potentially new contribution is their joint,
representation-factored selective affine-scan architecture. That claim remains
provisional pending a deeper prior-art audit and parameter-matched sequence
tasks.

## Next falsifier

The next task must require cross-copy information flow and long ordered
composition. Compare:

1. grade-preserving damped rotor;
2. SchurScan with the complete multiplicity action;
3. parameter-matched dense selective SSM;
4. Householder-product transition;
5. diagonal real/complex selective SSM.

All families must share state width, decoder budget, training sequences, and
dense length sweeps. The second independent axis is retention/write coupling:
the maintained `sqrt(1-d^2)` drive scale ties near-perfect memory to a weak
write. Because strict contraction already gives BIBO stability for any bounded
write gate, an independently gated write is the next optimization ablation.
