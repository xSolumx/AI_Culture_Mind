# Exact Diagonal Dominance on the Adjacent-Octet Central Core

**Exact computer-assisted theorem — 2026-08-16**

**Status:** every unique physical orientation margin is strictly positive on
the complete central subcube

\[
(u_d,u_e,u_g,u_i,y)\in[1/4,3/4]^5.
\]

**Source:**
[`spin8_dirac_endpoint_octet_core_dominance.py`](../../src/spin8_dirac_endpoint_octet_core_dominance.py)

**Artifact:**
[`spin8_dirac_endpoint_octet_core_dominance_20260816.json`](../../artifacts/spin8_dirac_endpoint_octet_core_dominance_20260816.json)

**Artifact SHA-256:**
`26611845fd3f5a5b5e63be145c04ede664e4ed033091fc74284381cb9c308fe2`

## Stronger target than the determinant

Write the eight surviving Walsh amplitudes as

\[
A_0, A_1,\ldots,A_7.
\]

The eight unique physical margins are signed character sums beginning with
the same trivial coefficient \(A_0\). It is therefore sufficient to prove
the stronger inequality

\[
A_0>\sum_{\mu=1}^7|A_\mu|.
\]

This is strict row diagonal dominance of the symmetric octet group-circulant.
It forces all eight unique Fourier eigenvalues, and hence all sixteen physical
orientation margins with multiplicity, to be strictly positive. In
particular the Schur matrix is positive definite and its determinant is
strictly positive.

## Exact 32-box proof

Split each coordinate interval into

\[
[1/4,1/2]\cup[1/2,3/4].
\]

The Cartesian product gives 32 exact dyadic boxes. On each box the source
reconstructs the eight hash-bound endpoint residual polynomials. The stored
residual coefficients have the common integer scale four. Exact tensor
Bernstein transforms give

\[
L_0\leq 4A_0
\]

and, for every nontrivial mode,

\[
L_\mu\leq R_\mu\leq U_\mu,
\qquad
S_\mu\leq \overline S_\mu.
\]

Here \(A_\mu=\tfrac14\sqrt{S_\mu}R_\mu\). The verifier constructs an
80-bit dyadic rational ceiling \(c_\mu\) satisfying

\[
c_\mu^2\geq\overline S_\mu
\]

in exact integer arithmetic. Therefore

\[
4|A_\mu|
\leq
\max(|L_\mu|,|U_\mu|)c_\mu.
\]

Every one of the 32 rational gaps

\[
L_0-
\sum_{\mu=1}^7\max(|L_\mu|,|U_\mu|)c_\mu
\]

is strictly positive. No floating-point value is used for acceptance.

The smallest certified physical-amplitude gap occurs in box `00011`, whose
first three coordinates lie in \([1/4,1/2]\) and last two in
\([1/2,3/4]\). Its exact lower bound is

\[
\frac{12422573099263179925270673686819}
{1267650600228229401496703205376}
\approx 9.799682260258942.
\]

The decimal is descriptive only; the rational inequality is the certificate.

## Replay

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_dirac_endpoint_octet_core_dominance `
  --output artifacts/spin8_dirac_endpoint_octet_core_dominance_20260816.json
python -m pytest -q tests/test_spin8_endpoint_octet_core_dominance.py
```

The compact replay rebuilds all residuals, all 32 box transforms, all 224
outward radical bounds, the exact minimum gap, and the theorem/nonclaim flags.

## Consequence and boundary

Together with the
[complete coordinate-boundary theorem](SPIN8_DIRAC_OCTET_DETERMINANT_BOUNDARY_RESULTS.md),
this establishes two disjoint exact regions:

1. all ten coordinate faces of \([0,1]^5\); and
2. the full compact core \([1/4,3/4]^5\), where the conclusion is strict and
   stronger than determinant positivity.

It does not cover the collars between these regions, the complete adjacent
endpoint octet, the unrestricted seven-variable Dirac--Gram domain, or global
five-query D-optimality. The next exact campaign should extend the dyadic
dominance atlas into those collars and delegate only boxes converging to the
known equality corner to the nested order-eight blow-up.

## Superseding continuation

The later
[extended-core adaptive atlas](SPIN8_DIRAC_OCTET_EXTENDED_CORE_DOMINANCE_RESULTS.md)
completes that first extension: 2,140 exact leaves prove the same strict
dominance inequality on \([1/8,7/8]^5\). This 32-box result remains the
smaller independent replay and mechanism check.
