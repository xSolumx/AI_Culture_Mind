# Adaptive Exact Dominance Atlas on the Extended Adjacent-Octet Core

**Exact computer-assisted theorem — 2026-08-16**

**Status:** every unique physical orientation margin is strictly positive on

\[
(u_d,u_e,u_g,u_i,y)\in[1/8,7/8]^5.
\]

**Source:**
[`spin8_dirac_endpoint_octet_core_dominance_atlas.py`](../../src/spin8_dirac_endpoint_octet_core_dominance_atlas.py)

**Artifact:**
[`spin8_dirac_endpoint_octet_core_dominance_atlas_20260816.json`](../../artifacts/spin8_dirac_endpoint_octet_core_dominance_atlas_20260816.json)

**Artifact SHA-256:**
`cab3e87fb0c0c0ff9abb1c21cf2ea32bec9a92edc6364e225a5a66720486f762`

## Result

The earlier 32-box theorem proves the Walsh diagonal-dominance inequality

\[
A_0>\sum_{\mu=1}^7|A_\mu|
\]

on \([1/4,3/4]^5\). This continuation expands every coordinate interval to
\([1/8,7/8]\). It retains the same exact acceptance rule: tensor Bernstein
bounds on the eight integer-scaled residual polynomials, exact bounds on the
forced radical squares, and 80-bit outward dyadic square-root ceilings checked
by exact squaring.

Strict diagonal dominance is stronger than the target determinant inequality.
It makes all eight unique Fourier eigenvalues of the octet group-circulant
strictly positive. Thus all sixteen physical margins with multiplicity are
positive, the Schur matrix is positive definite, and the final determinant is
strictly positive throughout the complete extended core.

## Adaptive cover

The 32 root boxes choose, independently in each coordinate, one of

\[
[1/8,1/2],\qquad [1/2,7/8].
\]

A rejected Bernstein bound is not interpreted as a negative function value.
Instead, that box is split into all 32 five-axis children. The exact frozen
tree closes as follows:

| Refinement depth | Boxes tested | Certified leaves | Refined basis failures | Unresolved |
|---:|---:|---:|---:|---:|
| 0 | 32 | 24 | 8 | 0 |
| 1 | 256 | 234 | 22 | 0 |
| 2 | 704 | 672 | 32 | 0 |
| 3 | 1,024 | 1,018 | 6 | 0 |
| 4 | 192 | 192 | 0 | 0 |

The resulting prefix-free atlas has 2,140 certified leaves and 68 internal
refinement nodes. Every rejected node delegates to all 32 children; no branch
is dropped and no leaf overlaps a descendant.

The smallest certified physical-amplitude gap occurs on path
`00001/01101`, corresponding to

\[
\begin{aligned}
u_d,u_i&\in[1/8,5/16],\\
u_e,u_g&\in[5/16,1/2],\\
y&\in[11/16,7/8].
\end{aligned}
\]

Its exact lower bound is

\[
\frac{320281275533252594456507202812099057}
{5316911983139663491615228241121378304}
\approx 0.060238212810159196.
\]

The rational inequality, not the decimal rendering, is the theorem gate.

## Replay tiers

The full exact replay takes several minutes on the recorded six-thread FLINT
contract:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_dirac_endpoint_octet_core_dominance_atlas `
  --flint-threads 6 `
  --output artifacts/spin8_dirac_endpoint_octet_core_dominance_atlas_20260816.json
```

The compact regression test is intentionally different:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q tests/test_spin8_endpoint_octet_core_dominance_atlas.py
```

It rehashes the exact coefficient sources, checks every stored rational gap,
verifies the physical/common-scale conversion, and reconstructs the complete
prefix tree. It does not recompute all 2,208 Bernstein transforms; that is the
full source-harness tier.

## Claim boundary and next gate

Together with the complete coordinate-boundary theorem, the proved regions
are now:

1. every coordinate face of \([0,1]^5\); and
2. the full extended core \([1/8,7/8]^5\), with strict margin positivity.

The remaining adjacent-octet domain is confined to the width-\(1/8\) collars
where at least one coordinate approaches a face. Those collars include the
orthonormal equality corner and cannot be discarded. The next certificate
should extend the adaptive atlas into them while delegating only the branches
converging to that equality set to the existing order-eight nested blow-up.

This result does not prove the complete adjacent endpoint octet, the
unrestricted seven-variable Dirac--Gram inequality, or global five-query
D-optimality.
