# Spin(9) candidate cusp charts

**Status:** four exact shape-uniform cusp theorems; upper compact complement open

**Code:** [`spin9_candidate_cusp_charts.py`](../../src/spin9_candidate_cusp_charts.py)

**Artifact:** [`spin9_candidate_cusp_charts_20260821.json`](../../artifacts/spin9_candidate_cusp_charts_20260821.json), SHA-256
`b5b3851efb61c85799ac52c102f1423d90d279c0efc1965c645e1f6ccf5ddc53`

## Certificate

On the first `V1+V5` graph, the exact candidate gap has the normal form

\[
G(x,r,z)=Q(x)^2H(x)+r^2K(x,r,z),\qquad 0\le z\le1.
\]

For each algebraic root cell, let `d` be its rational distance and use exact
Bernstein bounds for `H`, the signed derivative of `Q`, and the radial
coefficient. A coefficient norm bounds all higher radial terms. The resulting
uniform estimate is

\[
G\ge c d^2+\frac32ar^2-6Ldr^2-Mr^3.
\]

Young's inequality absorbs the problematic mixed term:

\[
c d^2-6Ldr^2
\ge \frac c2d^2-\frac{18L^2}{c}r^4.
\]

All quantities are exact elements of the ordered field `Q(sqrt(241))`, lowered
by a verified 180-digit rational enclosure only where a rational Bernstein
control is required.

## Closed charts

| chart | rational `x` interval | certified `r` radius | shape |
|---:|---:|---:|---:|
| 0 | `[-1/2,-2/5]` | `10^-47` | all `0<=z<=1` |
| 1 | `[-1/50,-1/100]` | `10^-52` | all `0<=z<=1` |
| 2 | `[1,11/10]` | `10^-35` | all `0<=z<=1` |
| 3 | `[31,33]` | `10^-8` | all `0<=z<=1` |

These intervals are macroscopic compared with the earlier `10^-140` root
cells. The proof remains valid through the extremely nearby sign changes of
the raw radial coefficient because the pure `Q^2 H` term absorbs the mixed
variation.

This closes the four equality-edge handoffs selected by the quadratic atlas.
It does not yet prove the compact complement above the four rational radial
floors, control the second `V5`, or prove the unrestricted quotient theorem.
