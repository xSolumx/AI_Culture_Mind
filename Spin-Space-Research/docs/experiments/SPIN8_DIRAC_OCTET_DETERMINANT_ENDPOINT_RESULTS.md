# Endpoint Theorem for the Adjacent-Octet Schur Determinant

**Computer-assisted exact theorem — 2026-08-12**

**Status:** exact reconstruction complete; \(D\geq0\) proved on the complete
\(y=0\) and \(y=1\) faces. Positivity for \(0<y<1\) remains open.

**Source:**
[spin8_dirac_endpoint_octet_determinant.py](../../src/spin8_dirac_endpoint_octet_determinant.py)

**Artifact:**
[spin8_dirac_endpoint_octet_determinant_20260812.json](../../artifacts/spin8_dirac_endpoint_octet_determinant_20260812.json)

## Exact reconstruction

For the Klein-four circulant second Schur block, write
\(A=\sqrt{s_1}Z_1\), \(B=\sqrt{s_2}Z_2\), and
\(C=\sqrt{s_3}Z_3\). The source harness independently expands the generic
four-by-four determinant and the product of its four Walsh eigenvalues. Both
give

\[
\begin{aligned}
D={}&Z_0^4-2Z_0^2\sum_{j=1}^3s_jZ_j^2
+8\tau Z_0Z_1Z_2Z_3\\
&+\sum_{j=1}^3(s_jZ_j^2)^2
-2\sum_{1\leq j<k\leq3}s_js_kZ_j^2Z_k^2,
\end{aligned}
\]

where exact reconstruction reverifies \(s_1s_2s_3=\tau^2\). The resulting
integer polynomial has 6,082,148 terms and multidegree
\((24,24,24,24,48)\), exactly matching the frozen structural prediction. The
artifact hash-binds all eight input sector-polynomial files.

## The \(y=1\) face

At \(y=1\), the coset term vanishes and the exact Schur identity gives
\(Z=X^2\). The independently proved complete \(X\)-block theorem therefore
implies

\[
\det Z=\det(X)^2\geq0.
\]

The dependency artifact is hash-bound rather than inferred from a stored
prose claim.

## The \(y=0\) face

The low face has 49,484 terms and multidegree \((24,24,16,4)\). Its native
53,125-control Bernstein tensor contains 41 negative controls. This rejects
the native certificate basis only.

A complete dyadic split in \((u_d,u_e,u_g,u_i)\) certifies 15 of the 16 coarse
cells. The sole rejected cell is '0001', meaning

\[
(u_d,u_e,u_g)\in[0,\tfrac12]^3,
\qquad u_i\in[\tfrac12,1].
\]

All sixteen dyadic children of that complete cell certify independently. Thus
the 15 retained coarse cells plus those 16 children form a finite 31-leaf
cover of the entire four-cube, proving \(D|_{y=0}\geq0\).

## Replay boundary

The full source harness reconstructs the 6,082,148-term polynomial and
recomputes all Bernstein transforms under the six-thread resource contract.
The compact test rechecks the generic determinant identity, all input and
dependency hashes, the exact 16-cell coarse cover, the exact 16-child delegated
cover, and every stored leaf sign count. It explicitly trusts the stored exact
Bernstein summaries; rerunning the source harness is the full transform replay.

## Nonclaims and next gate

This theorem closes the frozen reconstruction and endpoint stages. It does not
prove \(D\geq0\) for \(0<y<1\), the complete adjacent endpoint octet, the
unrestricted seven-variable Dirac--Gram inequality, or global five-query
optimality. The next exact object is the two-endpoint interior quotient from
the frozen selector decomposition, followed by tangent analysis only if its
negative Bernstein support localizes at an equality stratum.
