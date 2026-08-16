# Complete Coordinate-Boundary Theorem for the Adjacent-Octet Determinant

**Exact computer-assisted theorem — 2026-08-16**

**Status:** the determinant minor is nonnegative on the complete coordinate
boundary of the five-cube. Positivity when all five coordinates are strictly
interior remains open.

**Source:**
[`spin8_dirac_endpoint_octet_determinant_boundary.py`](../../src/spin8_dirac_endpoint_octet_determinant_boundary.py)

**Artifact:**
[`spin8_dirac_endpoint_octet_determinant_boundary_20260816.json`](../../artifacts/spin8_dirac_endpoint_octet_determinant_boundary_20260816.json)

**Artifact SHA-256:**
`9a8988673ce4c5af4e0dca4b822b818b0f14e58364656bcc5e884e6f7edcbbec`

## The collapse mechanism

Write the radical-free Klein-four determinant in the form

\[
\begin{aligned}
D={}&Z_0^4-2Z_0^2(q_1+q_2+q_3)+8\tau Z_0Z_1Z_2Z_3\\
&+q_1^2+q_2^2+q_3^2-2(q_1q_2+q_1q_3+q_2q_3),
\end{aligned}
\]

where \(q_j=s_jZ_j^2\), the \(s_j\) are the three forced radical
squares, and

\[
\tau=u_du_eu_gu_i y^2(1-u_d)(1-u_e)(1-u_g)(1-u_i).
\]

On every coordinate face except \(y=1\), exact substitution makes
\(\tau=0\) and leaves at most one nonzero \(s_j\). If mode \(k\) is the
survivor, the generic determinant identity becomes

\[
\boxed{D=(Z_0^2-s_kZ_k^2)^2\geq0.}
\]

The source verifies the zero-mode and all three one-mode identities in
characteristic zero before inspecting the actual face supports.

## Exact face table

| Face | Surviving nontrivial mode | Exact determinant reduction |
|---|---:|---|
| \(u_d=0\) | `0011001` | \((Z_0^2-s_1Z_1^2)^2\) |
| \(u_d=1\) | `0011001` | \((Z_0^2-s_1Z_1^2)^2\) |
| \(u_e=0\) | `0101010` | \((Z_0^2-s_2Z_2^2)^2\) |
| \(u_e=1\) | `0110011` | \((Z_0^2-s_3Z_3^2)^2\) |
| \(u_g=0\) | `0110011` | \((Z_0^2-s_3Z_3^2)^2\) |
| \(u_g=1\) | `0110011` | \((Z_0^2-s_3Z_3^2)^2\) |
| \(u_i=0\) | `0011001` | \((Z_0^2-s_1Z_1^2)^2\) |
| \(u_i=1\) | `0101010` | \((Z_0^2-s_2Z_2^2)^2\) |
| \(y=0\) | `0110011` | \((Z_0^2-s_3Z_3^2)^2\) |
| \(y=1\) | all three | \(Z=X^2\), hence \(D=\det(X)^2\) |

The final row is replayed through the hash-bound endpoint artifact. The other
nine rows are reconstructed directly from the eight exact unrestricted
sector polynomials. Their forced-square support and the vanishing of \(\tau\)
are checked symbolically, not sampled.

The earlier 31-leaf Bernstein atlas on \(y=0\) remains a valid independent
certificate. The perfect-square identity explains why that face was
nonnegative despite negative controls in its unsplit native Bernstein basis.

## Consequence

The five-cube has ten coordinate faces. All ten are now exact theorems, so a
negative value of \(D\), if one exists, must satisfy

\[
0<u_d,u_e,u_g,u_i,y<1.
\]

This removes every coordinate-boundary obligation from the determinant stage
of the adjacent endpoint-octet programme. Combined with the already-proved
scalar, quadratic, and cubic Schur minors, it leaves only the strict interior
determinant problem on this five-variable face.

## Replay

From the repository root:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_dirac_endpoint_octet_determinant_boundary `
  --output artifacts/spin8_dirac_endpoint_octet_determinant_boundary_20260816.json
python -m pytest -q tests/test_spin8_endpoint_octet_determinant_boundary.py
```

The compact test rebuilds all nine support collapses, rechecks the four generic
perfect-square identities, verifies both dependency hashes on \(y=1\), and
replays the stored scope flags.

## Nonclaims and next gate

This is not positivity on the open five-cube, the complete adjacent octet, the
unrestricted seven-variable Dirac--Gram inequality, or global five-query
D-optimality. Boundary positivity also does not imply interior positivity.

The next exact object is an interior certificate that uses the full boundary
collapse rather than subtracting only the two \(y\)-faces. Near the remaining
equality corner it must retain the established order-eight exceptional form
\(2^{48}F_4^2\) and resolve the zero set of \(F_4\) by a nested blow-up.

A subsequent
[extended-core dominance atlas](SPIN8_DIRAC_OCTET_EXTENDED_CORE_DOMINANCE_RESULTS.md)
now proves the stronger strict-margin statement on \([1/8,7/8]^5\). The
remaining determinant burden is therefore confined to the width-\(1/8\)
collars between that core and the coordinate boundary.
