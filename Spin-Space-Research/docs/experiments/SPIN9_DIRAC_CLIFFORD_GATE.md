# Spin(9) Dirac--Clifford gate

**Date:** 2026-08-07
**Status:** exact algebra and rational rank witnesses; generic sensing theorem proved separately

## Result

The maintained Spin(8) construction already contains a real Spin(9) Clifford
system. Its eight \(16\times16\) gamma matrices and chirality product form nine
symmetric integer matrices \(P_0,\ldots,P_8\) satisfying

\[
P_iP_j+P_jP_i=2\delta_{ij}I_{16}.
\]

The 36 matrices \(\frac12P_iP_j\), \(i<j\), give the real spin action of
\(\mathfrak{spin}(9)\). Restriction to \(0\leq i<j<8\) recovers the two
maintained chiral Spin(8) actions exactly.

The coefficient-level identity

\[
\sum_{i=0}^{8}(s^{\mathsf T}P_i s)^2=(s^{\mathsf T}s)^2
\]

also holds exactly. It is reconstructed as a zero quartic polynomial, not
inferred from sampled spinors.

## Frozen sensing witnesses

For probes \(s_1,\ldots,s_k\in\mathbb Z^{16}\), form the infinitesimal
observation matrix

\[
J(s_1,\ldots,s_k)
=
\begin{bmatrix}
G_1s_1&\cdots&G_{36}s_1\\
\vdots&&\vdots\\
G_1s_k&\cdots&G_{36}s_k
\end{bmatrix},
\qquad G_r=P_iP_j.
\]

Exact elimination over three prime fields gives ranks

\[
15,\qquad28,\qquad36
\]

for the frozen one-, two-, and three-spinor witnesses. The same three-prime
agreement gives the vector-probe sequence

\[
8,15,21,26,30,33,35,36.
\]

Replacing the third spinor by a linear combination of the first two leaves the
rank at 28. Thus the third-probe result measures a genuinely new observation
direction, not merely probe count.

For the three-spinor matrix, the verifier records the actual pivot rows and
columns. The corresponding nonzero \(36\times36\) determinant residues are

\[
546461\pmod{1000003},\qquad
421881\pmod{1000033},\qquad
869687\pmod{1000037}.
\]

Because one explicit 36-column minor is nonzero over a prime field, it is
nonzero over the rationals. Full infinitesimal rank therefore holds on a
nonempty Zariski-open subset of triples of spinors. This establishes generic
**local infinitesimal** identifiability independently of the global argument.
The companion
[stabilizer theorem](../manuscripts/SPIN9_THREE_SPINOR_IDENTIFIABILITY.md)
proves that a generic triple has trivial global stabilizer, while a generic
pair retains \(\operatorname{SU}(3)\).

## Dirac--Hurwitz interpretation

Define \(D(a)=\sum_i a_iP_i\). Then

\[
D(a)^2=\lVert a\rVert^2I_{16}.
\]

The classical Hurwitz--Radon theorem states that a real square composition
\([r,n,n]\) exists precisely for \(r\leq\rho(n)\). Since

\[
\rho(8)=8,
\qquad
\rho(16)=9,
\]

the local octonionic \([8,8,8]\) bind and the new \([9,16,16]\) bind both
saturate their respective square-composition bounds. This is an optimality
statement about linearly parameterized norm-preserving directions, not a claim
about the number of memory slots.

## What remains open

1. Give a coordinate-free derivation of the information spectrum and
   conditioning, beyond the stabilizer dimension.
2. Separate Hopf-base information from the seven-dimensional fibre in tasks.
3. Compare paired Dirac reflections against matched generic orthogonal and
   modern recurrent-memory baselines.
4. Establish whether any advantage survives state, parameter, and throughput
   matching.

## Reproduction contract

The JSON artifact is regenerated from the maintained Spin(8) source. The
verifier recomputes every Clifford relation, the Spin(8) restriction, the full
quartic coefficient identity, and all finite-field ranks. It does not treat a
stored `passed` field as evidence.

Artifact:
[`spin9_dirac_clifford_gate_20260807.json`](../../artifacts/spin9_dirac_clifford_gate_20260807.json),
SHA-256
`29a0f351e89a67ce1ec6b00ddecd2995d8e890c73bfc643bd9e6f817fe0d3277`.
