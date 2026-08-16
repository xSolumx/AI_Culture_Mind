# An Octet Schur Reduction on an Adjacent Spin(8) Dirac--Gram Endpoint Face

**Exact theorem note — updated 2026-08-12**
**Status:** exact subgroup reduction, complete first-block theorem, exact
scalar second-block theorem, and all three global quadratic second-block
theorems; the cubic second-block theorem and the complete ten-face determinant
coordinate boundary are proved, while the strict determinant interior remains
open
**Certificate:**
[`spin8_dirac_endpoint_octet.py`](../../src/spin8_dirac_endpoint_octet.py) and
[`spin8_dirac_endpoint_octet_determinant.py`](../../src/spin8_dirac_endpoint_octet_determinant.py)
**Artifact:**
[`spin8_dirac_endpoint_octet_20260807.json`](../../artifacts/spin8_dirac_endpoint_octet_20260807.json)
and
[`spin8_dirac_endpoint_octet_determinant_20260812.json`](../../artifacts/spin8_dirac_endpoint_octet_determinant_20260812.json)

## Abstract

The unrestricted signed Dirac--Gram problem for the balanced
\(\operatorname{Spin}(8)\) triality sensor has sixteen physical orientation
margins. On the adjacent Cayley-endpoint face

\[
u_a=0,\qquad z=c^2=1,\qquad u_h=1-y^2,
\qquad (u_d,u_e,u_g,u_i,y)\in[0,1]^5,
\]

exact Walsh annihilation leaves eight amplitudes. Their masks form
\((\mathbb Z/2\mathbb Z)^3\). Splitting this group by the \(u_h\)-bit yields a
Klein-four subgroup and one coset, and hence an exact block decomposition

\[
K_8=
\begin{pmatrix}
X&\sqrt{1-y^2}\,R\\
\sqrt{1-y^2}\,R&X
\end{pmatrix},
\]

where \(X\) and \(R\) are commuting symmetric Klein-four circulants. It
follows without relaxation that

\[
K_8\succeq0
\quad\Longleftrightarrow\quad
X\succeq0
\quad\text{and}\quad
Z:=X^2-(1-y^2)R^2\succeq0.
\]

This note proves \(X\succeq0\) on the complete five-cube. The proof combines
the previously certified \(y=1\) Klein-four face with exact
boundary-selector decompositions and a two-chart triangular certificate. It
also proves the scalar principal minor \(Z_0\ge0\) by an exact identity with
the global Fourier-energy polynomial. Subsequent global certificates prove
all three quadratic modes on the entire five-cube. On the codimension-two
boundary \(u_d=u_g=0\), those three quadratic \(Z\)-minors coincide and factor
as the square of a polynomial of multidegree \((6,6,12)\). A subsequent exact
endpoint-selector identity and complete 32-box Bernstein atlas prove the cubic
minor on the entire five-cube. The fourth-order determinant is reconstructed
exactly. A complete coordinate-boundary theorem proves all ten faces: nine
collapse to one-mode perfect squares, and (y=1) is the square identity
(Z=X^2). The all-five-coordinate strict interior remains unresolved.
Consequently this is a rigorous continuation beyond the four-variable endpoint
theorem, but not yet a proof of the full adjacent face or the unrestricted
seven-variable inequality.

## 1. The surviving orientation group

Let \(A_\mu\) denote the exact Walsh amplitudes reconstructed on the full
seven-circle chart. After imposing \(u_a=0\) and \(z=1\), the nonzero sectors
are

\[
\begin{aligned}
H_0={}&\{0000000,0011001,0101010,0110011\},\\
H_1={}&\{0001111,0010110,0100101,0111100\}.
\end{aligned}
\]

The union \(H_0\cup H_1\) is closed under bitwise exclusive-or and has eight
elements. Thus it is a copy of \((\mathbb Z/2\mathbb Z)^3\). The set \(H_0\)
is its index-two Klein-four subgroup, while \(H_1\) is the other coset. The
sixteen physical sign columns restrict to the eight characters of this group,
each with multiplicity two. These facts are checked directly from the exact
triality sign table, not inferred from numerical sparsity.

Write \(u_h=1-y^2\). Every amplitude in \(H_1\) contains the common factor
\(\sqrt{1-y^2}\); amplitudes in \(H_0\) do not. Ordering characters by the
quotient \(H/H_0\) produces the displayed matrix \(K_8\). Since both blocks
belong to the commutative real group algebra of the Klein four-group, \(X\)
and \(R\) commute. The orthogonal change of basis

\[
\frac1{\sqrt2}
\begin{pmatrix}I&I\\I&-I\end{pmatrix}
\]

diagonalizes the outer two-by-two structure into
\(X\pm\sqrt{1-y^2}R\). Because \(X\) and \(R\) commute, their common Fourier
basis then gives the radical-free Schur equivalence

\[
X\pm\sqrt{1-y^2}R\succeq0
\quad\Longleftrightarrow\quad
X\succeq0,
\quad X^2-(1-y^2)R^2\succeq0.
\]

## 2. Exact positivity of the first block

Write the four group-algebra coefficients of \(X\) as \(x,a,b,c\). In the
regular basis,

\[
X=
\begin{pmatrix}
x&a&b&c\\
a&x&c&b\\
b&c&x&a\\
c&b&a&x
\end{pmatrix}.
\]

Tetrahedral symmetry reduces its principal minors to six polynomials:

\[
x,\qquad x^2-a^2,\quad x^2-b^2,\quad x^2-c^2,
\]

\[
x^3-x(a^2+b^2+c^2)+2abc,
\]

and

\[
\begin{aligned}
\det X={}&x^4-2x^2(a^2+b^2+c^2)+8xabc\\
&+a^4+b^4+c^4
-2(a^2b^2+a^2c^2+b^2c^2).
\end{aligned}
\]

All radicals cancel in these expressions. Their respective multidegrees in
\((u_d,u_e,u_g,u_i,y)\) are

\[
(3,3,3,3,6),\quad
(6,6,6,6,12),\quad
(9,9,9,9,18),\quad
(12,12,12,12,24).
\]

At \(y=1\), the coset disappears and \(X\) is precisely the complete
four-variable Klein-four matrix certified in the companion endpoint theorem.
For each minor \(p\) of \(y\)-degree \(n\), the exact decomposition begins
with

\[
p=p\big|_{y=1}\,y^n+
\left(p-p\big|_{y=1}\,y^n\right).
\]

The remainder is natively Bernstein-nonnegative for the linear, all three
quadratic, and the cubic minors. The determinant remainder has only one
additional corner contribution. Writing it as \(Q\), one has

\[
Q=C(1-u_d)^{12}(1-u_i)^{11}+Q_{\mathrm{int}},
\]

where all 659,100 exact Bernstein controls of \(Q_{\mathrm{int}}\) are
nonnegative. The corner polynomial is divisible exactly by \(1-y^2\).

To certify the quotient, split the \((u_e,u_g)\)-square into the two triangular
charts

\[
(u_e,u_g)=(rs,r(1-s))
\]

and

\[
(u_e,u_g)=(1-rs,1-r(1-s)).
\]

The upper chart is directly Bernstein-positive. In the lower chart, subtract
the \(y=1\) face with its degree-22 selector. The remaining three-variable
polynomial has 14,375 nonnegative controls. The face factors exactly as

\[
-64r^3F_{5,6}(r,s)F_{8,8}(r,s).
\]

The factor \(F_{8,8}\) is strictly Bernstein-positive. For
\(G=-F_{5,6}\),

\[
G(0,s)=4096(2s-1)^2,
\]

and

\[
G=G(0,s)(1-r)^5+G_{\mathrm{int}},
\]

where every Bernstein control of \(G_{\mathrm{int}}\) is nonnegative. Hence
the lower face, the corner quotient, the determinant remainder, and finally
\(\det X\) are nonnegative. All six principal-minor families are therefore
nonnegative, proving

\[
\boxed{X\succeq0\quad\text{on the complete five-cube}.}
\]

## 3. The scalar Schur minor

Let \(E\) be the already-certified full-sector Fourier-energy polynomial,

\[
E=A_0^2-\sum_{\mu\ne0}A_\mu^2\ge0.
\]

The identity coefficient of \(Z=X^2-(1-y^2)R^2\) satisfies the exact
polynomial identity

\[
Z_0
=E+2\sum_{\mu\in H_0\setminus\{0\}}A_\mu^2.
\]

Both terms on the right are nonnegative. Therefore

\[
\boxed{Z_0\ge0.}
\]

This identity is more informative than the native Bernstein tensor of
\(Z_0\), which contains negative controls. It shows that those controls are a
basis artifact and that the correct proof is inherited from the global energy
law.

## 4. The complete global quadratic gate

All three nontrivial quadratic families now have complete exact certificates:

\[
Z_0^2-s_{0011001}Z_{0011001}^2\ge0
\quad\text{on }[0,1]^5.
\]

\[
Z_0^2-s_{0101010}Z_{0101010}^2\ge0
\quad\text{on }[0,1]^5.
\]

\[
Z_0^2-s_{0110011}Z_{0110011}^2\ge0
\quad\text{on }[0,1]^5.
\]

The proof uses a finite dyadic atlas away from the equality corner and a
five-chart max-coordinate blow-up at that corner. Each chart has exact radial
order four. Their chart certificates are mode-specific: the first mode needs
three boundary-adapted charts, whereas four charts of each remaining mode are
natively positive. The two remaining modes have the same exact perfect-square
exceptional divisor in their final chart, an identity discovered only after
their independent atlases and blow-ups were reconstructed. The full proof and
its explicit nonclaims are recorded in
[`SPIN8_DIRAC_OCTET_QUADRATIC_RESULTS.md`](../experiments/SPIN8_DIRAC_OCTET_QUADRATIC_RESULTS.md).

## 5. The complete global cubic gate

The cubic principal minor is now proved on the complete five-cube. Writing it
as \(C\), exact characteristic-zero arithmetic verifies

\[
C=C|_{y=0}(1-y)^{36}+C|_{y=1}y^{36}+y(1-y)Q.
\]

The \(y=0\) face has an exact selector and Sturm certificate; the \(y=1\)
face follows from \(Z=X^2\); and a complete 32-box dyadic Bernstein atlas
proves \(Q\geq0\). The final assembly recomputes the 1,546,277-term identity
and forced-radical cancellation rather than inferring them from stored pass
flags. Full details and the explicit trust boundary are recorded in
[`SPIN8_DIRAC_OCTET_CUBIC_RESULTS.md`](../experiments/SPIN8_DIRAC_OCTET_CUBIC_RESULTS.md).

## 6. Exact determinant reconstruction and endpoint theorem

The final Schur minor is now reconstructed as a radical-free polynomial
\(D=\det Z\) with 6,082,148 nonzero monomials and multidegree
\((24,24,24,24,48)\). The source harness independently verifies the generic
four-by-four determinant formula, its factorization into four Walsh
eigenvalues, and the forced-radical identity \(s_1s_2s_3=\tau^2\).

Both endpoint obligations from the frozen determinant preregistration are
complete. At \(y=1\), the exact identity \(Z=X^2\) gives
\(\det Z=\det(X)^2\geq0\). At \(y=0\), the 49,484-term face has 41 negative
controls in the native Bernstein basis, so that basis is rejected. A complete
four-variable dyadic atlas certifies 15 of 16 coarse cells; the unique rejected
cell '0001' is replaced by all 16 of its children, each of which certifies.
The resulting 31-leaf exact cover proves the complete \(y=0\) face.

A later support-collapse theorem closes the complete coordinate boundary, not
only these two endpoint faces. On each of the other eight coordinate faces,
and independently again on \(y=0\), the mixed product \(\tau\) vanishes and
at most one nontrivial forced radical square survives. The determinant is
therefore exactly

\[
D=(Z_0^2-s_kZ_k^2)^2\geq0.
\]

Together with \(Z=X^2\) at \(y=1\), this proves all ten coordinate faces.
The exact face table, hash-bound replay, and nonclaims are recorded in
[`SPIN8_DIRAC_OCTET_DETERMINANT_BOUNDARY_RESULTS.md`](../experiments/SPIN8_DIRAC_OCTET_DETERMINANT_BOUNDARY_RESULTS.md).

There is also a disjoint strict-interior theorem. On the complete central core
\([1/4,3/4]^5\), a 32-box exact Bernstein certificate proves

\[
A_0>\sum_{\mu\ne0}|A_\mu|.
\]

Thus every unique physical margin is strictly positive there, a conclusion
stronger than determinant positivity. The proof uses only rational Bernstein
bounds and exact outward dyadic square-root ceilings. It is recorded in
[`SPIN8_DIRAC_OCTET_CORE_DOMINANCE_RESULTS.md`](../experiments/SPIN8_DIRAC_OCTET_CORE_DOMINANCE_RESULTS.md).

An adaptive continuation expands the same exact inequality to
\([1/8,7/8]^5\). Its complete prefix tree has 2,140 certified leaves and 68
refined basis-failure nodes, with no unresolved box through depth four. This
stronger maintained core theorem is recorded in
[`SPIN8_DIRAC_OCTET_EXTENDED_CORE_DOMINANCE_RESULTS.md`](../experiments/SPIN8_DIRAC_OCTET_EXTENDED_CORE_DOMINANCE_RESULTS.md).

The maintained result, replay boundary, and explicit nonclaims are recorded in
[SPIN8_DIRAC_OCTET_DETERMINANT_ENDPOINT_RESULTS.md](../experiments/SPIN8_DIRAC_OCTET_DETERMINANT_ENDPOINT_RESULTS.md).

## 7. What remains open

The three quadratic principal-minor families already share an exact boundary
law. On \(u_d=u_g=0\), direct factorization in \(\mathbb Z[u_e,u_i,y]\)
gives

\[
Q_1=Q_2=Q_3=P_{6,6,12}(u_e,u_i,y)^2.
\]

The common face has multidegree \((12,12,24)\) and 1,414 power terms; its
square root has multidegree \((6,6,12)\) and 226 terms. Thus every quadratic
minor is nonnegative on that complete codimension-two boundary. This identity
also supplies the natural boundary term for the next exact selector
decomposition.

The exact reduction has isolated the remaining proof burden sharply. The
scalar, quadratic, and cubic principal minors are nonnegative on the complete
five-cube, and the determinant is nonnegative on its complete coordinate
boundary and strictly positive on the extended core \([1/8,7/8]^5\). If the
determinant of the Klein-four circulant \(Z\) is also nonnegative throughout
the remaining width-\(1/8\) collars, then
\(Z\succeq0\), hence \(K_8\succeq0\), and the complete adjacent endpoint face
is proved.

That statement has **not** yet been established. In particular:

- native Bernstein negativity of a \(Z\)-minor is not a counterexample;
- preliminary floating-point searches are falsifiers, not certificates;
- positivity of all proper principal minors does not imply the determinant;
- complete coordinate-boundary positivity does not imply strict-interior
  positivity;
- strict positivity on the extended core does not cover the remaining
  width-\(1/8\) boundary collars;
- this five-variable face does not settle the unrestricted seven-variable
  Dirac--Gram inequality.

The degree-matched two-endpoint quotient has now been constructed and exactly
rejected as an over-strong sufficient condition: it is negative at the
rational feasible point
\((u_d,u_e,u_g,u_i,t)=(1/100,0,0,0,99/100)\), where (t=y^2). This is not a
negative determinant value. In contrast, the determinant itself has exact
order-eight leading form (2^{48}F_4^2\geq0) at the remaining equality
corner. The next exact target is therefore a finite-radius max-coordinate
blow-up of (D) itself that retains this nonnegative exceptional divisor. The
complete coordinate-boundary theorem, proper-minor theorems, and local tangent
theorem still do not by themselves imply the determinant inequality in the
strict interior.
