# The Cubic Schur-Minor Gate on the Adjacent Endpoint Octet

**Computer-assisted exact theorem and certificate-basis audit — updated
2026-08-11**

**Status:** proved on the complete five-cube. Both endpoint faces of the
fourth-order determinant are now proved, while its open \(y\)-interior remains
unresolved.

**Preregistration:**
[`SPIN8_DIRAC_OCTET_CUBIC_PREREGISTRATION.md`](SPIN8_DIRAC_OCTET_CUBIC_PREREGISTRATION.md)

## Statement under study

Let \(Z_0,Z_1,Z_2,Z_3\) be the four radical-free Klein-four coefficients of
the second Schur block. With forced squares \(s_j\) and

\[
\tau=u_du_eu_gu_i\,y^2(1-u_d)(1-u_e)(1-u_g)(1-u_i),
\qquad s_1s_2s_3=\tau^2,
\]

the cubic principal minor is the exact polynomial

\[
C=Z_0^3-Z_0(s_1Z_1^2+s_2Z_2^2+s_3Z_3^2)
  +2\tau Z_1Z_2Z_3.
\]

The forced-radical product was reconstructed independently and agrees with
\(\tau^2\) exactly. The resulting integer polynomial has 1,546,277 power
terms and multidegree \((18,18,18,18,36)\).

## Exact theorem on the \(y=0\) face

The complete four-variable endpoint face satisfies

\[
\boxed{C(u_d,u_e,u_g,u_i,0)\geq0
       \quad\text{for every }(u_d,u_e,u_g,u_i)\in[0,1]^4.}
\]

The proof is a nested exact selector cascade.

First, the selected \(u_i=1\) face factors as

\[
C|_{y=0,u_i=1}=64P(u_d,u_e,u_g)^3.
\]

The only failed native Bernstein control of \(P\) lies on
\(u_d=u_g=0\). Its univariate corner is

\[
P_0(u)=17800u^6-123800u^5+337225u^4-444820u^3
       +270590u^2-43420u+2809.
\]

An exact Sturm chain gives equal variation counts at zero and one, an
independent exact interval root count gives zero roots, and
\(P_0(0)=2809\), \(P_0(1)=16384\). Hence \(P_0>0\) on the unit interval.
Every complementary selector remainder is Bernstein-nonnegative.

The last nested corner factors as

\[
64(1-u_i)Q(u_e,u_i)
\bigl(30u_e^3-135u_e^2+180u_e+53\bigr)^2.
\]

Writing \(Q\) as a quadratic in \(u_i\), exact symbolic arithmetic gives

\[
\operatorname{disc}_{u_i}Q
=-3S(u_e)^4P_0(u_e)^2,
\qquad S(u)=30u^3-135u^2+180u+53.
\]

Exact Sturm certificates prove that the leading coefficient of \(Q\),
\(P_0\), and \(S\) are all strictly positive on \([0,1]\). Consequently
the discriminant is strictly negative and \(Q(u_e,u_i)>0\) for every real
\(u_i\). This proves the selected corner and completes the \(y=0\) face.

## What failed in the interior

The native Bernstein tensor of \(C\) has 822 negative controls, including 21
interior controls. A float64 CUDA falsifier found no negative value; its
smallest normalized value was approximately \(0.0071873\) near a known
equality boundary. This is supporting evidence only.

The exact \(y=1\) face is inherited from the previously proved first Schur
block. Subtracting the two endpoint faces with degree-36 selectors gives an
exact remainder with 712 negative Bernstein controls, including 18 interior
controls. Dividing its forced factor \(y(1-y)\) exactly leaves the same 712
negative controls, including 14 interior controls. These outcomes reject the
native endpoint certificate; they do not falsify \(C\geq0\).

All quotient negatives initially localize to two dyadic signatures:

| First-level box | Negative controls | Interior controls |
|---|---:|---:|
| `00010` | 172 | 0 |
| `00001` | 168 | 1 |

Here the bits refer to lower/upper halves of
\((u_d,u_e,u_g,u_i,y)\). Both diagnostic boxes remain uncertified after one
pullback. On `00010`, the selector

\[
Q_{00010}=Q_{00010}|_{u_d=0}(1-u_d)^{18}+R
\]

has an exactly Bernstein-nonnegative remainder; all 172 obstructions belong
to the selected four-variable face. Exact FLINT factorization finds that face
primitive and irreducible over \(\mathbb Z\), so no square or cube shortcut is
available. The deeper pullback `00001/00001` completed its checkpoint before
the wrapper timeout: it has 416 negative controls, including 3 interior
controls. Thus one more dyadic level does not resolve that branch.

## Exact order-six tangent-cone theorem

At the persistent equality point, put \(t=1-y\). Exact Taylor reconstruction
finds the first nonzero homogeneous component at total degree six:

\[
C(d,e,g,i,1-t)=H_6(d,e,g,i,t)+O(\|(d,e,g,i,t)\|^7).
\]

The 210-term tangent polynomial factors exactly as

\[
H_6=2^{36}F_2F_4,
\]

where \(F_2\) has 15 terms and \(F_4\) has 70. Let

\[
L=2d+2e+2g+5i+4t.
\]

The quadratic factor has the manifest nonnegative decomposition

\[
\begin{aligned}
F_2={}&4(d-2t)^2+4e^2+4g^2+25i^2+8de+8dg+20di\\
&+24eg+20ei+16et+20gi+16gt+40it.
\end{aligned}
\]

The quartic satisfies the exact identities

\[
F_2=L^2+16eg-32dt,
\qquad
F_4=F_2^2-64egL^2.
\]

Set \(a=4\sqrt{eg}\) and \(b=4\sqrt{2dt}\). Then

\[
F_4=(L-a-b)(L-a+b)(L+a-b)(L+a+b).
\]

For nonnegative deviations, AM--GM gives

\[
2e+2g\ge a,
\qquad
2d+4t\ge b,
\]

and therefore \(L\ge a+b\). Every radical factor is nonnegative. Hence

\[
\boxed{H_6\ge0\quad\text{on the complete nonnegative tangent cone}.}
\]

This is a genuine local structural theorem. The next certificate upgrades it
from an infinitesimal statement to a finite-radius theorem.

## Exact finite-radius equality-corner theorem

Put \(t=1-y\). Exact blow-up certificates now prove

\[
\boxed{
C(u_d,u_e,u_g,u_i,y)\geq0
\quad\text{whenever}\quad
(u_d,u_e,u_g,u_i,t)\in[0,\tfrac14]^5.}
\]

This is not an asymptotic claim. Choose any largest nonzero deviation \(m\),
set \(R=4m\), and divide the other four deviations by \(m\). This places the
point in one of five compact charts with

\[
R\in[0,1],\qquad (x_1,x_2,x_3,x_4)\in[0,1]^4.
\]

Every chart has exact radius order six. The exceptional face \(R=0\) is a
positive integer multiple of the already proved \(H_6\). The \(u_i\)- and
\(t\)-pivot quotients are Bernstein-nonnegative directly. The remaining three
charts are closed by exact radial selectors, complete negative-support
partitions, and selected-face atlases:

| Pivot | Exact certificate route |
|---|---|
| \(u_d\) | radial selector; \(u_i=0\) and \(u_e=u_g=0\) faces; one 16-box and one 8-box atlas; two radial-face refinements |
| \(u_e\) | radial selector; \(u_i=0\) face; 16-box atlas plus all 16 children of its sole unresolved box |
| \(u_g\) | same logical cover as \(u_e\), reconstructed independently because the exact permutation audit failed |
| \(u_i\) | native tensor-product Bernstein positivity |
| \(t\) | native tensor-product Bernstein positivity |

The compact assembly verifier hash-binds 19 source artifacts. It recomputes
the tangent factor proof, checks every source hash, selector identity,
negative-support partition, binary atlas, parent path, and exact sign count.
It explicitly records that it trusts the expensive source harnesses' stored
FLINT transformations; those harnesses remain the full replay path.

The theorem covers the formerly unresolved nested cell `00001/00001`, but it
does **not** by itself certify the other 31 children of the first-level
`00001` box. The separate complete sibling atlas below supplies exactly that
missing cover.

## Exact closure of first-level box `00010`

The other first-level obstruction is now closed exactly. On box `00010`, the
selector complement was already Bernstein-nonnegative, leaving only the
primitive \(u_d=0\) face. A complete 16-cell dyadic atlas over its active
variables \((u_e,u_g,u_i,y)\) certifies every cell. Each cell contains 227,430
exact Bernstein controls, and every minimum control is strictly positive.

Thus the face, its selector complement, and hence the complete first-level
box satisfy

\[
\boxed{Q_{00010}\geq0.}
\]

The atlas reconstructs the face from the original endpoint quotient, matches
the frozen 139,328-term, multidegree-\((18,18,17,34)\) source fingerprint, and
hash-binds the selector artifact. Irreducibility was therefore not an
obstruction to positivity; it only ruled out a factorwise proof.

## Exact closure of first-level box `00001`

The second originally diagnosed obstruction is also closed exactly. A
complete 32-child dyadic atlas was run inside first-level box `00001`. The
central child `00001/00001` is delegated to the independently hash-bound
finite-radius corner theorem; each of the other 31 children has a strictly
nonnegative stored exact Bernstein audit. The compact verifier checks the
complete binary child set, rejects duplicates and missing siblings, verifies
the delegated corner source, and checks every stored tensor summary.

Consequently the child cover proves

\[
\boxed{Q_{00001}\geq0.}
\]

This promotion closes the second diagnosed first-level box. It did not by
itself make the native first-level diagnostic a global cover, so the remaining
30 coarse first-level pullbacks were subsequently audited independently.

## Complete coarse cover and final exact assembly

The final coarse computation audits the complete binary first-level partition
of \([0,1]^5\). It delegates boxes `00001` and `00010` only to the independent,
hash-bound certificates above and evaluates the exact tensor-product Bernstein
transform on each of the other 30 boxes. All 32 obligations pass, with no
missing or duplicate path.

The theorem is then assembled from the two exact endpoint faces and the
endpoint-factor quotient \(Q\). Characteristic-zero arithmetic recomputes the
1,546,277-term polynomial \(C\), its forced-radical cancellation, exact
division by \(y(1-y)\), and the zero-remainder identity

\[
C=C|_{y=0}(1-y)^{36}+C|_{y=1}y^{36}+y(1-y)Q.
\]

The \(y=0\) term is nonnegative by the exact selector/Sturm certificate. At
\(y=1\), one has \(Z=X^2\) with \(X\succeq0\), so the cubic principal minor is
nonnegative. The complete 32-box atlas proves \(Q\geq0\), and every selector is
nonnegative on the unit interval. Therefore

\[
\boxed{C(u_d,u_e,u_g,u_i,y)\geq0\quad\text{on }[0,1]^5.}
\]

The compact verifier recomputes the final polynomial identity and the
load-bearing endpoint algebra. It hash-binds and structurally verifies the
atlas summaries; regenerating the coarse atlas remains the independent full
replay of every expensive Bernstein transformation.

## Evidence ledger

| Artifact | SHA-256 | Evidence class |
|---|---|---|
| `spin8_dirac_endpoint_octet_cubic_20260808.json` | `1c76057a7a3812c364762ca44c8abf87c5ec1b87bd8b809e03eb0a1993090c53` | exact reconstruction and failed native basis |
| `spin8_dirac_endpoint_octet_cubic_falsifier_20260808.json` | `1405e9d203025e20e124a6fab764e98a395cc3655ce2828c28f006bd37df75d8` | numerical falsification only |
| `spin8_dirac_endpoint_octet_cubic_yzero_20260808.json` | `3787f09ce7b78b759f82ad5c5fa15544af03ec5887688047194707132f6dea1c` | exact \(y=0\) theorem |
| `spin8_dirac_endpoint_octet_cubic_endpoints_20260808.json` | `1cb031377ad728167996997f0b549b30f17023bbdb0e5c4dd8341d6c8af8960f` | failed endpoint-selector basis |
| `spin8_dirac_endpoint_octet_cubic_atlas_diagnostic_20260809.json` | `0298c7efc28cde89195a18484067c9e89a4d3b1216e925745a20f5671583a6a6` | failed first-level diagnostic boxes |
| `spin8_dirac_endpoint_octet_cubic_boundary_00010_20260809.json` | `938607ad46a6e5c449dcba41d7b6b0b2e739556226461f99d5e588359baed3d5` | exact lower-dimensional localization |
| `spin8_dirac_endpoint_octet_cubic_atlas_nested_00001_20260809.json` | `8077ba5709d21caff840e343fe037c62a3cd9165a2dcc65d7b42018517e75d18` | failed second-level diagnostic box |
| `spin8_dirac_endpoint_octet_cubic_tangent_20260810.json` | `bf71eb67ef08fe02a478b89e376ce34d25625337bd0b5e30911f50bdcf088b66` | exact order-six tangent theorem |
| `spin8_dirac_endpoint_octet_cubic_middle_symmetry_20260810.json` | `92a8b03318422f3a690e74b717a0b9814ddfbe58bd293bc53bba62da8e86ab55` | exact negative result: neither admissible middle-chart permutation is an identity |
| `spin8_dirac_endpoint_octet_cubic_corner_20260810.json` | `d9b8cd5951eba5bb2fc99235f42a9aa7be8943e0e5ef78ab9fed30539694a24e` | exact finite-radius equality-corner theorem and 19-artifact assembly |
| `spin8_dirac_endpoint_octet_cubic_boundary_00010_atlas_20260810.json` | `36ea7303fefada928c6b383b8df622c7b7015eaa8c6c59ceeff387dd639b66b2` | complete exact 16-cell proof of the `00010` selected face |
| `spin8_dirac_endpoint_octet_cubic_atlas_nested_00001_complete_20260810.json` | `7238a7ff759f1053ef8dc5ebf7f9153b8a903de88639cea34d62ba1353e61b7a` | complete exact 32-child proof of first-level box `00001` |
| `spin8_dirac_endpoint_octet_cubic_coarse_atlas_20260811.json` | `1747ea735a9bd9cfe16b392728eeb7b2c89293c6dbbae7056beca746bb461f79` | complete exact 32-box quotient cover |
| `spin8_dirac_endpoint_octet_cubic_certificate_20260811.json` | `86e86cbbba47639b716140abf15234d3a60f155ac25674f9356bd035f34217c0` | final characteristic-zero identity and complete five-cube cubic theorem |

## Claim boundary and next method

This work proves the five-variable cubic principal minor. Separate later work
proves exact reconstruction and both endpoint faces of the fourth-order
determinant, but not its open interior, the complete adjacent endpoint octet,
the unrestricted Dirac--Gram inequality, or global five-query optimality. The
determinant interior is now the sole remaining Schur-minor obstruction on this
adjacent endpoint octet. Its frozen gates and current endpoint result are
recorded in
[`SPIN8_DIRAC_OCTET_DETERMINANT_PREREGISTRATION.md`](SPIN8_DIRAC_OCTET_DETERMINANT_PREREGISTRATION.md)
and
[`SPIN8_DIRAC_OCTET_DETERMINANT_ENDPOINT_RESULTS.md`](SPIN8_DIRAC_OCTET_DETERMINANT_ENDPOINT_RESULTS.md).
