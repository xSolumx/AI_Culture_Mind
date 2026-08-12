# Preregistration: Cubic Schur Minor on the Adjacent Endpoint Octet

**Frozen before reconstructing the cubic polynomial — 2026-08-08**

## Scope

On the adjacent Cayley-endpoint face, the unresolved four-by-four Schur block
is the symmetric Klein-four circulant

\[
Z=
\begin{pmatrix}
Z_0&A&B&C\\
A&Z_0&C&B\\
B&C&Z_0&A\\
C&B&A&Z_0
\end{pmatrix}.
\]

The scalar and all three quadratic principal-minor families have separate
global certificates.  This gate concerns only the common three-by-three
principal minor

\[
c_3=Z_0^3-Z_0(A^2+B^2+C^2)+2ABC.
\]

Write

\[
A=\sqrt{s_1}Z_1,
\qquad
B=\sqrt{s_2}Z_2,
\qquad
C=\sqrt{s_3}Z_3,
\]

for the three nontrivial Klein-four modes.  The forced-square monomials obey

\[
s_1s_2s_3=\tau^2,
\]

where

\[
\tau=u_du_eu_gu_i\,y^2
(1-u_d)(1-u_e)(1-u_g)(1-u_i).
\]

Hence the target is the radical-free polynomial

\[
\boxed{
c_3=Z_0^3-Z_0\sum_{j=1}^3s_jZ_j^2+2\tau Z_1Z_2Z_3.
}
\]

The domain is the complete five-cube

\[
(u_d,u_e,u_g,u_i,y)\in[0,1]^5.
\]

## Frozen evidence hierarchy

1. The implementation must verify the identity
   \(s_1s_2s_3=\tau^2\) exactly before constructing the cubic.
2. A native tensor-product Bernstein representation with no negative exact
   coefficient proves \(c_3\ge0\) on the five-cube.
3. Negative native Bernstein controls reject only that certificate basis.
   Their complete boundary support must be recorded before selecting a
   boundary-adapted certificate.
4. Any dyadic atlas must be finite and complete.  Repeated subdivision toward
   an equality point is localization, not proof.
5. If failures localize to the common equality corner, a max-coordinate
   blow-up may replace infinite subdivision only after its chart definition,
   radial order, and coverage argument are stated prospectively.
6. Floating-point optimization is a falsifier only.  A disproof requires a
   feasible point with \(c_3<0\) verified in exact arithmetic.

## Acceptance and rejection

- **Cubic pass:** a domain-wide exact nonnegativity certificate for \(c_3\).
- **Exact disproof:** an exactly verified feasible point with \(c_3<0\).
- **Inconclusive:** neither a complete certificate nor an exact counterexample.

## Resource contract

- Exact stages use at most six FLINT worker threads and less than 16 GiB RAM.
- Expensive stages run in fresh processes and write progress atomically.
- GPU falsification is bounded and is not part of an exact proof.

## Nonclaims

Passing this gate would not prove \(\det Z\ge0\), positivity of the complete
adjacent endpoint octet, the unrestricted seven-variable Dirac--Gram
inequality, or global five-query D-optimality.

## Prospective amendment after the native audit and falsifier

**Added before computing any boundary-selector coefficient — 2026-08-08.**

The native cubic tensor contains 822 negative controls, including 21 interior
controls, so the native basis is not a certificate.  A bounded float64 GPU
screen found no negative value; its optimized minimum approached the common
orthonormal equality corner.  The screen has no proof role.

The first exact replacement will use the already-proved endpoint face at
\(y=1\).  There, the coset term vanishes, \(Z=X^2\), and the maintained proof
of \(X\succeq0\) implies \(Z\succeq0\).  Since the cubic has exact \(y\)-degree
36, decompose

\[
c_3=c_3\big|_{y=1}\,y^{36}+R.
\]

The face term is nonnegative by the exact matrix theorem.  The cubic gate
passes under this amendment only if the complete five-variable remainder
\(R\) has no negative exact Bernstein coefficient.  If it does contain a
negative control, this selector has failed and a further certificate must be
preregistered; the numerical screen cannot repair it.

## Second prospective amendment after the one-endpoint selector failed

**Added before computing the opposite endpoint or a two-endpoint remainder —
2026-08-08.**

The one-endpoint remainder retained 777 negative controls, including 18
interior controls.  The next test will first audit the complete four-variable
face \(c_3|_{y=0}\).  If and only if that face has a nonnegative exact
Bernstein tensor, form

\[
c_3=c_3\big|_{y=0}(1-y)^{36}
   +c_3\big|_{y=1}y^{36}+R_{01}.
\]

The two displayed weights are nonnegative on \([0,1]\), and the \(y=1\) face
is already delegated to the exact \(X\succeq0\) theorem.  This amendment passes
only if the \(y=0\) face and the complete five-variable remainder \(R_{01}\)
both have no negative exact Bernstein coefficient.  Otherwise the
two-endpoint selector is rejected without changing theorem status.

## Third prospective amendment for the \(y=0\) face

**Added after the native \(y=0\) audit and before computing any nested selector
coefficient — 2026-08-08.**

The \(y=0\) face has only 18 negative native controls.  Their exact support is
contained in the union

\[
\{u_i=1\}\cup\{u_d=0\}\cup\{u_g=0\},
\]

with no interior controls.  The frozen cascade is:

1. remove the \(u_i=1\) face using the selector \(u_i^{\deg_{u_i}}\);
2. only if every remaining negative control lies on \(u_d=0\), remove that
   face using \((1-u_d)^{\deg_{u_d}}\);
3. only if every remaining negative control lies on \(u_g=0\), remove that
   face using \((1-u_g)^{\deg_{u_g}}\).

At every stage, both the selected face and the complementary remainder must
have nonnegative exact Bernstein tensors.  If a selected face fails, if a
negative control escapes the preregistered next boundary, or if the final
remainder fails, this cascade is rejected.  A partial cascade is not a
certificate for \(c_3|_{y=0}\).

## Exact factor amendment for the first selected face

**Added after exact factor discovery and before auditing the factor's sign —
2026-08-08.**

FLINT factor discovery returned the structural identity

\[
c_3\big|_{y=0,u_i=1}=64P(u_d,u_e,u_g)^3,
\]

where \(P\) has multidegree \((6,6,4)\).  This explains why the expanded cube
can have negative Bernstein controls even when its base is nonnegative.  The
first selected face may replace native positivity by this factor certificate
only if all of the following hold:

1. exact factorization returns positive content 64 and exactly one primitive
   factor with multiplicity three;
2. exact re-multiplication recovers the complete face polynomial;
3. the primitive factor \(P\) has no negative exact Bernstein coefficient.

Failure of any item rejects the factor certificate.  Later faces in the
cascade remain governed by the preceding prospective amendment.

## Corner amendment for the cubic base

**Added after the base audit and before computing its corner selector —
2026-08-08.**

The primitive base \(P\) has exactly one negative native Bernstein control,
supported on the joint face \(u_d=u_g=0\).  Write

\[
P=P\big|_{u_d=u_g=0}(1-u_d)^6(1-u_g)^4+R_P.
\]

The factor certificate passes only if the one-variable corner polynomial and
the complete three-variable remainder \(R_P\) both have nonnegative exact
Bernstein tensors.  This decomposition must be verified by exact
re-multiplication; one positive component alone is insufficient.

## One-dimensional corner amendment

**Added after reconstructing the univariate corner and before restricting either
half-interval — 2026-08-08.**

The corner polynomial is

\[
17800u^6-123800u^5+337225u^4-444820u^3
+270590u^2-43420u+2809.
\]

Its native degree-six Bernstein form has one negative interior control.  Split
the interval exactly into \([0,1/2]\) and \([1/2,1]\).  The corner passes only
if both exact pullbacks have nonnegative Bernstein tensors; a pass on one half
does not cover the other.

## Exact Sturm amendment for the univariate corner

**Added after the half-interval basis failed and before generating a stored
Sturm chain — 2026-08-08.**

The lower-half Bernstein tensor retains one negative control.  The replacement
certificate is exact univariate real algebra:

1. construct the complete Sturm chain over \(\mathbb Z[u]\);
2. compute exact sign-variation counts at \(u=0\) and \(u=1\), omitting zero
   entries according to Sturm's rule;
3. require variation difference zero, so the polynomial has no root in
   \((0,1)\);
4. require strictly positive values at both endpoints;
5. independently require SymPy's exact interval root count to return zero.

These conditions imply strict positivity on \([0,1]\) by continuity.  A
floating-point minimum or approximate roots do not enter the certificate.

## Nested selector for the second selected face

**Added after the exact Sturm pass and the second-stage audit, before computing
the nested face coefficients — 2026-08-08.**

After the \(u_i=1\) face is removed, selecting the \(u_d=0\) face leaves a
natively nonnegative complementary remainder.  The selected face has four
negative controls, all on \(u_g=0\).  Write that face as

\[
F=F\big|_{u_g=0}(1-u_g)^{\deg_{u_g}}+R_F.
\]

The second selected face passes only if the nested \(u_g=0\) corner and the
complete remainder \(R_F\) are both exactly Bernstein-nonnegative and exact
re-multiplication verifies the identity.  No third selector is needed if the
outer second-stage remainder is already nonnegative.

## Exact factor amendment for the last nested corner

**Added after exact factor discovery and before auditing the remaining
bivariate factor — 2026-08-08.**

The last nested corner factors as

\[
64(1-u_i)Q(u_e,u_i)
\bigl(30u_e^3-135u_e^2+180u_e+53\bigr)^2.
\]

The corner passes only if exact FLINT factorization and re-multiplication
recover: content \(-64\), the linear factor \(u_i-1\), one multiplicity-two
cubic factor, and one remaining multiplicity-one factor \(Q\); and if \(Q\)
has no negative exact Bernstein coefficient on \([0,1]^2\).  The factor
\(-64(u_i-1)=64(1-u_i)\) and the square are then nonnegative.  Any mismatch or
negative control in \(Q\) rejects this certificate.

## Discriminant amendment for the bivariate factor

**Added after exact discriminant discovery and before storing its Sturm
certificates — 2026-08-09.**

Regard \(Q(u_e,u_i)\) as a quadratic in \(u_i\), with leading coefficient
\(A(u_e)\).  Exact symbolic discovery gives

\[
\operatorname{disc}_{u_i}Q
=-3S(u_e)^4P(u_e)^2,
\]

where

\[
S(u)=30u^3-135u^2+180u+53
\]

and \(P\) is the previously displayed sextic.  The factor \(Q\) passes only
if the verifier:

1. extracts the actual coefficients of \(Q\) and reproduces the displayed
   discriminant identity exactly;
2. proves \(A>0\) on \([0,1]\) by an exact Sturm chain, positive endpoint
   values, and an independent exact root count;
3. replays the same strict-positivity certificate for \(P\);
4. proves \(S>0\) on \([0,1]\) by the same exact criterion.

Then the discriminant is strictly negative and the leading coefficient is
strictly positive, so \(Q(u_e,u_i)>0\) for every real \(u_i\).  Approximate
root locations or a floating-point discriminant are not accepted.

## Delegated endpoint amendment

**Added after the complete \(y=0\) selector cascade passed and before
constructing the two-endpoint remainder — 2026-08-09.**

The native Bernstein tensor of the \(y=0\) face is not nonnegative, but the
independent nested certificate now proves that face exactly.  The
two-endpoint stage may therefore accept the \(y=0\) premise from
`spin8_dirac_endpoint_octet_cubic_yzero_20260808.json` only when the source
artifact exists, records a pass, and its SHA-256 is copied into the resulting
report.  Final promotion still requires a verifier that replays the nested
factor, Sturm, discriminant, and Bernstein obligations rather than trusting
the delegated `passed` field alone.

With that premise fixed, define

\[
R=C-C|_{y=0}(1-y)^{36}-C|_{y=1}y^{36}.
\]

The complete cubic passes this route only if exact re-multiplication verifies
the identity and every native tensor-product Bernstein coefficient of \(R\)
on \([0,1]^5\) is nonnegative.  A negative coefficient rejects this selector
basis, not the underlying cubic inequality.

## Endpoint-factor amendment

**Added after the raw two-endpoint remainder produced negative Bernstein
controls and before dividing that remainder — 2026-08-09.**

Because the selector identity makes \(R\) vanish at both \(y=0\) and \(y=1\),
the exact polynomial ring should contain the factor \(y(1-y)\).  Compute

\[
R=y(1-y)\widetilde R
\]

by exact quotient and remainder, require zero division remainder, and verify
the multiplication identity independently.  Since \(y(1-y)\geq0\) on the
unit interval, a nonnegative exact Bernstein tensor for \(\widetilde R\)
certifies \(R\geq0\).  If the quotient retains negative controls, this route
fails and only the quotient—not the artificially zero endpoint factor—may be
used in a later subdivision atlas.

## Dyadic-atlas amendment

**Added after the endpoint-factor quotient retained negative controls and
before evaluating any dyadic pullback — 2026-08-09.**

Using the complete stored list of 712 negative quotient controls, map each
Bernstein index to its lower/upper half signature by comparing twice the index
with the corresponding degree.  The observed support consists of exactly
`00010` (615 controls) and `00001` (97 controls), in coordinate order
\((u_d,u_e,u_g,u_i,y)\).  These two boxes are evaluated first as a diagnostic,
but they cannot prove the theorem by themselves because the original
Bernstein basis has global support.

If both diagnostic boxes pass, evaluate all 32 dyadic half-boxes.  Each box is
pulled back to \([0,1]^5\) using exact integer arithmetic and audited in its
native tensor-product Bernstein basis.  The quotient passes this atlas only
if every box has zero negative exact coefficient.  Results are checkpointed
after each box; resumption may reuse a stored box only when the source
endpoint-artifact SHA-256 agrees.

## First-refinement branch amendments

**Added after the two diagnostic boxes failed and before computing their next
certificates — 2026-08-09.**

On box `00010`, all 172 negative controls lie on the local face \(u_d=0\).
Use the exact selector

\[
Q_{00010}=Q_{00010}|_{u_d=0}(1-u_d)^{18}+R_{00010}.
\]

This branch passes only if exact re-multiplication holds and both the selected
face and the remainder have nonnegative native Bernstein tensors.

On box `00001`, the unique fully interior negative control has index
\((1,1,1,1,33)\), whose child-half signature is again `00001`.  Audit the
single nested box `00001/00001` as a diagnostic before considering a complete
second-level atlas.  This one child cannot certify its parent.

### Exact factor-probe outcome

The selected \(u_d=0\) face from box `00010` has 137,964 power terms and
multidegree \((18,18,17,34)\) in its four active variables.  Exact FLINT
factorization returned content \(-1\) and one exponent-one factor of the same
term count and multidegree.  Thus the face is primitive and irreducible over
\(\mathbb Z\); no square, cube, or lower-degree polynomial factor is available
for the next certificate.  This is a structural negative result, not evidence
that the face changes sign.

The nested diagnostic `00001/00001` completed with 416 negative controls,
including three interior controls.  Therefore the planned complete dyadic
atlas is not promoted: the prerequisite diagnostic failed, and further
subdivision is not assumed to improve monotonically.

## Equality-tangent blow-up gate

**Added after the nested dyadic diagnostic failed and before reconstructing
the cubic Taylor jet — 2026-08-10.**

At the persistent equality point set \(t=1-y\) and expand

\[
C(u_d,u_e,u_g,u_i,1-t)=\sum_{k\geq m}H_k,
\]

where \(H_k\) is homogeneous of total degree \(k\) in the five nonnegative
deviations.  Reconstruct coefficients exactly from the 1,546,277-term FLINT
polynomial, truncating only terms of total degree greater than 12.  Failure to
find a nonzero component by degree 12 is inconclusive and rejects this gate.

For the first nonzero component \(H_m\):

1. verify homogeneity and exact reconstruction of every retained coefficient;
2. factor it exactly over \(\mathbb Z\) and verify re-multiplication;
3. for each of the five max-coordinate charts, set one deviation equal to one
   and audit the resulting four-cube in exact Bernstein form;
4. call the tangent cone nonnegative only if all five chart faces certify, or
   if a separately stated exact factor certificate closes every failed face.

This gate concerns only the exceptional divisor.  It does not prove the full
cubic away from radius zero; a successful tangent theorem must still be
combined with a radial remainder certificate in every chart.

## Tangent-factor amendment

**Added after exact tangent factorization and before auditing the two factors
separately — 2026-08-10.**

The first nonzero component occurs at order six and factors exactly as

\[
H_6=2^{36}F_2F_4,
\]

where \(F_2\) and \(F_4\) are homogeneous factors with respectively 15 and
70 power terms and coordinatewise multidegrees two and four.  Store every
exact coefficient of both factors, verify multiplication back to \(H_6\), and
audit each factor separately on all five max-coordinate chart faces.  The
tangent cone passes this route only if both factors are nonnegative on every
chart.  A failed Bernstein basis for either factor rejects this direct factor
certificate, not the sign of \(H_6\).

## Radical-factor tangent amendment

**Added after the separate native factor audits failed and before storing the
radical-free identity checks — 2026-08-10.**

Let \(d,e,g,i,t\) denote the five deviations and set

\[
L=2d+2e+2g+5i+4t.
\]

Exact symbolic discovery gives

\[
F_2=L^2+16eg-32dt,
\qquad
F_4=F_2^2-64egL^2.
\]

The verifier must reconstruct both identities from the stored factor
coefficients.  It must also verify the manifest copositive decomposition

\[
\begin{aligned}
F_2={}&4(d-2t)^2+4e^2+4g^2+25i^2+8de+8dg+20di\\
&+24eg+20ei+16et+20gi+16gt+40it.
\end{aligned}
\]

For \(a=4\sqrt{eg}\) and \(b=4\sqrt{2dt}\), the second identity implies

\[
F_4=(L-a-b)(L-a+b)(L+a-b)(L+a+b).
\]

The sign certificate is exact AM--GM:

\[
2e+2g\ge4\sqrt{eg}=a,
\qquad
2d+4t\ge4\sqrt{2dt}=b,
\]

so \(L\ge a+b\) on the nonnegative cone.  All four factors are therefore
nonnegative.  This route passes only if the radical-free polynomial identities,
positive content \(2^{36}\), and homogeneity checks all replay exactly.

## First full radial-chart construction gate

**Added after the tangent theorem passed and before constructing a finite-radius
chart — 2026-08-10.**

Start with the \(t=1-y\) pivot. Shift \(y\mapsto1-t\) exactly in FLINT,
then map

\[
t=r,\qquad d=rx_0,\quad e=rx_1,\quad g=rx_2,\quad i=rx_3.
\]

Require exact divisibility by \(r^6\). Rescale \(r=R/4\) using only a common
positive integer denominator-clearing factor. Before any positivity audit,
verify that the \(R=0\) face is a positive integer multiple of the already
certified \(t\)-pivot face of \(H_6\). Record term counts and multidegrees.
Construction failure, a different radius order, or an exceptional-face
mismatch rejects the chart. A successful construction proves no
finite-radius sign until its quotient or an exact face/remainder decomposition
is certified.

## First radial-selector amendment

**Added after the \(t\)-pivot construction passed and before auditing its
13,416,103 Bernstein controls — 2026-08-10.**

The exact quotient has multidegree \((102,18,18,18,18)\), 4,799,931 power
terms, and exceptional face equal to a positive multiple of \(H_6\). First
audit the complete quotient in native Bernstein form. If that basis has a
negative coefficient, form the exact identity

\[
Q(R,x)=Q(0,x)(1-R)^{102}+R_{\mathrm{rem}}(R,x).
\]

The chart passes this route only if the tangent radical-factor proof replays,
the selector identity is exact, and \(R_{\mathrm{rem}}\) has no negative exact
Bernstein coefficient. A failed native quotient and failed selector remainder
reject these bases, not the chart inequality.

## Boundary-support selector amendment

**Added after all five first-selector audits completed and before any selected
face was factored or sign-audited — 2026-08-10.**

The \(u_i\)- and \(t\)-pivot charts certify outright. In the \(u_e\)- and
\(u_g\)-pivot charts, every negative control of the first radial-selector
remainder lies on the exact face \(u_i=0\). In the \(u_d\)-pivot chart, every
negative except twelve lies on \(u_i=0\), and the remaining twelve lie on the
codimension-two face \(u_e=u_g=0\).

Use exact Bernstein face selectors, not another global ansatz. For each
\(u_e/u_g\) chart write

\[
R=R|_{u_i=0}(1-u_i)^{18}+R_\perp.
\]

For the \(u_d\) chart, first apply the same \(u_i=0\) selector and then write
the surviving complement as

\[
R_\perp=R_\perp|_{u_e=u_g=0}
          (1-u_e)^{18}(1-u_g)^{18}+R_{\perp\perp}.
\]

Each identity must hold exactly. The complementary remainders must have no
negative exact Bernstein controls. The selected faces receive independent
factorization and sign audits; support localization alone does not prove them
nonnegative. A failed face basis rejects that certificate route, not the chart
inequality.

## Selected-face atlas amendment

**Added after the first \(u_e\)-chart selected-face factorization completed and
before any dyadic selected-face box was evaluated — 2026-08-10.**

The \(u_i=0\) face factors exactly into a nonzero integer content, a
nonnegative radial monomial, and one four-variable core. After orienting the
core so the total prefactor is positive, its native Bernstein tensor has 83
negative controls. This rejects the native factor basis; it is not a negative
value of the core.

Audit the complete first-level dyadic partition of the four active variables
\((R,u_d,u_{g/e},t)\), retaining the zero-degree \(u_i\) coordinate only for
consistent indexing. All sixteen boxes must be recorded. Any unresolved box
may be subdivided only after its exact path and sign ledger have been frozen.
The \(u_e\) and \(u_g\) faces remain independent until an exact variable
permutation identity is proved.

## Nested selected-face atlas amendment

**Added after all sixteen first-level \(u_e\)-face boxes completed and before
any child of the sole unresolved box was evaluated — 2026-08-10.**

Fifteen boxes certify. The only unresolved active-bit path is `0010`, meaning
lower \(R\), lower \(u_d\), upper \(u_g\), and lower \(t\). Its exact
Bernstein tensor has 16 negative controls. Audit all sixteen children of this
box and retain the complete sibling cover. Further subdivision is permitted
only for explicitly recorded unresolved children.

## Middle-chart symmetry audit amendment

**Added after the nested \(u_e\)-face atlas certified and before the
\(u_g\)-face atlas or any symmetry comparison was evaluated — 2026-08-10.**

The two middle charts have the same multidegree and sign counts, but numerical
or combinatorial resemblance is not a proof identity. Reconstruct both
oriented selected-face cores exactly. Test the identity map and the only
nontrivial permutation preserving the distinct radial, Cayley, and
zero-degree axes: exchange the two degree-18 ratio coordinates. Transfer the
\(u_e\) atlas only if one exact polynomial identity holds. Otherwise retain
the charts as independent obligations.

## Independent \(u_g\) nested-atlas amendment

**Added after all sixteen first-level \(u_g\)-face boxes completed and before
any \(u_g\) nested child was evaluated — 2026-08-10.**

The independent \(u_g\) atlas also certifies fifteen boxes and leaves only
`0010`, again with 16 negative controls. Because the exact middle-chart
permutation audit failed, the \(u_e\) nested certificate cannot be imported.
Audit all sixteen \(u_g\) children independently under the same acceptance
rule.

## \(u_d\)-chart selected-core atlas amendment

**Added after both \(u_d\) selected faces were factored and natively audited,
and before any dyadic box of either core was evaluated — 2026-08-10.**

The \(u_i=0\) selected face is a positive monomial prefactor times a
four-variable oriented core with 351 negative native Bernstein controls. The
residual \(u_e=u_g=0\) selected face is a positive monomial prefactor times a
three-variable oriented core with 12 negative native controls. Audit the
complete first-level dyadic partitions: sixteen boxes over the active axes
\((R,u_e,u_g,t)\) for the first core, and eight boxes over
\((R,u_i,t)\) for the second. Record all siblings and refine only explicitly
unresolved paths.

## Final radial-face selector amendment

**Added after all sixteen first-level \(u_d,u_i=0\) core boxes completed and
before either residual radial face was evaluated — 2026-08-10.**

Fourteen boxes certify. The two unresolved paths are `0000` and `0001`, with
32 and 23 negative controls respectively. Every negative control in both
boxes lies on the local radial face \(R=0\). For each parent use the exact
degree-101 selector

\[
P=P|_{R=0}(1-R)^{101}+P_\perp.
\]

The recorded support implies that \(P_\perp\) is Bernstein-nonnegative once
the identity is verified. Audit the two selected three-variable faces
independently over \((u_e,u_g,t)\), first natively and then, if necessary, on
their complete eight-box dyadic partitions. Do not merge the lower- and
upper-\(t\) parents without an exact identity.

## Central radial-face refinement amendment

**Added after both eight-box radial-face atlases completed and before any
grandchild was evaluated — 2026-08-10.**

Each radial face leaves one child with 12 negative controls. The lower-\(t\)
parent `0000` leaves child `001`; the upper-\(t\) parent `0001` leaves child
`000`. These are adjacent around \(t=1/2\), but their exact polynomials have
120 and 112 terms, so no reflection identity is assumed. Audit all eight
grandchildren of each path independently and retain both complete sibling
covers.

## Finite-radius corner assembly amendment

**Added after every source blow-up and selected-face atlas completed, and
before constructing the compact assembly artifact — 2026-08-10.**

Promote the equality corner only if one verifier hash-binds all source
artifacts and checks the logical cover rather than merely reading their
top-level `passed` fields. It must:

1. replay the stored tangent-factor proof from exact coefficient rows;
2. verify exact radius order six and exceptional-face proportionality in all
   five max-coordinate charts;
3. require native Bernstein positivity for the \(u_i\)- and \(t\)-pivot
   charts;
4. prove that every negative radial-remainder control in the other charts is
   accounted for by the selected \(u_i=0\), or \(u_e=u_g=0\), faces;
5. verify every selector and factorization identity recorded in the boundary
   artifacts;
6. check completeness, parent paths, unresolved cells, and exact sign counts
   for every selected-face atlas and refinement.

The accepted theorem is exactly

\[
C\geq0\quad\text{on}\quad
(u_d,u_e,u_g,u_i,1-y)\in[0,\tfrac14]^5.
\]

It does not certify the remaining siblings of the quotient cell
`00001/00001`, the separate `00010` face, or any larger domain. The verifier
must state which expensive polynomial transformations it trusts and name the
source harnesses required for a full replay.

## Complete `00001` sibling-atlas amendment

**Added after the finite-radius corner assembly passed and before evaluating
any previously untested child of first-level box `00001` — 2026-08-10.**

The corner theorem certifies only child `00001/00001`; it does not imply that
the other 31 children pass. Audit the complete 32-child dyadic partition of
the first-level box. Reuse the corner theorem for the central child only after
the other 31 children have zero negative exact Bernstein coefficients and
their complete sibling list is stored. Failure of any sibling leaves the
parent open and identifies the next explicit refinement target.

## `00010` selected-face atlas amendment

**Added after timing one untouched `00001` sibling and before evaluating any
dyadic child of the `00010` selected face — 2026-08-10.**

The exact selector complement on first-level box `00010` is already
Bernstein-nonnegative. Its remaining obligation is the primitive,
irreducible \(u_d=0\) face with active multidegree \((18,18,17,34)\).
Because this four-dimensional tensor is materially smaller than one
five-dimensional `00001` sibling, audit its complete 16-box dyadic partition
first.

The atlas must reconstruct the face from the original endpoint quotient,
hash-bind the earlier selector artifact, reproduce the stored term count and
multidegree, retain every sibling, and checkpoint after each exact batched
Bernstein transform. A pass closes box `00010` only when all sixteen children
have zero negative controls. Any failed child becomes the sole permitted
refinement target; a favorable subset is not a certificate.

## Complete coarse-atlas amendment

**Added after the `00010` face atlas passed and before evaluating any of the
30 previously untested first-level quotient boxes — 2026-08-10.**

The original negative-control localization selected `00010` and `00001` as
diagnostics; it did not prove their 30 coarse siblings. Construct the complete
32-box first-level atlas. Box `00010` may be delegated to its exact
face-plus-complement certificate. Box `00001` may be delegated only after its
complete 32-child cover passes. Every other coarse box must have zero negative
exact Bernstein controls in its own pullback. Global quotient positivity is
forbidden until the compact assembly verifier confirms both exhaustive
partitions and both delegated source hashes.
