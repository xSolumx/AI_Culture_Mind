# Design rationale for exact algebraic scalar extensions

**Maintained mathematical design note — 2026-08-11**

## Why this layer exists

The first reducible isotypic compiler deliberately worked over
\(\mathbb Q\). That was a sound initial contract: rational matrices admit
deterministic exact row reduction, rational minimal polynomials, rational
Chinese-remainder projectors, and byte-stable proof artifacts. Its first
important obstruction was not numerical noise but an exact field mismatch.
The concrete Spin(9) Grassmann slice naturally contains \(\sqrt2\).

A rationalization certificate solved that one representation by finding an
explicit change to a rational target. The broader mathematical question is:

> Can the compiler work directly with a real representation whose displayed
> matrices and invariant projectors live in an exact real algebraic field?

The first promoted answer is yes for

\[
K=\mathbb Q(\sqrt2),
\qquad \sqrt2>0.
\]

The chosen positive embedding is part of the field declaration. This matters
because real, complex, and quaternionic Schur gates contain exact sign tests.

## Coefficient field is not representation category

This distinction governs the implementation.

The represented space remains a real vector space \(V\), and the sought
commutant is \(\operatorname{End}_{G,\mathbb R}(V)\). The smaller field
\(K\subset\mathbb R\) records exact coefficients for the supplied matrices and
certificate witnesses. It does not redefine \(V\) as a \(K\)-module and does
not replace real Schur types by division algebras over \(K\).

For a constraint matrix \(M\) with entries in \(K\), exact row reduction over
\(K\) supplies a basis of \(\ker_KM\). Since \(\mathbb R\) is a field
extension of \(K\), scalar extension is flat:

\[
\ker_{\mathbb R}M
\cong \mathbb R\otimes_K\ker_KM.
\]

Thus the nullity computed over \(K\) is the real nullity, and the exact
\(K\)-valued matrices form a valid real basis of the commutant or intertwiner
space. The final division classification remains

\[
\mathbb D\in\{\mathbb R,\mathbb C,\mathbb H\}.
\]

## Where extending the field changes a decomposition

Consider

\[
A=\begin{pmatrix}0&2\\1&0\end{pmatrix},
\qquad A^2=2I.
\]

Over \(\mathbb Q\), the polynomial \(z^2-2\) is irreducible. The rational
compiler correctly refuses to invent rank-one rational projectors. Over
\(K=\mathbb Q(\sqrt2)\), it factors and gives

\[
E_\pm=\frac12I\pm\frac{\sqrt2}{4}A,
\qquad E_\pm^2=E_\pm,
\qquad E_+E_-=0.
\]

The two restricted actions are the real lines with eigenvalues
\(\pm\sqrt2\). This is a genuine scalar-extension result: the projectors do
not exist over the old coefficient field.

The required falsifier is

\[
B=\begin{pmatrix}0&-2\\1&0\end{pmatrix},
\qquad B^2=-2I.
\]

The polynomial \(z^2+2\) does not split in the declared real field. The
two-dimensional real representation remains one complex-type irreducible.
The extension therefore does not blindly split every quadratic commutant.

## Exact ordered-field arithmetic

Membership is checked by coercion into the declared algebraic field. An entry
containing \(\sqrt3\) is rejected under a \(\mathbb Q(\sqrt2)\) declaration.
No generic expression-domain fallback is permitted.

For

\[
x=a+b\sqrt2,
\qquad a,b\in\mathbb Q,
\]

the sign is determined exactly by rational signs and the comparison
\(a^2-2b^2\). The maintained controls include the cancellation-sensitive
values

\[
3-2\sqrt2>0,
\qquad 7-5\sqrt2<0.
\]

This explicit order implementation is preferable to decimal evaluation and
also avoids treating an abstract number field without a chosen real embedding
as automatically ordered.

## Compiler changes

The existing APIs now accept an optional `scalar_extension`:

```python
certificate = decompose_reducible_representation(
    generators,
    assume_completely_reducible=True,
    scalar_extension=sqrt(2),
)
```

The extension propagates through:

1. generator and witness membership;
2. simultaneous commutants;
3. algebra centers;
4. source-target intertwiner spaces;
5. minimal polynomials and factorization;
6. polynomial Chinese-remainder idempotents;
7. irreducible real Schur classification;
8. isotypic grouping and aligned compiler coordinates.

The rational API and stored rational artifacts are unchanged. Algebraic
certificates add field metadata, including the primitive element, degree, and
defining polynomial.

Algebraic nullspaces use SymPy's sparse polynomial-domain matrices over the
declared field. This is not only faster than generic expression simplification;
it also prevents a symbolic expression domain from silently becoming the
arithmetic contract. Every returned vector is substituted back into the
original constraints and simplified exactly.

## Spin(9): direct application

The concrete Cayley-null Grassmann slice can now be compiled in its natural
coordinates. No rational target is supplied first. The exact result is

\[
N_{P_0}\cong V_1\oplus V_5,
\]

with commutant dimension two. After adding the supported
\(\operatorname{Sym}_0(3)\) coefficient module, the same field-aware compiler
returns

\[
V_1\oplus2V_5,
\qquad
\operatorname{End}_{\mathrm{SO}(3)}
\cong\mathbb R\oplus\operatorname{Mat}_2(\mathbb R).
\]

The earlier determinant-\(-1/4\) rationalizing intertwiner remains valuable:
it supplies a small closed-form comparison with canonical coordinates. The
new direct decomposition independently shows that rationalization is no longer
a prerequisite for compiler use.

Potential Spin(9) applications now made technically possible include:

- compiling the two equivalent \(V_5\) Hessian channels directly in their
  native exact coordinates;
- constructing symmetry-legal multiplicity-space operators without manually
  selecting a rational basis;
- applying the same field contract to exact local Dirac information blocks
  whose coefficients lie in a declared quadratic field;
- testing transport of isotypic data along algebraic points of the symmetric
  curve.

None of these compiler capabilities proves the global three-spinor optimum or
the open finite-radius coupled determinant inequality.

## Spin(8): the indispensable control

Spin(8) is not treated as a historical stepping stone. The exact restriction
of the Spin(9) spin module gives

\[
S_{9}\big|_{\operatorname{Spin}(8)}=8_+\oplus8_-.
\]

The chirality involution supplies the two rank-eight central projectors. The
seven adjacent coordinate rotations generate the complete
\(28\)-dimensional \(\mathfrak{so}(8)\) algebra under exact brackets.

The vector and two half-spin modules satisfy the exact Hom-space table

\[
\dim\operatorname{Hom}_{\operatorname{Spin}(8)}(8_i,8_j)
=\delta_{ij},
\qquad i,j\in\{v,+,-\}.
\]

Every module is real Schur type. The same table is recovered after independent
dense \(\mathbb Q(\sqrt2)\) changes of basis. This proves that the algebraic
field layer preserves the three inequivalent triality representations; it
does not identify them or turn triality into multiplicity mixing.

## The precise \(\mathrm{Cl}(3,0)\to\mathrm{Cl}(1,4)\) bridge

Let \(P_0,\ldots,P_4\) be five positive involutions from the maintained
Spin(9) Clifford system. Define

\[
e_0=P_0,
\qquad
e_i=P_0P_i\quad(1\le i\le4).
\]

Then

\[
e_0^2=+I,
\qquad
e_i^2=-I,
\qquad
e_ie_j+e_je_i=0\quad(i\ne j).
\]

The 32 blade images are linearly independent, giving a faithful matrix model
of \(\mathrm{Cl}(1,4)\). Its volume element is a central involution with two
rank-eight sectors. The full algebra acts through two inequivalent
quaternionic irreducibles, consistent with

\[
\mathrm{Cl}(1,4)
\cong\operatorname{Mat}_2(\mathbb H)
\oplus\operatorname{Mat}_2(\mathbb H).
\]

The even algebra has blade rank 16 and the two sectors become equivalent:

\[
\mathrm{Cl}^0(1,4)
\cong\operatorname{Mat}_2(\mathbb H).
\]

Inside it,

\[
c_i=e_0e_i=P_i,
\qquad 1\le i\le3,
\]

are three positive anticommuting generators. Their eight blade images define
an injective copy

\[
\mathrm{Cl}(3,0)
\hookrightarrow
\mathrm{Cl}^0(1,4).
\]

On the faithful 16-real-dimensional module, this embedded algebra acts as four
copies of its four-real-dimensional complex-type spinor. Its commutant is
\(\operatorname{Mat}_4(\mathbb C)\), of real dimension 32.

The dimension ledger must remain visible:

| Algebra | Real dimension |
|---|---:|
| \(\mathrm{Cl}(3,0)\) | 8 |
| \(\mathrm{Cl}^0(1,4)\) | 16 |
| \(\mathrm{Cl}(1,4)\) | 32 |

The maintained eight-coordinate \(\mathrm{Cl}(3,0)\) multivector state is the
regular algebra under Spin(3) conjugation, with type
\(2V_0\oplus2V_1\). The 16-coordinate faithful Clifford module above is a
different representation. The embedding does not produce a same-state model
upgrade from 8 to 16 or 32 coordinates, and it supplies no quality or capacity
claim.

## Generalization ladder

The next extensions should be promoted in this order:

1. additional declared real quadratic fields \(\mathbb Q(\sqrt d)\), each
   with exact sign controls;
2. composita and quadratic towers with an explicit chosen real embedding;
3. general real number fields represented by a primitive polynomial plus an
   isolating interval for the selected root;
4. automatic computation of a smallest useful splitting field;
5. certified approximate-to-exact reconstruction from noisy generators.

Only the first item is implemented, with \(d=2\) as the published gate.
Automatic field discovery, arbitrary algebraic signs, and noisy input remain
open. If a minimal polynomial does not split over the declared field, the
compiler continues to report an unresolved leaf or a valid real division
type; it does not guess a larger field.

## Programme boundaries

- Programme 01 owns the exact field arithmetic and compiler semantics.
- Programme 04 owns the Spin(8), Spin(9), and Clifford representation bridges.
- Programme 05 may consume the Spin(9) isotypic coordinates for sensing proofs,
  but no determinant theorem follows from decomposition alone.
- Programme 06 may study a future signature-extended recurrent state, but the
  current \(\mathrm{Cl}(3,0)\) model remains the maintained contract.
- Programmes 02 and 03 receive no identification, retrieval, or model result
  from exact scalar extension.
- Programme 07 remains standalone; exact number-field tooling is not evidence
  about Collatz dynamics.

Shared algebra crosses these boundaries. Claims do not.

