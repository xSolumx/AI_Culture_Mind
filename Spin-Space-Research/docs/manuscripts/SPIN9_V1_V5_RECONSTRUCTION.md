# Coupled Spin(9) \(V_1\oplus V_5\) determinant reconstruction

**Computer-assisted exact theorem — promoted 2026-08-12**  
**Status:** exact characteristic-zero determinant identity and global
\(21/20\) finite-radius bound on the complete coupled \(V_1\oplus V_5\)
Grassmann normal slice; exact optimality of the algebraic candidate and the
unrestricted quotient remain open  
**Code:**
[`spin9_v1_v5_reconstruction.py`](../../src/spin9_v1_v5_reconstruction.py)  
**Boundary blow-up certificate:**
[`spin9_v1_v5_blowup.py`](../../src/spin9_v1_v5_blowup.py)  
**Raw boundary identity certificate:**
[`spin9_v1_v5_boundary_char0.py`](../../src/spin9_v1_v5_boundary_char0.py)  
**Global projective atlas:**
[`spin9_v1_v5_global.py`](../../src/spin9_v1_v5_global.py)  
**Characteristic-zero lift:**
[`spin9_v1_v5_char0.py`](../../src/spin9_v1_v5_char0.py)  
**Theorem assembler:**
[`spin9_v1_v5_theorem.py`](../../src/spin9_v1_v5_theorem.py)  
**Falsification screen:**
[`spin9_v1_v5_screen.py`](../../src/spin9_v1_v5_screen.py)

## Abstract

At the Cayley-null three-plane, the Grassmann normal slice is

\[
V_1\oplus V_5,
\qquad
V_5\cong\operatorname{Sym}_0(3).
\]

The pure-\(V_5\) family already has a global finite-radius determinant theorem.
This note crosses the next algebraic gate: it reconstructs the determinant on
the full coupled slice. In rational invariant coordinates

\[
x=\sqrt2s,
\qquad
p=3u^2+v^2,
\qquad
y=\sqrt2u(v^2-u^2),
\]

the recovered ratio is

\[
R(x,p,y)=\frac{N(x,p,y)}{\delta(x,p,y)^{14}},
\]

where \(N\in\mathbb Z[x,p,y]\) has weighted degree \(84\) for weights
\((1,2,3)\). All 18,600 allowed coefficients are nonzero integers. Twelve
prime fields, a 96-digit CRT modulus, ten held-out Cartan directions per
prime, an unused prime, and both embeddings of \(\sqrt2\) support the
reconstruction. The pure-\(V_1\) and pure-\(V_5\) restrictions reproduce their
maintained exact formulas.

Exact Bernstein exploration identified two rank-loss base points.
Boundary-adapted blow-ups retain the finite column lost by the uniform
compactification. A separate raw \(\mathbb Q(\sqrt2)\) information-matrix
calculation proves both boundary factorizations without using the modular
table, and strict Bernstein atlases place both complete exceptional planes
below \(26/25\).

The finite-radius obstruction is now closed by a projective atlas. Two compact
sign chambers have 312 strictly positive ordinary leaves and eight exact
handoff boxes. Rational inequalities map every handoff box into one of eight
strict local core/low-\(q\) charts. A separate 22-prime raw identity replay
checks both embeddings of \(\sqrt2\), 18 graded invariant-rank gates, and
13,914,692 exact determinant points. An explicit 175-digit CRT product exceeds
twice a characteristic-zero residual coefficient bound. Consequently

\[
\boxed{R(x,p,y)\leq\frac{21}{20}}
\]

on the complete coupled slice. This is a global theorem on the stated normal
slice, not exact maximality of the algebraic symmetric candidate, control of
the second supported \(V_5\), or a theorem on the unrestricted quotient.

## 1. Exact normal-slice coordinates

Write the Cayley-null frame as \(B\), the scalar normal variation as \(Z_s\),
and the two Cartan generators of \(V_5\) as \(Z_a,Z_t\). The scalar variation
has nonzero entries, with zero-indexed rows and columns,

\[
\begin{aligned}
&(1,1,1),(2,2,-1),(5,2,1),(6,1,1),(8,1,-1),\\
&(9,0,-\sqrt2),(11,2,1),(12,2,-1),(14,0,-\sqrt2),(15,1,-1).
\end{aligned}
\]

The column metrics and cross terms are

\[
Z_s^{\mathsf T}Z_s=4I_3,
\qquad
Z_a^{\mathsf T}Z_a=\operatorname{diag}(10,10,4),
\qquad
Z_t^{\mathsf T}Z_t=\operatorname{diag}(2,2,4),
\]

\[
Z_s^{\mathsf T}Z_a=\operatorname{diag}(2,2,-4),
\qquad
Z_s^{\mathsf T}Z_t=
\begin{pmatrix}0&-2&0\\-2&0&0\\0&0&0\end{pmatrix}.
\]

For \(X=B+sZ_s+uZ_a+vZ_t\), the Gram determinant factors as

\[
\begin{aligned}
\det(X^{\mathsf T}X)
={}&\bigl(1+4(s-u)^2+4v^2\bigr)\\
&\times\bigl(1+4s^2+4su-4sv+10u^2+4uv+2v^2\bigr)\\
&\times\bigl(1+4s^2+4su+4sv+10u^2-4uv+2v^2\bigr).
\end{aligned}
\]

The raw scalar coordinate \(s\) is not the correct rational generator.
Putting \(x=\sqrt2s\) removes all radical coefficients and gives

\[
\begin{aligned}
\delta={}&1+8p+20p^2+16p^3-16y^2
+6x^2+24x^2p+8x^2p^2\\
&+12x^4+16x^4p+8x^6
+24xy+80xpy+80x^3y.
\end{aligned}
\]

The orbit domain remains

\[
x\in\mathbb R,
\qquad p\geq0,
\qquad 27y^2\leq2p^3.
\]

## 2. Reconstruction architecture

The maintained observation numerator \(J\) splits into fixed blocks of sizes
\(16\) and \(20\) on the Cartan section. Generic line calculations repeatedly
show blockwise Gram exponents \(10\) and \(12\); their product explains the
observed \(\delta^{22}\) factor. The separate block quotients are not full
stabilizer invariants, so no blockwise invariant factorization is claimed.

After cancellation,

\[
\det J=\det J(B)\,\delta^{22}N(x,p,y),
\qquad
R=\frac{N}{\delta^{14}}.
\]

There are 18,600 monomials

\[
x^c p^a y^b,
\qquad c+2a+3b\leq84.
\]

Set \(x=q\), \(u=u_0q\), and \(v=v_0q\). At each weight, the dependence on
the Cartan direction is a polynomial in

\[
p_0=3u_0^2+v_0^2,
\qquad
y_0=\sqrt2u_0(v_0^2-u_0^2).
\]

One invertible 631-direction matrix therefore recovers every weight layer.
Eighty-five scale nodes recover the line polynomial. Each prime enforces
35,035 forbidden-weight zeros and 850 held-out direction coefficients.

The completed artifact records:

- 12 reconstruction primes;
- a 96-decimal-digit CRT modulus;
- 18,600 nonzero integral coefficients;
- maximum coefficient size of 33 decimal digits;
- an unused-prime replay under both roots of \(2\);
- exact pure-\(V_1\) and pure-\(V_5\) boundary identities.

The full four-worker run completed in 394 seconds on the recorded Windows
workstation.

## 3. Exact boundary controls

At \(p=y=0\), the numerator reduces to

\[
\begin{aligned}
N(x,0,0)={}&(4x^4+8x^3-4x+1)^{10}\\
&\times(4x^4-16x^3+12x^2+8x+1)^3\\
&\times(4x^4-4x^3+6x^2+2x+1)^5
\times(1+2x^2)^6.
\end{aligned}
\]

Since \(\delta(x,0,0)=(1+2x^2)^3\), this is exactly the maintained symmetric
equiangular determinant curve in the Cayley-null graph chart.

At \(x=0\), every coefficient agrees with the independently certified
631-term pure-\(V_5\) numerator. These are exact table identities, not sampled
comparisons.

## 4. Falsification and the remaining positivity problem

Float64 direct-information optimization was used only to reject bad proof
ideas and locate difficult strata.

- Six differential-evolution seeds returned to the pure-\(V_1\) algebraic
  candidate within \(1.7\times10^{-14}\), with the optimized \(V_5\) radius
  numerically collapsing to zero.
- A 30,000-point compactified random screen remained
  \(1.55\times10^{-7}\) below the candidate.
- The tempting pointwise statement
  \(R(x,p,y)\leq R(x,0,0)\) is false. For fixed scalar coordinates where the
  pure-\(V_1\) determinant is small, a large spin-two component can restore a
  ratio near \(1.004\).
- Separate maxima of the \(16\)- and \(20\)-blocks occur at incompatible
  points, so multiplying independent block bounds is too loose.

For exact compactification, use

\[
x=\pm a\rho,
\qquad r=(1-a)\rho,
\qquad \rho=\frac{t}{1-t},
\qquad 0\leq t,a,z\leq1,
\]

with

\[
p=\frac{3+9z^2}{2}r^2,
\qquad
y=\frac{9z^2-1}{2}r^3.
\]

The leading degree-six Gram form factors exactly. For the positive scalar
chart,

\[
\delta_6|_{z=0}=2(2a-1)^2(5a^2-8a+5)^2,
\]

and for the negative scalar chart,

\[
\delta_6|_{z=1}=8(3a-2)^2(9a^2-18a+10)^2.
\]

Thus the joint projective chart has rank-loss base points at
\((+,z,a)=(+,0,1/2)\) and \((-,1,2/3)\). A uniform strict Bernstein atlas for
even the rational \(21/20\) gap cannot close across these zeros without local
blow-up charts. The exploratory tool constructs the exact 614,125 controls
and exposes the unresolved cells; no coupled positivity theorem is promoted
from those failed uniform atlases.

## 5. Boundary-adapted blow-up theorem

The failed uniform chart discards one finite spinor column at each base point.
Keeping that column gives two affine planes. At the positive base point put

\[
x=r+d,
\qquad w=3zr,
\qquad n_A=1+2d^2+2w^2.
\]

At the negative base point put

\[
x=-2r+d,
\qquad k=(1-z)r,
\qquad n_B=1+2d^2-6dk+9k^2.
\]

Because \(\delta\sim\rho^4n_A\) or \(\rho^4n_B\), the boundary numerator is
the coefficient of \(\rho^{56}\) in the reconstructed \(N\). Direct exact
extraction gives 209 terms for family A and 401 for family B, both of total
degree 28. They factor as

\[
R_A(d,w)=\frac{(n_A-2)^6A_1A_2A_3A_4}{4n_A^{14}},
\qquad
R_B(d,k)=\frac{(n_B-2)^6B_1B_2B_3B_4}{8n_B^{14}},
\]

where

\[
\begin{aligned}
A_1={}&4d^4+8d^2w^2+2d^2+4w^4+2w^2+1,\\
A_2={}&4d^4+8d^2w^2+2d^2+4w^4+4w^2+1,\\
A_3={}&8d^4+4d^3+16d^2w^2-4d^2w+7d^2
       +4dw^2-2dw+2d\\
     &\quad+8w^4-4w^3+7w^2-2w+2,\\
A_4={}&8d^4+4d^3+16d^2w^2+4d^2w+7d^2
       +4dw^2+2dw+2d\\
     &\quad+8w^4+4w^3+7w^2+2w+2,
\end{aligned}
\]

and

\[
\begin{aligned}
B_1={}&4d^4-24d^3k+72d^2k^2+2d^2-108dk^3-6dk
       +81k^4+9k^2+1,\\
B_2={}&8d^4-48d^3k+144d^2k^2+4d^2-216dk^3-12dk
       +162k^4+27k^2+2,\\
B_3={}&8d^4-48d^3k+4d^3+144d^2k^2-12d^2k+7d^2
       -216dk^3\\
     &\quad+18dk^2-18dk+2d+162k^4+27k^2+2,\\
B_4={}&8d^4-48d^3k+4d^3+144d^2k^2-24d^2k+7d^2
       -216dk^3\\
     &\quad+54dk^2-24dk+2d+162k^4-54k^3+36k^2-6k+2.
\end{aligned}
\]

These factorizations are independently raw characteristic-zero identities on
the boundary planes. In family A the limiting columns have Gram matrix

\[
\operatorname{diag}(18,18,n_A),
\]

and in family B they have Gram matrix

\[
\operatorname{diag}(36,72,2n_B).
\]

Clearing the information denominators gives

\[
J_A=n_A(A_0+A_1)+18A_2,
\qquad
J_B=2n_BB_0+n_BB_1+36B_2,
\]

where each entry has total degree at most two. The product of the fixed
\(16+20\) block determinants therefore has total degree at most 72. For each
family, the raw determinant product agrees with the displayed factorization on
the 2,701-node lower Newton grid

\[
\{(i,j)\in\mathbb Z_{\geq0}^2:i+j\leq72\}.
\]

That lower set is unisolvent for all bivariate polynomials of total degree at
most 72, so the agreement is a polynomial identity over
\(\mathbb Q(\sqrt2)\), not a numerical screen. The two ordered determinant-row
hashes are recorded in the characteristic-zero artifact. The modular
18,600-coefficient table is not loaded by this replay.

For each of the four sign quadrants, compactify

\[
d=\varepsilon_d a\rho,
\qquad
w\text{ or }k=\varepsilon_w(1-a)\rho,
\qquad
\rho=\frac{t}{1-t}.
\]

After multiplication by \((1-t)^{28}\), the exact gap for \(26/25\) is a
bidegree-\((28,28)\) Bernstein polynomial. Dyadic half-splitting terminates in
leaf counts \((3,3,8,8)\) for A and \((8,2,12,11)\) for B. Every coefficient
on every leaf is a strictly positive integer after the recorded common scale.
Combining the raw identities with these atlases gives

\[
\boxed{R_A(d,w)<\frac{26}{25},\qquad R_B(d,k)<\frac{26}{25}}
\]

on both complete real planes. The symmetric algebraic candidate has ratio

\[
1.048039892262\ldots>\frac{26}{25},
\]

with the final comparison certified in \(\mathbb Q(\sqrt{241})\), not by its
decimal approximation.

## 6. Global finite-radius projective atlas

For each scalar sign, use the joint compactification

\[
x=\pm a\rho,
\qquad r=(1-a)\rho,
\qquad \rho=\frac{t}{1-t},
\qquad 0\leq t,a,z\leq1.
\]

The exact compact gap is

\[
(1-t)^{84}\bigl(21\delta^{14}-20N\bigr).
\]

The positive scalar chamber has 163 strictly positive Bernstein leaves and
four depth-18 handoff boxes. The negative chamber has 149 positive leaves and
four handoff boxes. No unresolved box is discarded. On the positive side the
family-A coordinates satisfy, box by box,

\[
h\leq\frac{192}{1891},
\qquad q=hR\leq\frac{221}{1984}<1.
\]

On the negative side the family-B coordinates satisfy

\[
h\leq\frac{64}{427},
\qquad q=hR\leq\frac{43}{352}<1.
\]

Thus every handoff box is covered exhaustively by the core chart when
\(R\leq1\) or the low-\(q\) toric chart when \(R\geq1\). Both signs of the
local displacement are checked. The eight required local charts terminate in
29 strictly positive leaves: family A contributes \((1,9,1,8)\) and family B
contributes \((1,1,1,7)\) for core negative/positive and low-\(q\)
negative/positive. Prefix-free terminal paths and the full binary-tree count
are replayed by the theorem tests.

This proves the \(21/20\) gap for the reconstructed rational function on the
complete orbit domain. It is stronger than a sampled interior screen and does
not rely on the compact polynomial at its artificial projective zeros.

## 7. Characteristic-zero lift

The raw residual

\[
\det J_{16}\det J_{20}
-65536\cdot262144\,\delta^{22}N(x,p,y)
\]

has total degree at most 216 and is an invariant of
\(V_1\oplus\operatorname{Sym}_0(3)\). Hence it lies in the invariant ring
\(\mathbb Q[x,p,y]\) with weights \((1,2,3)\). The largest shape layer has
3,997 monomials. A deterministic direction set is checked at the 18 graded
gates \(12,24,\ldots,216\); the corresponding prefix ranks grow from 19 to
3,997. After lower weights have vanished, each newly introduced direction
requires only enough nonzero scale nodes to determine the remaining weights.
This reduces a prime embedding to 316,243 raw determinant points without
changing the exact interpolation argument.

For each of 22 primes, both square roots of 2 are checked. The final replay
therefore contains 13,914,692 raw determinant comparisons. At every point it
also checks that the directly computed Gram determinant equals
\(\delta(x,p,y)\).

For the submultiplicative coefficient norm

\[
\lVert a+b\sqrt2\rVert_q=|a|+2|b|,
\]

exact information-row sums give the residual bound

\[
B={}
220272702984017229062697335451285571237991358707936809882983471833807271618980909916841052557515704118374409075849390505743718180043891686161177170305228892260076652255510528.
\]

The 22-prime product is

\[
9848626211911077483055601399563014332881566814685599947018489458129701832889918787888691461325272559783154072087648116972984153227060540264550973170817828546718237266833471977,
\]

which exceeds \(2B\). Both rational and radical residual coefficients must
therefore vanish in characteristic zero. Chaining this identity to the global
atlas proves

\[
\boxed{
\frac{\det I(P)}{\det I(P_{\rm Cayley\text{-}null})}\leq\frac{21}{20}
}
\]

for every finite graph in the complete coupled \(V_1\oplus V_5\) normal
slice.

## 8. Claim boundary and next gate

Established here:

- the exact scalar normal generator and coupled Gram invariant;
- a deterministic 18,600-coefficient modular reconstruction;
- unused-prime conjugacy under both embeddings of \(\sqrt2\);
- exact agreement with both maintained pure-module boundaries;
- explicit identification of the two projective rank-loss base points;
- exact extraction and factorization of both exceptional boundary families
  from the 18,600-coefficient table;
- independent raw characteristic-zero determinant identities for both
  exceptional boundary families;
- strict \(26/25\) Bernstein bounds on both complete blow-up planes;
- an exhaustive compact-plus-local projective atlas for the rational
  \(21/20\) gap;
- the full raw characteristic-zero identity by a 22-prime graded replay and
  explicit coefficient bound;
- the global \(21/20\) determinant theorem on the complete finite-radius
  coupled \(V_1\oplus V_5\) slice.

Not established here:

- exact maximality of the algebraic symmetric candidate even on this slice;
- control of the second supported \(V_5\), the nonpolar Grassmann quotient, or
  unrestricted rank-three frames.

The pure-\(V_1\) stationary equation initially exposed the next exact gate.
Besides lower-degree factors, its derivative contains

\[
16x^8-544x^7+1072x^6-1040x^5-280x^4+520x^3+268x^2+68x+1.
\]

The follow-on exact certificate
[`spin9_v1_candidate_line.py`](../../src/spin9_v1_candidate_line.py) proves that
this octic is the graph-coordinate pullback of the symmetric-curve stationary
quadratic. Explicitly,

\[
c(x)=\frac{-4x(x-1)(2x+1)}{(1+2x^2)^2},
\]

and

\[
c(x)+\frac12=\frac{(2x^2-4x-1)^2}{2(1+2x^2)^2},
\qquad
1-c(x)=\frac{(2x^2+2x-1)^2}{(1+2x^2)^2}.
\]

Thus the complete real graph line maps into \([-1/2,1]\), and its determinant
ratio is exactly the symmetric-curve ratio evaluated at \(c(x)\). The unique
curve maximum at \(c_\star=(-17+\sqrt{241})/24\) therefore proves global
candidate optimality on the pure-\(V_1\) line. The octic has exactly four real
roots; four disjoint rational isolating intervals show they are precisely the
four graph-coordinate preimages of the same maximizing symmetric-curve
parameter. This certificate does not separately reconstruct group elements
between those four graph frames.

The remaining candidate-gap problem is now genuinely mixed: prove the
algebraic candidate gap for \(p>0\) on the coupled orbit domain, with equality
excluded away from the pure-\(V_1\) fiber. That statement is strictly stronger
than both the pure-line theorem and the rational \(21/20\) theorem and remains
open.

## 9. Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m spin9_v1_v5_reconstruction `
  --workers 4 --quiet `
  --output artifacts/spin9_v1_v5_reconstruction_20260811.json
python -m spin9_v1_v5_screen `
  --output artifacts/spin9_v1_v5_screen_20260811.json
python -m spin9_v1_v5_boundary_char0 `
  --workers 4 `
  --output artifacts/spin9_v1_v5_boundary_char0_20260811.json
python -m spin9_v1_v5_blowup `
  --output artifacts/spin9_v1_v5_blowup_20260811.json
foreach ($scalarSign in 1,-1) {
  python -m spin9_v1_v5_global --compact-sign $scalarSign `
    --output "runtime/spin9-v1-v5-global/compact_$scalarSign.json"
}
foreach ($family in "A","B") {
  foreach ($chart in "core","r_infinity_q_low") {
    foreach ($dSign in 1,-1) {
      python -m spin9_v1_v5_global --family $family --chart $chart `
        --d-sign $dSign `
        --output "runtime/spin9-v1-v5-global/${family}_${chart}_${dSign}.json"
    }
  }
}
python -m spin9_v1_v5_global `
  --assemble-report-dir runtime/spin9-v1-v5-global `
  --output artifacts/spin9_v1_v5_global_20260812.json
python -m spin9_v1_v5_char0 `
  --workers 4 `
  --output artifacts/spin9_v1_v5_char0_20260812.json
python -m spin9_v1_v5_theorem `
  --output artifacts/spin9_v1_v5_theorem_20260812.json
python -m pytest tests/test_spin9_v1_v5_reconstruction.py `
  tests/test_spin9_v1_v5_boundary_char0.py `
  tests/test_spin9_v1_v5_blowup.py `
  tests/test_spin9_v1_v5_global.py `
  tests/test_spin9_v1_v5_char0.py `
  tests/test_spin9_v1_v5_theorem.py -q
```

The reconstruction test includes the unused-prime replay under both square-root
embeddings and therefore takes roughly two and a half minutes. The four-worker
raw boundary replay took 176 seconds on the recorded workstation. The blow-up
command chains that raw identity to the strict Bernstein atlases and takes
roughly nine seconds. The uniform gap tool remains exploratory infrastructure;
its failure cells are now retained and handed to the exact local atlas rather
than treated as counterexamples. The two compact chamber replays took about
12.7 and 11.3 minutes on the recorded workstation. The four-worker 22-prime
characteristic-zero lift took 4,995 seconds. The default tests rebuild the
coefficient bound and proof topology; set `SPIN9_FULL_CHAR0_REPLAY=1` to rerun
all 13,914,692 raw determinant comparisons. The float64 screen remains
counterexample-search evidence only.
