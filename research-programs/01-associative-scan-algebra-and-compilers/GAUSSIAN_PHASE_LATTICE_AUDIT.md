# Exact Gaussian phase-lattice audit of a Machin compression

**Maintained exact note — 2026-08-11**

**Status:** one four-term identity and its rational norm-one-torus divisor
relation are certified exactly; the proposed global height-descent
interpretation is corrected and remains open

**Code:** [`gaussian_phase_lattice.py`](src/gaussian_phase_lattice.py)

**Artifact:**
[`gaussian_phase_lattice_20260811.json`](artifacts/gaussian_phase_lattice_20260811.json)

**SHA-256:**
`8f87bac12e2a13fea80a63c094b68c9c891644b122005a38b01d641d349c2c7a`

**Replay:**
[`test_gaussian_phase_lattice.py`](tests/test_gaussian_phase_lattice.py)

## Correct algebraic phase object

Let (K=\mathbb Q(i)), let conjugation be (z\mapsto\bar z), and define the
norm-one torus

\[
T=\ker\left(
\operatorname{Res}_{K/\mathbb Q}\mathbb G_m
\xrightarrow{N_{K/\mathbb Q}}
\mathbb G_m
\right).
\]

The exact phase quotient defined over \(\mathbb Q\) is

\[
\rho:K^\times\longrightarrow T(\mathbb Q),
\qquad
\rho(z)=\frac{z}{\bar z}.
\]

Its kernel is \(\mathbb Q^\times\), and Hilbert 90 identifies
\(K^\times/\mathbb Q^\times\) with \(T(\mathbb Q)\). For
\(z=a+bi\),

\[
\rho(z)=
\frac{a^2-b^2}{a^2+b^2}
+i\frac{2ab}{a^2+b^2},
\]

so both circle coordinates are rational.

This corrects the supplied expression \(z/|z|\). That analytic normalization
generally has coefficients in a further square-root extension. It has the same
argument, but it is not generally a rational point of the algebraic torus.

## What the Spin(2) language means precisely

Over the reals, \(T(\mathbb R)\cong S^1\), and the Spin(2)-to-SO(2) covering is
the squaring morphism \(u\mapsto u^2\). For an integer \(n\), the normalized
analytic spinor

\[
u=\frac{n+i}{\sqrt{n^2+1}}
\]

need not be rational, but its square is the rational torus point

\[
u^2=\rho(n+i)
=\left(\frac{n^2-1}{n^2+1},\frac{2n}{n^2+1}\right).
\]

Consequently its slope is exactly

\[
\frac{2n}{n^2-1}
=\tan\left(2\arctan\frac1n\right).
\]

The statement that \((n+i)^2\) itself lies in a multiplicative Galois
\(-1\) eigenspace is not used: it is false without quotienting by its norm
factor.

## Prime phase divisors

Unique factorization in \(\mathbb Z[i]\) gives the useful lattice. Inert
primes \(p\equiv3\pmod4\) contribute only fixed rational factors. The prime 2
is ramified and contributes torsion through \(1+i\). Every split prime
\(p\equiv1\pmod4\) supplies a conjugate pair \(\pi_p,\bar\pi_p\); the free
phase coordinate is the valuation difference

\[
d_p(z)=v_{\pi_p}(z)-v_{\bar\pi_p}(z).
\]

Thus \(T(\mathbb Q)\) is the fourth roots of unity together with a free
abelian phase lattice indexed by the split rational primes. A Machin identity
becomes an integer relation among the \(d_p\), plus a torsion and positivity
condition.

## Certified four-term identity

For

\[
\beta=(239+i)(7+i)^2(4+i)^2(268+i)^2,
\]

exact Gaussian multiplication gives

\[
\boxed{
\beta=10{,}317{,}661{,}250(1+i).
}
\]

The artifact independently factors the four Gaussian inputs and verifies that
every free split-prime valuation difference cancels. It also verifies

\[
\rho(\beta)=\rho(1+i)=i.
\]

Because the remaining quotient is the displayed positive rational number, the
principal arguments give

\[
\arctan\frac1{239}
+2\arctan\frac17
+2\arctan\frac14
+2\arctan\frac1{268}
=\frac\pi4.
\]

The positivity clause is essential: equality in the norm-one torus alone
determines phase only modulo \(\pi\).

## Exact content of one Alferov-style step

If a residual tangent is (a/b>0) and (q) is the nearest integer to (b/a),
then subtracting \(\arctan(1/q)\) gives

\[
\tan\left(\arctan\frac ab-\arctan\frac1q\right)
=\frac{aq-b}{bq+a}.
\]

Nearest-integer choice proves the elementary Euclidean bound

\[
2|aq-b|\leq a.
\]

The artifact checks both identities on a nonzero residual control. This is a
real descent mechanism on the unreduced numerator, but it is not yet a theorem
that a complete Alferov pipeline strictly decreases Lehmer's measure, minimizes
a geodesic height, terminates optimally, or behaves like LLL.

## Claim boundary and relation to the repository

Established here:

- the exact norm-one-torus reformulation;
- the split-prime phase-divisor lattice;
- the displayed four-term Gaussian factorization and Machin identity;
- the exact tangent update and nearest-integer numerator bound.

Open or not claimed:

- a canonical finite-dimensional “character lattice” for all of
  \(\mathbb Q(i)^\times\);
- strict descent of the displayed Lehmer measure at every greedy step;
- global termination, minimality, optimal compression, or an LLL equivalence;
- novelty relative to the classical Gaussian-integer treatment of Machin
  formulas;
- any consequence for Spin(8), Spin(9), Dirac--Gram sensing, memory, rotor
  models, or Collatz dynamics.

This belongs to Programme 01 because it is an exact arithmetic compiler/descent
identity. The Spin(2) double-cover analogy is precise, but it does not merge
this claim with the exceptional Spin programme.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m gaussian_phase_lattice `
  --output artifacts/gaussian_phase_lattice_20260811.json
python -m pytest tests/test_gaussian_phase_lattice.py -q
```
