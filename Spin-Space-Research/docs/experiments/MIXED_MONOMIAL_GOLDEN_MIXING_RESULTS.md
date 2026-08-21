# Exact Low-Degree Mixing Bounds for the Dense Mixed Groups

**Exact computer-assisted result — 2026-08-17**

**Status:** the fixed symmetric random walks associated with the vector and
two half-spin monomial/golden generating alphabets contract strictly in the
defining, adjoint, and traceless-symmetric representations of $SO(8)$.

**Source:**
[`mixed_monomial_golden_mixing.py`](../../src/mixed_monomial_golden_mixing.py)

**Artifact:**
[`mixed_monomial_golden_mixing_20260817.json`](../../artifacts/mixed_monomial_golden_mixing_20260817.json)

**Artifact SHA-256:**
`0082b9df621dfe4b14c41a26cc914f124cebde37abb38974ed79c19411f7eac9`

## Question

The exact closure result proves

\[
\overline{\langle N,\rho(2.A_5)\rangle}=SO(8)
\]

for the maintained vector and two half-spin views. Density is qualitative: it
does not say how quickly a random word loses directional bias. This experiment
asks the first finite, exactly decidable rate question. Does the averaging
operator contract in the three lowest maintained polynomial representations?

## Fixed probability measures

The monomial alphabet consists of the seven left-octonion operators and two
signed-Fano generators, closed under inverses and deduplicated within that
source. It has 17 steps. Each view contributes its golden $(2,3,5)$ pair and
their inverses, again deduplicated within that source.

The two symmetric source alphabets are then concatenated and sampled uniformly
by label. Cross-source equal matrices retain multiplicity. This distinction is
material:

| View | Labelled alphabet | Distinct matrices | Label weight |
|---|---:|---:|---:|
| Vector | 20 | 20 | $1/20$ |
| Positive half-spin | 21 | 19 | $1/21$ |
| Negative half-spin | 21 | 19 | $1/21$ |

Thus this is a fully specified walk. It is not the uniform measure on the
set-theoretic union in the half-spin cases.

For each representation $\pi$, the exact averaging operator is

\[
M_\pi=\frac1{|S|}\sum_{s\in S}\pi(s).
\]

The labelled alphabets are inverse closed, so every $M_\pi$ is self-adjoint in
its invariant inner product. The contraction radius is therefore the largest
absolute eigenvalue of $M_\pi$.

## Certified bounds

| View | Representation | Certified radius | Certified contraction gap |
|---|---|---:|---:|
| Vector | defining $8$ | $<7/25$ | $>18/25$ |
| Vector | adjoint $\Lambda^2$, $28$ | $<5/8$ | $>3/8$ |
| Vector | traceless $\operatorname{Sym}^2_0$, $35$ | $<1/3$ | $>2/3$ |
| Positive half-spin | defining $8$ | $=4/21$ | $=17/21$ |
| Positive half-spin | adjoint $\Lambda^2$, $28$ | $<3/4$ | $>1/4$ |
| Positive half-spin | traceless $\operatorname{Sym}^2_0$, $35$ | $<3/8$ | $>5/8$ |
| Negative half-spin | defining $8$ | $=4/21$ | $=17/21$ |
| Negative half-spin | adjoint $\Lambda^2$, $28$ | $<3/4$ | $>1/4$ |
| Negative half-spin | traceless $\operatorname{Sym}^2_0$, $35$ | $<3/8$ | $>5/8$ |

All inequalities in the table are strict. No floating-point eigenvalue is an
acceptance condition.

## Defining-representation certificates

In the vector view the mean matrix lies in $\mathbb Q(\sqrt5)^{8\times8}$.
Let $p(x)$ be its characteristic polynomial and let $p^\sigma$ apply the
nontrivial Galois conjugation $\sqrt5\mapsto-\sqrt5$. The rational norm

\[
P(x)=p(x)p^\sigma(x)\in\mathbb Q[x]
\]

has degree 16. Exact Sturm root counting finds all 16 roots strictly inside

\[
\left(-\frac7{25},\frac7{25}\right),
\]

and verifies that neither endpoint is a root. Every eigenvalue of the original
mean is a root of $P$, proving the vector bound.

The positive and negative half-spin means are exactly equal. Their common
characteristic polynomial factors over $\mathbb Q$ as

\[
x\left(x-\frac4{21}\right)\left(x-\frac1{21}\right)q(x),
\]

where

\[
q(x)=x^5-\frac{x^4}{7}-\frac{x^3}{63}
+\frac{5x^2}{3087}+\frac{13x}{194481}-\frac5{1361367}.
\]

Exact Sturm counting places all five roots of $q$ strictly inside
$(-4/21,4/21)$. The displayed linear factor attains $4/21$, so the radius and
gap are exact.

## Exact LDL certificates in dimensions 28 and 35

For the adjoint representation, the increasing-pair basis of
$\Lambda^2(\mathbb R^8)$ is orthonormal up to one common scalar. For every
displayed rational $c$, the replay computes exact unpivoted decompositions of

\[
cI+M_{28},\qquad cI-M_{28}.
\]

All 28 pivots in both decompositions are strictly positive in the fixed real
embedding of $\mathbb Q(\sqrt5)$. Hence $-cI<M_{28}<cI$.

For $\operatorname{Sym}^2_0(\mathbb R^8)$ the implementation uses the rational
basis

\[
E_{ii}-E_{77}\quad(0\le i<7),
\qquad E_{ij}+E_{ji}\quad(i<j).
\]

This basis is not orthonormal. Its exact Frobenius Gram matrix is $G$, with a
$7\times7$ diagonal-two/off-diagonal-one block and 28 remaining diagonal
entries equal to two. The replay first verifies

\[
GM_{35}=(GM_{35})^T
\]

and then proves positive definiteness of

\[
cG+GM_{35},\qquad cG-GM_{35}
\]

through 35 positive exact LDL pivots for each sign. The artifact records a
SHA-256 digest of every canonical pivot sequence so the proof objects can be
replayed without bloating the maintained JSON with enormous expressions.
Signs of $a+b\sqrt5$ are decided exactly by rational comparisons between
$a^2$ and $5b^2$.

## What this establishes—and what it does not

The result turns qualitative density into exact quantitative evidence in
dimensions 8, 28, and 35. In particular, the chosen walks do not merely avoid
finite closure: they erase all bias seen by these low-degree representations
at a substantial one-step rate.

It does **not** prove a spectral gap on the full mean-zero
$L^2(SO(8))$. Peter–Weyl theory contains infinitely many irreducible
representations, and an uncontrolled higher representation could have norm
arbitrarily close to one. It also does not give a total-variation or
Wasserstein mixing time, prove that the uniform label weights are optimal, or
establish any SSM or kernel advantage.

The clean next question is higher-weight coverage: generate irreducible
$SO(8)$ representations in increasing highest weight, exploit triality to
remove duplicates, and search for the worst observed operator norm. A genuine
full spectral-gap theorem would still require a uniform argument over that
infinite family or an applicable expansion theorem with its hypotheses checked
for these exact algebraic generators.

## Replay

From `Spin-Space-Research`:

```powershell
$env:PYTHONPATH = "src"
python src/mixed_monomial_golden_mixing.py `
  --output artifacts/mixed_monomial_golden_mixing_20260817.json
python -m unittest tests/test_mixed_monomial_golden_mixing.py -v
```
