# Higher-Weight Cayley Bottleneck and Compiled Sandwich Walk

**Exact computer-assisted result — 2026-08-17**

**Status:** the original mixed monomial/golden walks have now been certified
through a six-representation band containing dimensions
$8,28,35,56,35,35$. One Hodge-four sector is a provable bottleneck because
the monomial subgroup fixes a unique Cayley-form line. A symmetric compiled
three-letter distribution $N*H*N$ improves the worst certified gap per
macro-step by more than $3\times$ in the vector view and more than
$56/25\times$ in both half-spin views.

**Source:**
[`mixed_monomial_golden_higher_weight.py`](../../src/mixed_monomial_golden_higher_weight.py)

**Artifact:**
[`mixed_monomial_golden_higher_weight_20260817.json`](../../artifacts/mixed_monomial_golden_higher_weight_20260817.json)

**Artifact SHA-256:**
`d5b3fb35092d2fae603c546530502a95e04eb31ac1566035ef866fc7e033b1d6`

## Representations added

The earlier exact rate certificate covered the defining representation,
$\Lambda^2(\mathbb R^8)$, and $\operatorname{Sym}^2_0(\mathbb R^8)$. This
continuation adds

\[
\Lambda^3(\mathbb R^8),\qquad \dim=56,
\]

and uses the standard orientation $e_0\wedge\cdots\wedge e_7$ to split

\[
\Lambda^4(\mathbb R^8)=\Lambda^4_+\oplus\Lambda^4_-,
\qquad \dim\Lambda^4_\pm=35.
\]

Every compound-matrix entry is an exact minor over $\mathbb Q(\sqrt5)$. For a
four-index $I$ with $I<I^c$, the maintained Hodge basis is

\[
b_I^\pm=e_I\pm s(I,I^c)e_{I^c},
\]

where $s(I,I^c)$ is the orientation sign of the concatenation. The replay
checks exact invariance of both Hodge blocks for every generator before using
them.

## Original-walk bounds

The uniform labelled measure remains exactly the one from the low-degree
report: 17 symmetric monomial labels concatenated with three vector-golden or
four half-spin-golden labels. Cross-source equal matrices retain multiplicity.

| View | $8$ radius | $28$ radius | $\operatorname{Sym}^2_0$ radius | $\Lambda^3$ radius | $\Lambda^4_+$ radius | $\Lambda^4_-$ radius |
|---|---:|---:|---:|---:|---:|---:|
| Vector | $<1/3$ | $<5/8$ | $<1/3$ | $<7/25$ | $<1/3$ | $<24/25$ |
| Positive half-spin | $<1/5$ | $<3/4$ | $<3/8$ | $<1/5$ | $<3/8$ | $<99/100$ |
| Negative half-spin | $<1/5$ | $<3/4$ | $<3/8$ | $<1/5$ | $<3/8$ | $<99/100$ |

These bounds use exact positive-definite $LDL^T$ certificates for both
$cI+M$ and $cI-M$, with the exact Frobenius Gram matrix inserted for the
non-orthonormal traceless-symmetric basis. The earlier report contains the
sharper defining bounds $<7/25$ in the vector view and $=4/21$ in the two
half-spin views.

## The exact Cayley-form obstruction

In the orientation-labelled Hodge-minus block, define

\[
\begin{aligned}
\Omega={}&-b^-_{0123}-b^-_{0145}+b^-_{0167}-b^-_{0246}\\
&-b^-_{0257}-b^-_{0347}+b^-_{0356}.
\end{aligned}
\]

Every one of the 17 symmetric monomial steps fixes $\Omega$ exactly. Stacking
their fixed-vector equations produces a $595\times35$ rational system of rank
34. Therefore

\[
\dim\operatorname{Fix}_{\Lambda^4_-}(N)=1.
\]

This is the algebraic reason the previously attractive low-degree gaps were
not representative of the next triality-sensitive sector: most labels in the
union walk do nothing at all to this one-dimensional direction.

Sparse exact Rayleigh witnesses make the obstruction quantitative. In the
vector view, the three-coordinate witness

\[
5b^-_{0123}+4b^-_{0145}+b^-_{0246}
\]

has quotient

\[
\frac{98+\sqrt5}{105}>\frac{19}{20}.
\]

Thus the actual six-representation band gap of the original vector walk is at
most

\[
\frac{7-\sqrt5}{105}.
\]

In both half-spin views, the seven-coordinate witness stored in the artifact
has the particularly clean quotient

\[
\frac{221}{224},
\]

so each original half-spin band gap is at most $3/224$. These are exact lower
bounds on the obstructing operator norm, not sampled eigenvalues.

## Compiled monomial–golden–monomial sandwich

Let $M_N$ be uniform averaging over the 17 monomial labels and $M_H$ uniform
averaging over the view-specific golden labels. The new macro distribution
chooses independently

\[
n_1\sim N,\qquad h\sim H,\qquad n_2\sim N
\]

and applies the word $n_1hn_2$. Its representation operator is

\[
M_{NHN}=M_NM_HM_N.
\]

Because both source measures are symmetric and the outer factors agree, the
macro distribution is symmetric. Exact certificates give:

| View | $8$ radius | $28$ radius | $\operatorname{Sym}^2_0$ radius | $\Lambda^3$ radius | $\Lambda^4_+$ radius | $\Lambda^4_-$ radius | Six-band gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vector | $<1/25$ | $<1/2$ | $<1/10$ | $<1/25$ | $<1/10$ | $<17/20$ | $>3/20$ |
| Positive half-spin | $<1/100$ | $<2/5$ | $<1/10$ | $<1/100$ | $<1/10$ | $<97/100$ | $>3/100$ |
| Negative half-spin | $<1/100$ | $<2/5$ | $<1/10$ | $<1/100$ | $<1/10$ | $<97/100$ | $>3/100$ |

Combining the strict sandwich lower gaps with the original Rayleigh upper gaps
proves

\[
\frac{\Delta_{NHN}}{\Delta_{\mathrm{original}}}>3
\]

in the vector band and

\[
\frac{\Delta_{NHN}}{\Delta_{\mathrm{original}}}>\frac{56}{25}
\]

in both half-spin bands. The vector artifact records the sharper comparison
source $63(7+\sqrt5)/176$.

## Execution interpretation

The macro support has at most

\[
17\cdot3\cdot17=867
\]

vector words or

\[
17\cdot4\cdot17=1156
\]

half-spin words. Storing every resulting $8\times8$ matrix in `float32` would
cost at most 222 kB or 296 kB per view before deduplication. A discrete
compiler can therefore precompute the macro dictionary and apply one selected
matrix at runtime.

Without precompilation, however, one macro consumes three primitive group
multiplications. The theorem compares one macro decision with one original
label decision. It does **not** prove improvement per primitive multiplication,
per FLOP, or per token. For continuous learned generators the finite dictionary
argument does not apply directly.

## Claim boundary and next falsifier

This is still a finite Peter--Weyl band, not a spectral gap on the full
mean-zero $L^2(SO(8))$. It does not prove that $N*H*N$ is the optimal word
distribution, provide a global mixing time, or establish an SSM training or
hardware advantage.

The exact dictionaries and first CPU/CUDA microbenchmark are now completed in
the [macro compiler continuation](MIXED_MONOMIAL_GOLDEN_MACRO_COMPILER_RESULTS.md).
Direct labelled lookup won that bounded endpoint test. The next falsifier is a
chunked scan that separates endpoint-only execution from every-prefix output,
together with the first irreducibles not generated by exterior powers of the
eight-vector. If interior-state reconstruction erases the lookup gain or a new
higher weight has radius near one, the proposed supercharge fails there.

## Replay

From `Spin-Space-Research`:

```powershell
$env:PYTHONPATH = "src"
python src/mixed_monomial_golden_higher_weight.py `
  --output artifacts/mixed_monomial_golden_higher_weight_20260817.json
python -m unittest tests/test_mixed_monomial_golden_higher_weight.py -v
```
