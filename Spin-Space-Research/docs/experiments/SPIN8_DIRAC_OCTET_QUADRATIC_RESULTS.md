# The Complete Quadratic Schur-Minor Gate on the Adjacent Endpoint Octet

**Exact computer-assisted theorem result — 2026-08-08**

**Status:** all three quadratic principal-minor families proved globally

**Preregistration:**
[`SPIN8_DIRAC_OCTET_QUADRATIC_PREREGISTRATION.md`](SPIN8_DIRAC_OCTET_QUADRATIC_PREREGISTRATION.md)

**Assembly verifier:**
[`spin8_dirac_endpoint_octet_quadratic_certificate.py`](../../src/spin8_dirac_endpoint_octet_quadratic_certificate.py)

**Assembled artifacts:**
[`spin8_dirac_endpoint_octet_quadratic_0_global_20260808.json`](../../artifacts/spin8_dirac_endpoint_octet_quadratic_0_global_20260808.json)

[`spin8_dirac_endpoint_octet_quadratic_1_global_20260808.json`](../../artifacts/spin8_dirac_endpoint_octet_quadratic_1_global_20260808.json)

[`spin8_dirac_endpoint_octet_quadratic_2_global_20260808.json`](../../artifacts/spin8_dirac_endpoint_octet_quadratic_2_global_20260808.json)

**Artifact SHA-256 values:**
`b596f3a845a2f67a7e545c767da51524104980007ff5ac855f7ee7b23c0eddd0`

`21329fbb76c19d665407e51f63936730bcd2b66a7afd4ce065d331c2ab1a5520`

`1e6b96e90f8c55edafdf6fdbf33ece9da40c48e735b4fbe0f321ff64409cbbb7`

## Result

On the adjacent Cayley-endpoint face, the exact octet reduction writes the
remaining Schur block as

\[
Z=X^2-(1-y^2)R^2,
\qquad
(u_d,u_e,u_g,u_i,y)\in[0,1]^5.
\]

Let \(Z_0\) be its identity coefficient and let \(Z_\mu\) be a nontrivial
Klein-four coefficient. The three nontrivial modes are

\[
\mu_1=0011001,
\qquad
\mu_2=0101010,
\qquad
\mu_3=0110011.
\]

The corresponding two-by-two principal-minor condition is

\[
q_\mu=Z_0^2-s_\mu Z_\mu^2,
\]

where \(s_\mu\) is the exact forced-square monomial carried by that mode. The
assembled certificates establish

\[
\boxed{
q_{0011001}(u_d,u_e,u_g,u_i,y)\ge 0
\quad\text{throughout }[0,1]^5.
}
\]

and

\[
\boxed{
q_{0101010}(u_d,u_e,u_g,u_i,y)\ge 0
\quad\text{throughout }[0,1]^5.
}
\]

and

\[
\boxed{
q_{0110011}(u_d,u_e,u_g,u_i,y)\ge 0
\quad\text{throughout }[0,1]^5.
}
\]

This is a domain-wide exact sign theorem. It is not a numerical minimum, a
sampled statement, or an inference from a failed counterexample search.

## Why the native certificate failed

The three polynomials have respectively 233,064, 233,051, and 233,048 power
terms, with 224, 202, and 206 negative controls in their native tensor-product
Bernstein forms. Those controls do **not** imply that any polynomial is
negative: Bernstein coefficients are sufficient, not necessary, for
positivity.

A bounded float64 CUDA falsifier searched 4,096 random points and optimized
eight starts for every physical orientation. It found no negative value. Its
smallest normalized margin, approximately \(0.02032\), approached the known
equality corner rather than an interior counterexample. This screen motivated
the local analysis but contributes no step to the proof.

## Exact domain decomposition

Each proof begins with a dyadic partition of the five-cube. Independently for
all three modes, 30 of 32 first-level boxes are natively
Bernstein-nonnegative. The
two exceptions are
labelled `00001` and `00010`, using the coordinate order

\[
(u_d,u_e,u_g,u_i,y).
\]

For each mode, all 32 children of `00010` certify. Of the 32 children of
`00001`, 31 certify. The sole residual box in all three cases is
`00001/00001`, namely

\[
0\le u_d,u_e,u_g,u_i\le\frac14,
\qquad
\frac34\le y\le1.
\]

Equivalently, all five nonnegative deviations

\[
u_d,\quad u_e,\quad u_g,\quad u_i,\quad 1-y
\]

lie in \([0,\tfrac14]\).

## The five-chart equality blow-up

At a nonzero point of the residual box, choose a largest deviation \(m\). Set

\[
r=4m,
\qquad
x_j=\frac{d_j}{m}\quad(j\ne\text{pivot}).
\]

Then \(r,x_j\in[0,1]\). The five possible choices of largest deviation give
five charts whose union covers the complete residual box. In every chart, the
minor is exactly divisible by \(r^4\). Thus the unresolved equality is a
quartic germ, as predicted independently before the blow-up calculation.

For mode `0011001`, the quotient is certified differently in the five charts:

| Pivot deviation | Exact certificate |
|---|---|
| \(u_d\) | signed-square tangent product, two exact boundary factorizations, nested Bernstein selectors, and a four-box corner atlas |
| \(u_e\) | signed-square tangent product and an exact degree-one selector comparison |
| \(u_g\) | signed-square tangent product and an exact degree-two selector comparison |
| \(u_i\) | native tensor-product Bernstein positivity |
| \(1-y\) | native tensor-product Bernstein positivity |

The two middle charts use the identity

\[
H_M=H_m+F\sum_{j=m}^{M-1}(1-r)^j,
\]

where \(M=68\), \(m=1\) or \(2\), \(F\ge0\) is the exceptional-divisor
polynomial, and \(H_m\) is natively Bernstein-nonnegative. The correction is
therefore nonnegative exactly; changing selector exponent does not change the
polynomial being proved.

The hardest radial axis in the \(u_d\)-chart contains the exact factor

\[
(2t-1)^2\bigl(-56t^3+220t^2+70t+13\bigr).
\]

The cubic has degree-three Bernstein coefficients

\[
13,\qquad \frac{109}{3},\qquad 133,\qquad 247,
\]

so it is strictly positive on \([0,1]\). The square records the only remaining
axis equality. Four exact half-boxes certify the transverse two-variable
remainder around that equality.

### The shared tangent law of the second and third modes

For each of modes `0101010` and `0110011`, four of the five blow-up quotients
are natively Bernstein-nonnegative. More strongly, the exceptional divisors in
their remaining \(u_d\)-pivot charts are identical, including their positive
integer content:

\[
C\,B(e,g,i,t)^2,
\qquad C=2^{160}>0,
\]

where

\[
\begin{aligned}
B={}&4e^2+24eg+20ei+16et+8e+4g^2+20gi+16gt+8g\\
&+25i^2+40it+20i+16t^2-16t+4.
\end{aligned}
\]

For each mode, the radial remainder has negative native controls only on
\(e=g=i=0\). Four exact half-boxes cover the remaining \((r,t)\)-square,
and the complementary five-variable remainder is natively
Bernstein-nonnegative. Thus both charts pass without importing any first-mode
factorization. The exact equality of the two tangent squares is an output of
the independent reconstructions, not an assumption in either preregistration.

## What the lightweight verifier checks

The assembly verifier recomputes:

1. the SHA-256 hash of every source artifact;
2. completeness of all three binary atlases and their exact unresolved paths;
3. the stored exact Bernstein sign counts of every accepted atlas cell;
4. radius-order four and all five pivot acceptance predicates;
5. the load-bearing stored factor, selector, and comparison identities.

It deliberately states what it trusts: the exact coefficient arrays and
power-to-Bernstein transforms produced by the source harnesses. A full replay
uses
[`spin8_dirac_endpoint_octet_quadratic.py`](../../src/spin8_dirac_endpoint_octet_quadratic.py)
and
[`spin8_dirac_endpoint_octet_blowup.py`](../../src/spin8_dirac_endpoint_octet_blowup.py).

## Plain-language interpretation

The unresolved polynomial was not negative. Its first coordinate system was
poorly adapted to a point where several directions vanish together. Ordinary
dyadic boxes isolated that point; a blow-up then recorded the *relative rates*
at which the five deviations approach zero. In those relative coordinates,
the apparent sign ambiguity separates into squares, positive one-variable
factors, and finitely many nonnegative Bernstein tensors.

This is a useful methodological result as well as a theorem slice: when an
exact positivity certificate fails only near a high-order equality, refine the
ordinary domain until the obstruction is isolated, then replace infinite
subdivision by a finite max-coordinate blow-up.

## Nonclaims and next gate

These results close all three quadratic principal-minor conditions required for
\(Z\succeq0\). They do not prove:

- the cubic principal minor or \(\det Z\);
- positivity of the complete adjacent endpoint octet;
- the unrestricted seven-circle Dirac--Gram inequality;
- global five-query D-optimality.

The next exact gate is the cubic principal minor. It must be reconstructed and
audited independently; three nonnegative quadratic minors do not imply the
cubic or determinant conditions.
