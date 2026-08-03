# Continuous-alias family-completion lemmas

Date: 2026-08-03.

These lemmas state the structural mechanisms in
`spin8_continuous_alias.py`. They are elementary consequences of simplex and
inner-product geometry; the scientific question is whether endpoint
optimization finds them and extrapolates them to unseen aliases.

## Alias support is device-independent

Let the semantic centers `c_1,...,c_K` be orthonormal. Let `u` be a unit
nuisance vector orthogonal to their complete span. A radius-`r` alias for
class `k` is

\[
 x = \frac{c_k + r u}{\sqrt{1+r^2}}.
\]

Therefore

\[
 \langle x,c_k\rangle = \frac{1}{\sqrt{1+r^2}},\qquad
 \langle x,c_j\rangle=0\quad(j\ne k).
\]

The radius controls support exactly, independent of the pseudorandom stream's
density or device implementation. Randomness changes only the nuisance
direction within the specified orthogonal sphere.

## Paired endpoints fix the relative encoder gauge

For a write route `w` and query route `q` in the probability simplex, a
single write from zero memory followed by a query returns
`<q,w> v`. Exact retrieval of every nonzero value requires `<q,w>=1`. The
maximum inner product of two probability vectors is one, attained only when
both are the same simplex vertex. Thus paired endpoints align the write and
query encoders without prescribing the common slot permutation.

For normalized delta keys, the same endpoint coefficient is `<q,k>`. Equality
to one implies `q=k` by the equality case of Cauchy--Schwarz. Separate
orthogonal gauges cannot survive exact paired retrieval. A common orthogonal
gauge remains, as it should.

## Unlabeled marginal balance removes class collisions

Assume `K=H`, uniform class frequency, exact paired endpoints, and vertex
routes. The mean route equals the uniform slot distribution if and only if
each slot is selected by exactly one semantic class. Therefore local endpoint
alignment plus unlabeled marginal balance makes the class-to-slot map a
permutation, without assigning any class to a privileged slot.

The distributed-key analogue is also exact. Let the `K=D` unit class keys be
rows of a square matrix `U`. Marginal whitening requires

\[
 \frac1K\sum_k u_k u_k^T = \frac1D I.
\]

With `K=D`, this is `U^T U=I`; the class keys form an orthonormal basis. It
removes cross-key interference for an exact delta update but does not quantize
noisy aliases to that basis. This distinction predicts a robustness gap:
hard slot routing can tolerate within-margin continuous variation, whereas
distributed delta keys inherit small angular errors continuously.

## Scan boundary

The alias encoder, softmax, and training-only balance/whitening losses are
outside the recurrent binary operator. At sequence execution, the encoded
route or key is a function of the current input only. Slot overwrite, delta
overwrite, and additive fast-weight updates are affine maps of the previous
state. Their compositions remain associative and admit both prefix scan and
constant-state recurrence.

No claim here applies to a router that reads the evolving memory or runs a
state-dependent Sinkhorn procedure inside the recurrence.
