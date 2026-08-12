# Spin(8) Cayley-Spectrum Theorem Preregistration

**Frozen:** 2026-08-03, after the seed 20--29 spectrum was explained
symbolically, but before any seed 30--39 optimization or adversarial
orthonormalization search.

## Scientific correction

The five-probe theorem established identifiability. Joint sensor retraction
established reliable conditioning. Neither result proved why the balanced
sensor has its recurring exact spectrum or why no other five-query design can
have larger determinant.

The post-hoc algebraic reduction of the seed 20--29 optimum suggests a sharper
mechanism. Fix the singleton vector query by `Spin(8)` gauge. The remaining two
positive and two negative probes become four vectors in the common eight-real
spin representation of its `Spin(7)` stabilizer. The unique invariant Cayley
four-form supplies the remaining orbit coordinate after their Gram matrix is
fixed.

## Exact theorem target already exposed by symbolic reduction

For the orthonormal canonical family

\[
(v;p_1,p_2;n_1,n_2)
=(e_0;e_0,e_1;e_2,c e_3+s e_4),\qquad c^2+s^2=1,
\]

the Cayley calibration is `c`. The exact characteristic polynomial target is

\[
\begin{aligned}
\chi_{I_c}(\lambda)=-\frac{1}{1024}
&(\lambda-1)^4(\lambda^2-3\lambda+1)\\
&\cdot(c-2\lambda^2+4\lambda-1)^2
(c-2\lambda^2+6\lambda-3)^2\\
&\cdot(c+2\lambda^2-6\lambda+3)^2
(c+2\lambda^2-4\lambda+1)^2\\
&\cdot(2c\lambda-c-2\lambda^3+8\lambda^2-6\lambda+1)\\
&\cdot(2c\lambda-c+2\lambda^3-8\lambda^2+6\lambda-1).
\end{aligned}
\]

It implies

\[
\det I_c=\frac{(1-c^2)^3(9-c^2)^2}{1024}.
\]

The maintained harness must verify this identity in exact rational polynomial
arithmetic, not by floating-point sampling. It must also verify that the
Cayley form is infinitesimally invariant under all 21 generators of the
`Spin(7)` stabilizer and that `|c|<=1` on orthonormal four-frames.

Passing these checks proves the following scoped theorem: among orthonormal
balanced `(2,2,1)` sensors, the Cayley-null orbit `c=0` is globally D-optimal,
with determinant `81/1024`; the calibrated endpoints `c=+/-1` have rank 25.
It does **not** yet prove global optimality over nonorthogonal frames or other
allocations.

## Fresh numerical falsifiers

### A. Allocation-spectrum replication

Untouched seeds 30--39 will optimize the five unordered allocation partitions,
with at least 12 independent restarts per partition. Up to triality
permutation, the frozen targets inferred from the earlier cohort are:

| Allocation | Target determinant | Target `trace(I^-1)` |
|---|---:|---:|
| `(5,0,0)` | singular, rank 25 | undefined |
| `(4,1,0)` | `1/32` | `57.5` |
| `(3,2,0)` | `1/16` | `45.5` |
| `(3,1,1)` | `135/2048` | `45.4` |
| `(2,2,1)` | `81/1024` | `43` |

Every full-rank partition must reproduce its target determinant within
`1e-8`; no candidate may exceed `81/1024 + 1e-10`. This is a fresh replication
and counterexample search, not a proof of the optimizer's global completeness.

### B. Orthonormal-completion falsifier

For the balanced singleton-vector gauge, concatenate the four remaining unit
probes into a `4 x 8` frame `X`. Let `Q` be its row-orthonormal QR completion,
with row signs ignored because information projectors are sign-invariant.

The proposed lemma is

\[
\det I(X)\leq\det I(Q).
\]

It will be attacked by both:

- at least 10,000 deterministic random full-rank frames;
- at least 32 gradient-based adversarial starts maximizing
  `logdet(I(X))-logdet(I(Q))`.

Any determinant advantage above `1e-10` or log-determinant advantage above
`1e-8` falsifies the lemma. No observed violation is evidence only; promotion
to a theorem requires an analytic inequality or an exact sum-of-squares
certificate.

## Interpretation boundaries

- Exact symbolic factorization establishes a theorem only on the orthonormal
  balanced orbit family.
- The fresh allocation cohort may falsify global optimality but cannot prove it.
- The QR experiment is explicitly designed to attack the missing lemma; a
  numerical pass must not be called a proof.
- If the QR lemma is false, the counterexample is the primary result and the
  global-proof route must change.
- No downstream sequence-model advantage follows from this sensor theorem.
