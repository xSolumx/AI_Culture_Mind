# Spin(8) Cayley-Spectrum Theorem Results

**Date:** 2026-08-03

**Preregistration:** `SPIN8_CAYLEY_SPECTRUM_PREREGISTRATION.md`

**Fresh cohort:** seeds 30--39

**Raw artifact:** `spin8_cayley_spectrum_seeds30_39.json`

**Artifact SHA-256:** `de75720ede819d7e658f3877a3f21e1ab872d210ab1e528f5b5f71fc5cc1bb2d`

## Result in one sentence

The recurring balanced Spin(8) spectrum is now explained by an exact
`Spin(7)` Cayley-angle theorem: among orthonormal balanced sensors the
determinant is `(1-c^2)^3(9-c^2)^2/1024`, so the optimum is the Cayley-null
orbit and calibrated Cayley planes are rank-deficient; fresh global falsifiers
found no counterexample, but the unrestricted orthonormal-completion lemma is
still a conjecture.

## Scientific correction

The five-probe result solved identifiability. Joint sensor retraction solved the
observed conditioning failures. The present result explains the exact
conditioning spectrum geometrically; it does not retroactively turn either
earlier gate into a global design theorem.

## Exact reduction

Fix the singleton vector query by `Spin(8)` gauge. Its stabilizer is `Spin(7)`.
Clifford multiplication by the fixed vector identifies the positive and
negative spinor spaces under this stabilizer, so the remaining two positive
and two negative queries become one four-frame in `R^8`.

The maintained Cayley four-form was derived in the repository convention and
annihilated exactly by all 21 infinitesimal stabilizer generators. This matches
the established relationship between triality, the Cayley form's unit comass,
and its `Spin(7)` stabilizer described by
[Katz and Shnider](https://arxiv.org/abs/0801.0283).

On an orthonormal four-frame, the `Spin(7)` action on the oriented
four-plane Grassmannian has one orbit coordinate: its signed Cayley calibration
`c`. An exact representative is

\[
(v;p_1,p_2;n_1,n_2)
=(e_0;e_0,e_1;e_2,c e_3+s e_4),\qquad c^2+s^2=1.
\]

## Exact characteristic law

Exact rational characteristic-polynomial elimination gives

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

At `lambda=0`,

\[
\det I_c=\frac{(1-c^2)^3(9-c^2)^2}{1024}.
\]

Writing `z=c^2`,

\[
\frac{d}{dz}\det I_c
=-\frac{(z-9)(z-1)^2(5z-29)}{1024}<0
\quad\text{for }0\le z<1.
\]

Therefore:

- `c=0` is the unique unoriented orbit maximum, with determinant `81/1024`;
- the frozen degree-28 polynomial is exactly the `c=0` specialization;
- `c=+/-1` gives rank 25;
- the information-optimal four-plane is **Cayley-null**, not
  Cayley-calibrated.

This does not contradict classical calibrated geometry. Cayley calibration
and information-determinant maximization are different extremal problems: the
calibrated endpoints are geometrically distinguished but informationally
rank-deficient here.

This is a theorem over the full orthonormal balanced orbit, not a numerical
fit.

## Exact allocation representatives

Exact coordinate representatives recover every previously inferred partition
value:

| Allocation type | Rank | Determinant | `trace(I^-1)` |
|---|---:|---:|---:|
| `(5,0,0)` | 25 | singular | undefined |
| `(4,1,0)` | 28 | `1/32` | `115/2` |
| `(3,2,0)` | 28 | `1/16` | `91/2` |
| `(3,1,1)` | 28 | `135/2048` | `227/5` |
| `(2,2,1)` | 28 | `81/1024` | `43` |

This proves exact attainability. It does not alone prove each row is the global
maximum of its allocation class.

## Fresh allocation falsifier

Seeds 30--39 optimized all five partition types with 12 restarts each and
1,200 steps per restart.

- partition replication: **10/10 seeds**;
- maximum determinant error from any frozen exact target: below `1e-15`;
- largest determinant observed: `0.07910156250000097`;
- candidates exceeding `81/1024 + 1e-10`: **zero**.

Thus the earlier rational allocation table replicated prospectively. This is a
strong counterexample search, not proof of optimizer completeness.

## Orthonormal-completion falsifier

For each balanced frame `X`, the experiment compared its determinant with the
determinant after row-orthonormal QR completion `Q`.

### Random attack

- deterministic full-rank frames: `10,000`;
- violations: `0`;
- maximum raw-minus-QR determinant: `-3.24e-7`;
- largest sampled absolute Cayley calibration: `0.9913`.

### Gradient adversary

- independent starts: `32`;
- optimization steps: `1,200`;
- violations above the frozen tolerance: `0`;
- maximum final determinant advantage: `5.42e-16`;
- maximum final log-determinant advantage: `4.36e-14`.

The adversary did not settle into a merely negative local basin. Its best
regularized log-determinant advantage moved from `-0.725` at step 1 to
`-4.05e-11` at step 200 and numerical equality at step 1,200. All 32 searches
converged toward the proposed equality manifold.

## Two exact slices of the missing lemma

Two one-correlation deformations were factored symbolically. With
`u=a^2` and `z=c^2`, both contain `(1-u)^3(1-z)^3`, and every remaining factor
is nonincreasing in `u` on `[0,1]^2`. One slice perturbs the repeated-view
orthogonality; the other introduces a cross-view correlation. Both are exactly
maximized at `u=0` for every Cayley orbit.

These are load-bearing analytic slices, but not a proof for a general
four-frame Gram matrix with all six correlations active.

## What is established

1. The Cayley form used by the harness is exactly invariant under the full
   21-dimensional `Spin(7)` stabilizer algebra.
2. The complete characteristic polynomial is proved for every orthonormal
   balanced orbit.
3. The balanced determinant is maximized at the Cayley-null orbit and collapses
   by three rank directions at the calibrated endpoints.
4. Exact representatives exist for all five rational allocation spectra.
5. Fresh allocation, random-frame, and gradient-adversarial searches found no
   counterexample to `81/1024` or to orthonormal completion.

## What remains open

- The unrestricted inequality `det I(X) <= det I(Q)` is not yet proved.
- Exact attainability plus ten-seed optimization does not prove the four
  nonbalanced partition upper bounds.
- Consequently `81/1024` is not yet a global theorem over every nonorthogonal
  five-query design.
- No language-model or downstream memory advantage follows from this result.

## Next mathematical gate

Express the balanced determinant in the invariant ring generated by the
four-frame Gram matrix and Cayley four-form. The desired certificate is either:

1. an explicit factorization or sum-of-squares proof that row-orthonormal
   completion cannot lower determinant; or
2. an exact counterexample, which would supersede the numerical conjecture.

After that lemma, the same machinery must prove the exact upper bounds for
the `(4,1,0)`, `(3,2,0)`, and `(3,1,1)` allocation families.

The strengthened follow-up, including two exact Bernstein-positive slices and
the coupled whitening-flow route, is documented in
`SPIN8_DIRAC_GRAM_RESULTS.md`.
