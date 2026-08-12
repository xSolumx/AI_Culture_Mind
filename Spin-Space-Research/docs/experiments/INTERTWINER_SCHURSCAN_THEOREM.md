# Intertwiner SchurScans: the general triangular theorem

> **Systems boundary, 2026-08-10.** The theorem and its staged implementation
> are unchanged. A later co-moving compiler supplies official FLA timing for a
> transported delta recurrence, not for the general bilinear triangular lift;
> the two claims must remain separate.

- **Date:** 2026-08-07
- **Status:** constructive algebraic theorem with executable parity checks
- **Reference implementation:** `src/intertwiner_schurscan.py`
- **Theorem artifact:** `artifacts/intertwiner_schurscan_20260807.json`
- **Benchmark results:** `docs/experiments/INTERTWINER_SCHURSCAN_BENCHMARK_RESULTS.md`
**Preregistration:** none; this generalization was extracted from the earlier
triality-specific staged scan

## The theorem

Let `U`, `V`, and `W` be finite-dimensional real vector spaces and let

\[
\beta:U\times V\longrightarrow W
\]

be any bilinear map. Consider the triangular recurrence

\[
\begin{aligned}
u_t &= A_tu_{t-1}+a_t,\\
v_t &= B_tv_{t-1}+b_t,\\
w_t &= C_tw_{t-1}+c_t+\beta(u_t,v_t).
\end{aligned}
\]

Then:

1. the complete sequence is computable by two stages of associative affine
   scans;
2. recurrent inference stores exactly
   `dim(U) + dim(V) + dim(W)` scalars;
3. the recurrence admits an exact one-stage homogeneous linear lift on

   \[
   \mathbb R\oplus U\oplus V\oplus(U\otimes V)\oplus W;
   \]

4. if a group acts on `U`, `V`, and `W` and `beta` is an intertwiner, the
   bilinear drive is equivariant without changing the scan proof.

The practical two-stage schedule is

```text
U affine scan ----\
                   beta at each position ---> W affine scan
V affine scan ----/
```

Both stages have logarithmic parallel depth. The number of stages is a fixed
constant independent of sequence length.

### Work and depth are separate claims

The original executable scan used Hillis--Steele.  That tree has logarithmic
dependency depth but \(O(N\log N)\) total matrix-composition work.  The
maintained default is now an ordered Blelloch-style tree.  For

\[
P=2^{\lceil\log_2N\rceil},
\]

it uses \(3P-2=O(N)\) compositions, while preserving logarithmic depth and
the noncommutative chronological order.  The tradeoff is a longer critical
path:

\[
d_{\mathrm{Hillis}}=\lceil\log_2N\rceil,
\qquad
d_{\mathrm{work\text{-}efficient}}
=2\lceil\log_2N\rceil+1.
\]

The source retains both trees so that correctness, operation count, memory,
and wall time can be compared rather than inferred from asymptotic notation.

## Controlled empirical consequence of equivariance

The maintained
[`equivariant-identification gate`](INTERTWINER_SCHURSCAN_EQUIVARIANT_IDENTIFICATION_RESULTS.md)
fits the bilinear drive from endpoints supported on a proper coordinate
subspace. A scalar multiple of the complete one-dimensional intertwiner family
extrapolates to held-out group orbits at float64 roundoff, whereas an
unrestricted minimum-norm tensor trained on the identical endpoints
interpolates the subspace but fails off it. Explicit group augmentation makes
the generic design full rank and closes the gap. Spin(8) and SO(3) both pass,
so the result supports the general equivariant prior rather than an exceptional
triality advantage. It is a teacher-aligned identification result, not a
retrieval benchmark.

## Exact homogeneous lift

Write `q = u tensor v`. One step gives

\[
\begin{aligned}
q'={}&(A\otimes B)q
 +(Au)\otimes b
 +a\otimes(Bv)
 +a\otimes b,\\
w'={}&Cw+c+\widetilde\beta q',
\end{aligned}
\]

where `beta-tilde: U tensor V -> W` is the linearization of the bilinear map.
Every right-hand term is linear in

\[
[1,u,v,q,w].
\]

Therefore each token is one finite matrix on the lifted state. Matrix
composition is associative, so an ordinary prefix scan gives an independent
machine-check of the staged construction.

The lift is a proof device. It is not required in the streaming cache and need
not be materialized in the efficient training implementation.

## The feedback obstruction

The direction of the dependency arrows is load-bearing. To make the
obstruction precise, suppose `w` feeds back linearly into one source while the
other remains affine. If \(d_t\) is the generic polynomial degree of `w`
after step \(t\), then \(d_{t+1}=d_t+1\), giving

\[
2,3,4,5,\ldots.
\]

If `w` feeds linearly into both sources, then \(d_{t+1}=2d_t\), and the
bilinear term generically doubles the degree:

\[
2,4,8,16,\ldots,2^t.
\]

Here "generic" means outside the algebraic set on which the leading
coefficients cancel. Consequently no fixed finite collection of monomials of
bounded degree can give an exact linear lift for these generic cyclic
recurrences at arbitrary length. Special algebras may close by additional
identities, and another non-polynomial coordinate system or algorithm is not
ruled out; associativity cannot be claimed merely from bilinearity.

This gives a clean architectural boundary:

> The displayed acyclic bilinear coupling has a finite staged affine scan.
> The two displayed generic cyclic feedback graphs have no fixed finite
> monomial linear lift.

The frozen 2026-08-06 and 2026-08-07 JSON artifacts used the broader field
name `generic_cyclic_feedback_is_scan_compatible`. That historical label is
superseded by the scoped statement above; the artifact bytes remain unchanged
for provenance. Current code emits both
`generic_cyclic_feedback_has_fixed_finite_monomial_linear_lift: false` and
`all_scan_algorithms_for_cyclic_feedback_ruled_out: false`.

## Nonexceptional control

To separate the universal scan theorem from Spin(8), the executable harness
uses the ordinary SO(3) cross product

\[
\beta(u,v)=u\times v.
\]

It is an equivariant bilinear intertwiner

\[
(Ru)\times(Rv)=R(u\times v),\qquad R\in SO(3).
\]

At float64 precision over random length-31 recurrences, the maintained artifact
reports:

- default work-efficient scan versus recurrence: `3.55e-14` maximum absolute
  error;
- Hillis--Steele versus recurrence: `2.84e-14`;
- direct affine tree versus recurrence: `2.84e-14`;
- homogeneous lift versus recurrence: `4.26e-14`;
- SO(3) equivariance: `1.78e-15`;
- streaming state: 9 scalars;
- proof lift: 19 scalars.

Thus scan compatibility is not a triality-only effect.

These are floating-point discrepancies caused by different parenthesizations.
The displayed recurrence is equal algebraically, not bitwise under finite
precision.

## What Spin(8) adds

The universal theorem supplies scheduling and closure. Spin(8) triality adds
properties the SO(3) control does not:

- three inequivalent eight-dimensional representations tied by one invariant;
- an exact normed-division-algebra bind/unbind operation for unit arguments;
- the five-probe cross-view identifiability structure;
- the binary coordinate geometry and exceptional stabilizer ladder documented
  separately.

Those are the properties that a triality model must exploit to beat matched
generic bilinear or direct-memory baselines. Merely passing scan parity is not
a Spin(8) advantage.

## Why this could be a paper

The contribution is not “we can scan matrix products.” It is a general design
rule for adding structured multiplicative interactions to recurrent models
without silently destroying parallelism:

1. factor the state into representation streams;
2. orient bilinear or multilinear intertwiners along a directed acyclic graph;
3. scan each layer of the graph;
4. keep nonlinear cleanup outside the recurrence;
5. reject cyclic couplings unless a finite closure algebra is proved.

The construction naturally extends to a representation quiver and to higher
multilinear maps. A depth-`d` dependency graph yields `d` staged scans, still
with `O(log N)` sequence depth for fixed `d`.

Publication novelty remains subject to a dedicated literature comparison.
In particular, Fujii and Yamakita's 2026
[Bilinear Input Modulation for Mamba](https://arxiv.org/abs/2604.17221)
explicitly distinguishes sequential bilinear computation from scan-compatible
bilinear variants. It is a mandatory nearby comparison. The candidate distinct
point here is the acyclic multistream intertwiner construction, exact finite
homogeneous lift, and representation-equivariant formulation—not the broad
idea that bilinear modulation can appear in an SSM.

## Claim boundary

Established here:

- exact staged/recurrent equivalence for the displayed triangular recurrence;
- an explicit finite homogeneous lift;
- an independent SO(3) intertwiner instance;
- formal degree growth showing why the two specified generic feedback graphs
  have no fixed finite monomial linear lift.

Not yet established:

- throughput superiority over fused production SSM kernels;
- better retrieval or language modelling than delta-rule/attention baselines;
- a no-go theorem covering every possible nonlinear coordinate transform;
- novelty relative to every prior cascaded-scan, semidirect-product, or
  equivariant recurrent construction.

## Engineering result

The work-efficient implementation has passed non-power-of-two ordering,
autograd, malformed-input, and length-2,048 tests.  On the maintained RTX 2070
SUPER at batch 8 and length 4,096, its homogeneous form replicated a 3.59-fold
forward speedup over the previous Hillis--Steele tensor program while retaining
a same-dtype relative discrepancy below \(8\times10^{-7}\).  This is an eager
PyTorch implementation comparison.  It is not a throughput comparison against
fused production SSM kernels.  Full protocol, CPU results, memory measurements,
and raw hashes are recorded in the benchmark-results document.

## Replay

```powershell
$env:PYTHONPATH='src'
python -m intertwiner_schurscan `
  --output artifacts/intertwiner_schurscan_20260807.json
python -m pytest -q `
  tests/test_intertwiner_schurscan.py `
  tests/test_benchmark_intertwiner_schurscan.py
```
