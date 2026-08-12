# Spin(8) triality/direct addressed-memory equivalence

- **Date:** 2026-08-10
- **Status:** exact algebraic equivalence with dynamic scan diagnostics
- **Implementation:** [`spin8_triality_direct_memory_equivalence.py`](../../src/spin8_triality_direct_memory_equivalence.py)
- **Artifact:** [`spin8_triality_direct_memory_equivalence_20260810.json`](../../artifacts/spin8_triality_direct_memory_equivalence_20260810.json)
- **SHA-256:** `c017c62fcbc53a2f4cd36aa2cfeb0c4f12dc7029595ef2189feb3cf67428371c`

## The equivalence

For a unit positive-spinor key \(p\in S^+\), define

\[
M(p):S^-\longrightarrow V,\qquad M(p)n=\beta(p,n),
\]

where \(\beta\) is the maintained Spin(8) triality map. The normed binding
identity gives

\[
M(p)^\mathsf{T}M(p)=I_8.
\]

Because both spaces are eight-dimensional, \(M(p)\) is orthogonal. A direct
slot holding \(n_h\in S^-\) and a triality slot holding
\(m_h=M(p_h)n_h\in V\) therefore contain exactly the same information in two
orthogonal coordinate systems.

For a shared group element \(g\), triality equivariance is the commuting
diagram

\[
V(g)M(p)=M(S^+(g)p)S^-(g).
\]

Consequently shared transport preserves the gauge relation:

\[
n_h' = S^-(g)n_h,
\qquad
p_h' = S^+(g)p_h,
\qquad
m_h'=V(g)m_h=M(p_h')n_h'.
\]

An addressed overwrite replaces both sides by a fresh pair
\(n_h'=n_{\rm new}\) and \(m_h'=M(p_{\rm new})n_{\rm new}\). By induction, the
relation holds after every interleaving of transports and overwrites. Query
unbinding applies \(M(p_h)^\mathsf{T}\) and returns the direct slot exactly.

## Dynamic executable check

The maintained diagnostic uses three batches, eight slots, and 127 steps.
Every step applies a fresh noncommuting Spin(8) transport and overwrites one
random address. Direct slots use the negative-spinor action; triality slots use
the vector action and bound payload. Both transitions are independently prefix
scanned and compared with recurrent execution.

| Diagnostic | Maximum absolute error |
|---|---:|
| binding gauge orthogonality | `4.44e-16` |
| transport/gauge commuting square | `1.22e-15` |
| direct parallel versus recurrent | `1.33e-15` |
| triality parallel versus recurrent | `8.88e-16` |
| triality state versus gauged direct state | `4.27e-15` |
| unbound triality retrieval versus direct slot | `6.55e-15` |
| direct transition associativity | `3.33e-16` |
| triality transition associativity | `3.33e-16` |

Both memories stream exactly 64 scalars. The floating-point residuals above
come from reassociated matrix products and are not deviations from the exact
algebraic identity.

## Consequence for benchmark design

With supplied unit keys, correct shared actions, equal addresses, and equal
slot count, triality binding cannot have a capacity, retrieval-error, or
long-horizon advantage over direct slots. The two systems are conjugate.
Running more seeds on that oracle regime cannot reveal a triality-specific
memory win.

A triality model can still earn credit through a different resource:

- completing an unobserved representation action from shared cross-view data;
- learning or extrapolating an equivariant bilinear drive from fewer examples;
- robustness when the structured prior is correct and the generic tensor is
  underidentified;
- a structured scan kernel with better measured compute or memory behavior;
- a task where the three inequivalent views are actually observed and useful.

Direct slots should otherwise be expected to win on implementation simplicity:
they do not pay bind/unbind work at write and query boundaries.

## Claim boundary

The theorem assumes unit keys and the exact maintained triality tensor. It does
not cover approximate learned keys, a misspecified tensor, raw superposition
without multiplicity slots, overcomplete coding, soft addressing, or partial
action observation. Those are precisely the regimes in which empirical
differences may occur.

This equivalence is independent of the open unrestricted Dirac--Gram global
inequality. The Dirac--Gram program concerns sensing/design optimality; neither
the scan monoid nor this memory conjugacy uses that global theorem.

## Replay

```powershell
$env:PYTHONPATH='src'
python -m spin8_triality_direct_memory_equivalence `
  --output artifacts/spin8_triality_direct_memory_equivalence_20260810.json
python -m pytest -q tests/test_spin8_triality_direct_memory_equivalence.py
```
