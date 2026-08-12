# Complex- and quaternionic-type Schur scan blocks

> **Later status (2026-08-11):** The caller no longer has to name the type or
> basis for exact irreducible inputs satisfying complete reducibility; see the
> [exact Schur-type detector](SCHUR_TYPE_DETECTION_RESULTS.md). A later
> [reducible isotypic layer](REDUCIBLE_ISOTYPIC_DECOMPOSITION_RESULTS.md)
> constructs and certifies rationally split repeated and mixed blocks. This
> report's canonical block and scan claims remain unchanged.

**Implementation and exact-algebra audit — 2026-08-11**

**Status:** canonical \(\mathbb C\)- and \(\mathbb H\)-type multiplicity blocks
implemented; automatic decomposition of arbitrary representations and model
superiority remain open

**Code:** [`division_schur_scan.py`](../../src/division_schur_scan.py)

**Artifact:**
[`division_schur_scan_20260811.json`](../../artifacts/division_schur_scan_20260811.json)

## Question

The maintained Schur scanner previously implemented only real-type isotypic
blocks. For a real irreducible representation of complex or quaternionic type,
Schur's lemma gives a larger division-algebra commutant. The missing local gate
was whether those multiplicity transitions could be represented and composed
without materializing dense real matrices or accidentally commuting
quaternionic factors.

This was a post-selection implementation audit, not a preregistered model
benchmark. Its acceptance conditions are algebraic completeness for the two
canonical irreps, correct chronological multiplication, scan/recurrent parity,
and gradient parity.

## Construction

Identify the canonical complex-type real irrep with \(\mathbb C\), acted on by
left unit-complex multiplication. Its real commutant is right multiplication by
\(\mathbb C\). Identify the canonical quaternionic-type real irrep with
\(\mathbb H\), acted on by left unit-quaternion multiplication. Its commutant is
right multiplication by \(\mathbb H\).

For \(m\) isotypic copies, store the state as a row vector
\(x\in\mathbb D^m\), where \(\mathbb D\) is \(\mathbb C\) or \(\mathbb H\), and
define

\[
x\longmapsto (g x)M+b,
\qquad M\in\operatorname{Mat}_m(\mathbb D).
\]

Left and right multiplication commute by associativity. If transition 1 is
applied before transition 2, then

\[
g_{21}=g_2g_1,
\qquad
M_{21}=M_1M_2,
\qquad
b_{21}=(g_2b_1)M_2+b_2.
\]

The order \(M_1M_2\) is essential for \(\mathbb H\). The tests explicitly
check \(ij=k\) and \(ji=-k\), so replacing the quaternionic block by a
commutative or reversed product fails exactly.

## Exact commutant audit

The audit constructs integer real matrices for left and right multiplication.
It solves the exact linear centralizer equations

\[
XL_a=L_aX
\]

for the imaginary basis generators and compares the nullspace dimension with
the span of the right-multiplication basis.

| Canonical real irrep | Exact centralizer dimension | Right-basis rank | Exact commutation |
|---|---:|---:|---|
| \(\mathbb C\), real dimension 2 | 2 | 2 | passed |
| \(\mathbb H\), real dimension 4 | 4 | 4 | passed |

Containment plus equal dimension proves that the displayed right-division-
algebra basis is the complete real commutant for each canonical irrep.

## Numerical implementation falsifiers

Float64 tests used batch size 2, sequence length 17, and multiplicity 3.

| Gate | Complex | Quaternionic |
|---|---:|---:|
| composition associativity maximum error | \(2.22\times10^{-16}\) | \(2.22\times10^{-16}\) |
| scan/recurrent maximum error | \(5.55\times10^{-16}\) | \(6.66\times10^{-16}\) |
| gradient maximum error | \(3.47\times10^{-17}\) | \(8.67\times10^{-18}\) |
| left-action/right-multiplicity commutation error | \(1.78\times10^{-15}\) | \(7.11\times10^{-15}\) |

Separate tests materialize the corresponding real linear maps and compare them
with the factored applications. All gates pass.

## Claim boundary

Established:

- canonical complex- and quaternionic-type Schur multiplicity blocks;
- exact completeness of their right-multiplication commutants;
- correct noncommutative chronological composition;
- differentiable logarithmic-depth prefix scan with recurrent and gradient
  parity.

Not established:

- automatic isotypic decomposition or Schur-type detection for an arbitrary
  supplied real representation;
- fused-kernel speed or memory advantages;
- sequence-model quality, sample efficiency, or language-model superiority;
- novelty relative to all representation-aware scan literature.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m division_schur_scan `
  --output artifacts/division_schur_scan_20260811.json
python -m pytest tests/test_division_schur_scan.py -q
```
