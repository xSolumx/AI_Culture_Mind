# The binary geometry hidden inside coordinate triality

**Date:** 2026-08-06  
**Status:** exact exhaustive theorem for coordinate probes  
**Verifier:** `src/spin8_coordinate_geometry.py`  
**Artifact:** `artifacts/spin8_coordinate_geometry_20260806.json`  
**Preregistration:** none; this theorem was discovered while extending the
global five-probe certificate

## The structural result

The 24 coordinate probes in the three eight-dimensional Spin(8) triality
representations are not 24 unrelated objects. They are the visible part of one
five-dimensional binary vector space.

Label the three representations by the three nonzero two-bit words

\[
V=01,\qquad S^+=10,\qquad S^-=11,
\]

and label the eight coordinates by the three-bit words `000` through `111`.
A coordinate probe is then one five-bit word `(representation, coordinate)`.

In the repository's exact octonion convention, the support of every triality
product obeys

\[
\rho_{i k j}\ne0\quad\Longleftrightarrow\quad k=i\mathbin{\mathrm{XOR}}j.
\]

The representation colours also XOR:

\[
01\oplus10=11,
\]

and cyclically. Therefore triality contraction between two different views is
literally addition in \(\mathbb F_2^5\), up to a nonzero sign that affects the
value but not the generated coordinate support.

For four- and five-probe multiview sensors, exhaustive exact enumeration shows
that the full invariant triality closure is precisely the nonzero-colour part
of the binary span.

## Sharp five-versus-four classification

Every multiview coordinate sensor was checked, not sampled:

| probes | sensors | binary rank distribution | full closures |
|---:|---:|---:|---:|
| 4 | 10,416 | rank 3: 1,680; rank 4: 8,736 | 0 |
| 5 | 42,336 | rank 3: 672; rank 4: 20,160; rank 5: 21,504 | 21,504 |

Thus:

> **Coordinate five-probe theorem.** A multiview set of five coordinate probes
> globally identifies the shared triality action exactly when its five binary
> labels form a basis of \(\mathbb F_2^5\). No set of four coordinate probes
> can do so.

The statement is global for a full-rank five-set. Its invariant closure contains
all eight coordinate vectors in all three faithful representations, so an
action fixing the probes must be the identity everywhere. This rules out
remote finite ambiguities, not only infinitesimal ones.

## The stabilizer ladder

There are only 141 distinct closures across all 52,752 sensors. Exact rational
rank calculations give:

| binary rank | coordinates in each view | distinct closures | Lie rank | stabilizer dimension |
|---:|---:|---:|---:|---:|
| 3 | 2 | 112 | 20 | 8 |
| 4 | 4 | 28 | 25 | 3 |
| 5 | 8 | 1 | 28 | 0 |

For exact representatives, the verifier constructs the stabilizer brackets,
derived algebra, centre, and Killing form. It finds:

\[
\mathfrak{su}(3)\quad\longrightarrow\quad
\mathfrak{su}(2)\quad\longrightarrow\quad 0.
\]

The eight-dimensional stage is centreless, perfect, and has a negative-definite
Killing form, giving compact type \(A_2\), hence \(\mathfrak{su}(3)\). The
three-dimensional stage has the same exact tests and compact type \(A_1\),
hence \(\mathfrak{su}(2)\).

This makes the observed `8 -> 3 -> 0` loss of ambiguity structural. It is the
classical octonionic stabilizer ladder appearing directly as binary sensor
rank.

## Why the Hamming code appeared

Fix `e0` in `V` and choose four coordinates in `S+`. The five binary probe
labels fail to be independent precisely when the four three-bit coordinate
labels XOR to zero. The 14 such four-subsets are exactly the affine planes of
\(\mathbb F_2^3\).

Those 14 planes are simultaneously:

- the blocks of the Steiner system `S(3,4,8)`;
- the weight-four words of the extended Hamming `[8,4,4]` code;
- exactly the 14 exceptional coordinate sensors whose closure stays
  four-dimensional in each view.

The earlier Hamming observation was therefore not a coincidence. It is forced
by the five-bit triality addition law.

## Plain-language version

Imagine each sensor carries a five-switch barcode. Combining two compatible
sensors flips the switches on which their barcodes differ. Four barcodes can
span at most four independent directions, so they can never reveal the whole
five-direction system. Five barcodes reveal everything exactly when none can
be made by combining the others.

When the barcodes span only three directions, eight continuous motions remain
invisible. At four directions only three remain. At all five directions none
remain. The neural recovery experiments were seeing this exact hidden algebra,
not merely benefiting from a lucky optimizer.

## What is genuinely new here

The classical links among octonions, triality, Hamming codes, and exceptional
Lie groups are established mathematics. The candidate new result is their
joint role as a **complete coordinate-sensor identifiability classification**:

1. the 24 probes admit one explicit \(\mathbb F_2^5\) labelling;
2. triality closure is binary span for every four- and five-probe multiview
   coordinate sensor;
3. binary ranks `3, 4, 5` recover the exact stabilizer ladder
   `SU(3), SU(2), trivial`;
4. five-probe global identifiability becomes ordinary binary basis selection.

This deserves a standalone mathematical note after comparison with the
closest finite-geometric formulations in the literature.

The closest source found in the initial literature audit is Arizmendi and
Herrera's [binary encoding of spinors](https://arxiv.org/abs/1905.10613), which
explicitly encodes Clifford multiplication and Spin(8) triality by bit
operations. Schray and Manogue's
[octonionic triality construction](https://arxiv.org/abs/hep-th/9407179)
supplies the established octonionic representation background. These sources
substantially reduce any novelty claim for the binary encoding itself. The
present candidate contribution is the exhaustive probe-identifiability,
closure, and stabilizer classification built on that encoding.

## Claim boundary

Proved exactly:

- the 64 coordinate support products obey the XOR law;
- all 52,752 multiview coordinate sensor closures of sizes four and five match
  their binary spans;
- no coordinate four-set is globally identifying;
- exactly 21,504 coordinate five-sets are globally identifying;
- all 141 distinct closures have the exact Lie ranks in the table;
- exact representatives realize the compact `SU(3) -> SU(2) -> trivial`
  stabilizer ladder.

Subsequent progress and remaining work:

- the later
  [continuous orbit theorem](SPIN8_CONTINUOUS_PROBE_ORBIT_THEOREM.md) proves
  universal four-probe insufficiency and generic global freedom for every
  mixed five-probe allocation;
- a complete classification of exceptional nonprincipal continuous
  five-probe strata remains open;
- whether the binary geometry extends canonically to an intrinsic matroid or
  building description independent of the chosen octonion basis;
- publication-priority and closest-prior-art determination;
- any claim that this theorem alone improves language modelling.

## Replay

```powershell
$env:PYTHONPATH='src'
python -m spin8_coordinate_geometry `
  --output artifacts/spin8_coordinate_geometry_20260806.json
python -m unittest discover -s tests `
  -p "test_spin8_coordinate_geometry.py" -v
```

The verifier rebuilds the tensor support law, enumerates every sensor, compares
two closure constructions, recomputes all 141 exact rational Lie ranks, and
reconstructs the representative compact Lie-algebra certificates. It does not
trust the artifact's stored `passed` field.
