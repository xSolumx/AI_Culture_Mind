# Exact Schur-type detection and basis extraction

> **Later status (2026-08-11):** The reducible-input obligation identified in
> this report is now implemented within an explicit rational-splitting domain;
> see the
> [exact reducible isotypic decomposition layer](REDUCIBLE_ISOTYPIC_DECOMPOSITION_RESULTS.md).
> This report remains the authority for irreducible type and division-basis
> extraction.

**Exact implementation audit — 2026-08-11**

**Status:** real, complex, and quaternionic Schur types detected from supplied
exact generators under an explicit complete-reducibility assumption; general
reducible isotypic decomposition remains open

**Code:** [`schur_type_detector.py`](../../src/schur_type_detector.py)

**Artifact:**
[`schur_type_detection_20260811.json`](../../artifacts/schur_type_detection_20260811.json)

## Question

The preceding division-algebra scan result implemented canonical
\(\mathbb C\)- and \(\mathbb H\)-type blocks, but the caller still had to name
the Schur type and supply its basis. The next local gate was whether exact real
representation generators could determine their own commutant type and expose
an auditable multiplication basis.

This is a post-selection exact implementation audit. It is not a noisy-data
classifier or a model benchmark.

## Exact detector

For supplied rational \(n\times n\) generators \(A_r\), the detector solves

\[
\mathcal C=\{X\in\operatorname{Mat}_n(\mathbb R):XA_r=A_rX\text{ for every }r\}
\]

as an exact rational nullspace. It then applies stronger gates than commutant
dimension alone:

- dimension 1 must be exactly the scalar algebra;
- dimension 2 must contain a traceless \(J\) with
  \(J^2=-aI\), \(a>0\);
- dimension 4 must have scalar anticommutators on its traceless complement, a
  positive-definite exact imaginary norm, and an extracted basis
  \((1,i,j,k)\) satisfying generalized quaternion relations.

The extracted basis includes its complete exact multiplication table. The
quaternion basis need not be orthonormally scaled over \(\mathbb Q\): it may
have \(i^2=-aI\), \(j^2=-bI\), and \(k^2=-abI\). This avoids introducing
unnecessary square roots while retaining an exact algebra isomorphism.

## Logical assumption

Complete reducibility is a declared input, not inferred from arbitrary
matrices. Under that assumption, a reducible representation has a nontrivial
commuting projection, so its commutant cannot be a division algebra. Therefore
an exact \(\mathbb R\), \(\mathbb C\), or \(\mathbb H\) commutant certifies
irreducibility and its real Schur type.

This assumption is standard for finite or compact group representations after
choosing an invariant inner product. It need not hold for a general
nonsemisimple algebra representation, so the implementation refuses to issue a
type when the caller does not supply it.

## Positive controls

| Exact generator family | Real dimension | Commutant dimension | Detected type | Extracted relation |
|---|---:|---:|---|---|
| \(\mathfrak{so}(3)\) vector generators | 3 | 1 | real | scalar commutant |
| realification of the \(U(1)\) line | 2 | 2 | complex | \(J^2=-I\) |
| realification of the \(SU(2)\) quaternionic spinor | 4 | 4 | quaternion | \(i^2=j^2=k^2=-I\), \(ij=k=-ji\) |

All exact gates pass. Non-orthogonal rational changes of basis also preserve
all three detected types. In the changed quaternionic coordinates, the
detector extracts a rational generalized basis with squares
\((-2,-6,-12)I\), demonstrating that it is detecting the algebra rather than
matching canonical matrix entries.

## Direct falsifiers

| Control | Commutant dimension | Required outcome | Result |
|---|---:|---|---|
| split sum of two real lines | 2 | reject rather than call it complex | rejected: traceless square is \(+I\) |
| doubled complex irrep | 8 | reject irreducible classification | rejected |
| canonical complex example without complete reducibility supplied | 2 | remain logically inconclusive | rejected |

The first control is decisive: dimension two by itself does not distinguish
\(\mathbb C\) from the split algebra \(\mathbb R\oplus\mathbb R\).

## Claim boundary

Established:

- exact simultaneous-commutant construction for supplied rational matrices;
- fail-closed real/complex/quaternionic type detection under complete
  reducibility;
- exact division-basis and multiplication-table extraction;
- invariance under tested rational changes of representation basis;
- explicit rejection of split, repeated, and missing-assumption controls.

Not established:

- decomposition of a reducible representation into irreducible or isotypic
  summands;
- robust type detection from approximate, noisy, or floating-point matrices;
- automatic conversion of an arbitrary representation basis into optimized
  scan-kernel coordinates;
- novelty relative to all computational representation-theory systems;
- any sequence-model quality or throughput advantage.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m schur_type_detector `
  --output artifacts/schur_type_detection_20260811.json
python -m pytest tests/test_schur_type_detector.py -q
```

The published artifact SHA-256 is
`e9b5de99a0ee772b49c1265329079d967b73f9cd20a491d52d725ed4d0e6fa14`.
