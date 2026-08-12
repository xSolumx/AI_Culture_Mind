# Exact reducible isotypic decomposition over rational splitting data

**Exact implementation and representation-theory audit — 2026-08-11**

**Status:** certified decomposition of supplied rational, completely reducible
representations whenever exact rational commutant idempotents expose the
irreducible summands; fail-closed for unresolved rational splittings

**Code:**
[`reducible_isotypic_decomposition.py`](../../src/reducible_isotypic_decomposition.py)

**Artifact:**
[`reducible_isotypic_decomposition_20260811.json`](../../artifacts/reducible_isotypic_decomposition_20260811.json)

**Replay:**
[`test_reducible_isotypic_decomposition.py`](../../tests/test_reducible_isotypic_decomposition.py)

> **Concrete Spin(9) extension (2026-08-11).** The later
> [slice-to-isotypic bridge](SPIN9_SLICE_ISOTYPIC_BRIDGE_RESULTS.md) proves that
> the rational \(V_1\oplus2V_5\) fixture used below is exactly reached from
> the geometric \(\mathbb Q(\sqrt2)\) Grassmann slice plus the supported
> coefficient module. At the date of this result the compiler itself remained
> rational and fail-closed on unsupported scalar extensions. The later
> [native algebraic-field result](ALGEBRAIC_ISOTYPIC_DECOMPOSITION_RESULTS.md)
> extends the same compiler to the declared ordered field
> \(\mathbb Q(\sqrt2)\); general number fields remain open.

## Question

The preceding Schur detector certified an already irreducible rational
representation as real, complex, or quaternionic type. It deliberately
rejected repeated irreducibles because a matrix algebra such as
\(\operatorname{Mat}_2(\mathbb R)\) is not a division algebra. The remaining
Programme 01 obligation was to turn that rejection into a constructive,
auditable decomposition:

\[
V\cong\bigoplus_\lambda W_\lambda^{\oplus m_\lambda},
\qquad
\operatorname{End}_G(V)
\cong
\bigoplus_\lambda
\operatorname{Mat}_{m_\lambda}(\mathbb D_\lambda),
\]

where \(\mathbb D_\lambda\in\{\mathbb R,\mathbb C,\mathbb H\}\).

The acceptance criterion was not a guessed list of dimensions. The compiler
had to return exact projectors, restricted generator matrices, irreducible
Schur certificates, inter-copy intertwiners, central isotypic projectors, and
the complete double-centralizer dimension accounting.

## Exact construction

For rational generator matrices \(A_r\), the implementation first solves the
simultaneous commutant

\[
\mathcal C
=\{X:XA_r=A_rX\text{ for every }r\}.
\]

It searches deterministic low-complexity rational elements of \(\mathcal C\).
First it solves the center \(Z(\mathcal C)\) and splits its rational primitive
sectors, separating inequivalent isotypic components before touching their
multiplicity coordinates. It then searches the full corner algebra inside each
central sector. For a candidate \(X\), exact power dependence gives its
minimal polynomial. When that polynomial has coprime primary factors,
polynomial Chinese-remainder idempotents split the current invariant summand.
Every proposed split is checked directly:

\[
E^2=E,
\qquad EA_r=A_rE.
\]

The process recurses until each leaf has a division commutant certified by the
existing real/complex/quaternionic detector. Exact intertwiner nullspaces then
group equivalent leaves. Their projector sum must be central in the complete
commutant.

For a block with multiplicity \(m\) and Schur algebra \(\mathbb D\), the
certificate verifies

\[
\dim_{\mathbb R} E\mathcal C E
=m^2\dim_{\mathbb R}\mathbb D.
\]

It also computes the center of the corner algebra. Its dimension must be two
for complex type and one for real or quaternionic type. Finally, invertible
intertwiners align every copy with one reference irrep, producing explicit
isotypic tensor coordinates rather than merely an unordered collection of
invariant subspaces.

The compiler-facing API is:

```python
certificate = decompose_reducible_representation(
    generators,
    assume_completely_reducible=True,
)
if certificate.certified:
    basis = aligned_isotypic_basis(certificate)
    block_generators = transform_to_isotypic_coordinates(
        generators, certificate
    )
```

In `block_generators`, every multiplicity copy contains the same aligned
reference action. The remaining legal equivariant transition is therefore the
reported matrix algebra on multiplicity coordinates. Each block certificate
includes complete exact bases for that aligned commutant and its center, not
only their dimensions.

## Exact controls

| Representation | Recovered blocks | Full commutant dimension | Result |
|---|---|---:|---|
| \(2V_3\), real \(\mathfrak{so}(3)\) vector | \(\operatorname{Mat}_2(\mathbb R)\) | 4 | passed |
| two realified \(U(1)\) lines | \(\operatorname{Mat}_2(\mathbb C)\) | 8 | passed |
| two realified quaternionic \(SU(2)\) spinors | \(\operatorname{Mat}_2(\mathbb H)\) | 16 | passed |
| one real, one complex, one quaternionic irrep | \(\mathbb R\oplus\mathbb C\oplus\mathbb H\) | 7 | passed |
| \(\mathrm{Cl}(3,0)\) under Spin(3) conjugation | \(\operatorname{Mat}_2(\mathbb R)\oplus\operatorname{Mat}_2(\mathbb R)\) on \(2V_0\oplus2V_1\) | 8 | passed |
| Spin(9) quotient model | \(\mathbb R\oplus\operatorname{Mat}_2(\mathbb R)\) on \(V_1\oplus2V_5\) | 5 | passed |

The spin-two fixture is constructed as the rational commutator action on
\(\operatorname{Sym}_0(3)\). Its Casimir is independently checked to be
exactly \(6I_5\), so the five-dimensional block is the spin-2 module rather
than a dimension-only label.

Both the \(\mathrm{Cl}(3,0)\) and Spin(9) fixtures are repeated after a dense,
non-orthogonal rational change of basis that mixes their displayed coordinate
blocks. The recovered isotypic signatures, corner dimensions, centers, and
intertwiner gates are unchanged.

## Direct falsifiers and refusal behavior

| Control | Required behavior | Result |
|---|---|---|
| complete reducibility not supplied | do not interpret the commutant semisimply | refused |
| search budget fixed to zero on \(2V_3\) | do not mislabel the four-dimensional commutant as quaternionic | unresolved rank-6 leaf |
| witness with the wrong shape | reject before algebra | rejected |
| irrational witness in the rational API | reject scalar-domain drift | rejected |
| rational witness outside the commutant | reject invalid split evidence | rejected |

The zero-budget control is important: failure to find a rational idempotent is
not treated as irreducibility.

## Claim boundary

Established:

- exact rational commutant and intertwiner construction;
- recursive decomposition from certified rational commutant idempotents;
- exact irreducible real/complex/quaternionic leaf classification;
- central isotypic grouping and explicit aligned copy coordinates;
- complete aligned corner-algebra and center bases, with exact dimensions;
- the law \(m^2\dim_{\mathbb R}\mathbb D\) for every reported block;
- the maintained \(2V_0\oplus2V_1\) rotor and
  \(V_1\oplus2V_5\) Spin(9) fixtures, including rational conjugacy controls.

Not established:

- termination for every rational semisimple representation;
- decomposition requiring real algebraic projectors outside the rational
  scalar field;
- inference that arbitrary supplied matrices are completely reducible;
- robust decomposition from approximate or noisy floating-point generators;
- learned discovery of the generators, group, or representation;
- automatic model-quality, memory-capacity, or throughput improvement.

Thus “general” describes the representation-independent compiler interface
and its exact verification gates. It does not erase the declared rational
splitting and complete-reducibility assumptions.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m reducible_isotypic_decomposition `
  --output artifacts/reducible_isotypic_decomposition_20260811.json
python -m pytest tests/test_reducible_isotypic_decomposition.py -q
```

The published artifact SHA-256 is
`743eb00a72ff810fabe36d13580f562b78911d25620d7d3774a4f87d971cb260`.
