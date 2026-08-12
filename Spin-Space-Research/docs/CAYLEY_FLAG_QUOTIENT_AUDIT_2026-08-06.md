# Cayley Flag Quotient Audit

**Date:** 2026-08-06
**Status:** exact local isotropy audit plus an explicitly identified classical
global input

## Question

After one vector probe is fixed, the four spinor probes form an orthonormal
four-frame with a labelled \(2+2\) view split. The spectral manuscript reduces
this design to

\[
(e_0;e_0,e_1;e_2,c e_3+s e_4),
\qquad c^2+s^2=1.
\]

The audit asks whether \(c\), or \(z=c^2\) after pair-orientation is forgotten,
really is the only orbit coordinate. This has two logically separate parts:

1. Does the Cayley value classify the underlying oriented four-plane?
2. At fixed four-plane, can the internal \(2+2\) split carry another
   invariant?

The first is a classical global orbit theorem. The second is the
repository-specific flag calculation.

## What the exact calculation establishes

The maintained \(21\) infinitesimal \(\operatorname{Spin}(7)\) generators were
restricted to the stabilizer of each normal-form four-plane. All ranks below
were computed over \(\mathbb Q\).

| Stratum | Exact samples | Plane stabilizer | Image in \(\mathfrak{so}(4)\) | Split stabilizer |
|---|---:|---:|---:|---:|
| Non-endpoint | \(c=3/5,0,-3/5,5/13,12/13\) | 6 | 6 | 2 |
| Oriented Cayley endpoints | \(c=1,-1\) | 9 | 6 | 5 |

The rank-six image is the whole Lie algebra \(\mathfrak{so}(4)\). Therefore
the identity component of the four-plane stabilizer acts on the plane as the
full \(SO(4)\). Since \(SO(4)\) is transitive on oriented orthogonal
\(2+2\) splittings, no continuous split invariant survives at a fixed
four-plane. The endpoint calculation proves the same transitivity there,
despite the larger plane stabilizer.

The dimensions are consistent in both parameterizations:

- the oriented Stiefel space of four-frames in \(\mathbb R^8\) has dimension
  \(22\);
- quotienting the two same-view basis rotations removes \(2\) dimensions;
- the resulting oriented split-flag space has dimension \(20\);
- a generic flag orbit has dimension \(21-2=19\);
- the local quotient therefore has dimension \(1\).

These equalities rule out the proposed hidden continuous split parameter.
Repeating the exact ranks at five non-endpoint rational points also rules out
an accidental rank result peculiar to the original \(c=3/5\) sample.

## What supplies the global step

Berndt and Tamaru record that the
\(\operatorname{Spin}(7)\)-action on the oriented Grassmannian
\(\widetilde{\operatorname{Gr}}_4(\mathbb R^8)\) has cohomogeneity one, with
the two orientations of the Cayley four-planes as its singular orbits. Their
classification supplies the orbit interval. On the maintained standard normal
representative, the exact Cayley four-form evaluates to \(c\in[-1,1]\); its
\(\operatorname{Spin}(7)\)-invariance then identifies \(c\) with the signed
orbit coordinate. This is the global separation step used by the manuscript;
it is not reconstructed by a finite collection of Lie-rank calculations.

Combining that theorem with the exact full-\(SO(4)\) split action gives

\[
\frac{\{\text{oriented orthogonal }2+2\text{ plane flags}\}}
{\operatorname{Spin}(7)}
\cong[-1,1].
\]

The information operator is invariant under all of \(O(2)\) in either
same-view probe pair. A reflection reverses one plane orientation and sends
\(c\) to \(-c\). The information-equivalence quotient is consequently
parameterized by

\[
z=c^2\in[0,1].
\]

The point \(z=0\) deserves care: it is a regular Lie-isotropy type in the
oriented cover but becomes an exceptional boundary orbit after orientation
reversal identifies \(c\) and \(-c\). This does not add a second continuous
coordinate.

## What the flag calculation does not establish

The exact artifact does **not**, by itself:

- prove the global cohomogeneity-one theorem;
- prove that two arbitrary four-planes with the same Cayley value are
  conjugate;
- classify disconnected components of every isotropy group;
- prove the information-spectrum formulas;
- extend the result to nonorthogonal probes or another allocation;
- prove global five-query optimality.

Those obligations belong respectively to the cited orbit theorem, the exact
information-block certificate, the nonorthogonal Dirac--Gram certificates,
and the still-open allocation problem.

## Correction made during this audit

An earlier displayed quotient used the already-reduced \(20\)-dimensional
space of \(2+2\) plane flags and then divided by
\(O(2)\times O(2)\) a second time. The surrounding dimension calculation was
correct, but the notation double-counted basis gauge. The manuscript now
separates:

1. oriented plane flags modulo \(\operatorname{Spin}(7)\), giving
   \(c\in[-1,1]\); and
2. same-view pair reflections, giving the spectral coordinate
   \(z=c^2\in[0,1]\).

## Falsification verdict

The local evidence survives the stronger audit:

- no hidden continuous \(2+2\)-split invariant appears;
- no rank jump occurs at the tested Cayley-null or rational principal points;
- both oriented Cayley endpoints retain full \(SO(4)\) split action.

What has been falsified is the stronger methodological reading that a
one-point rank calculation proves the global interval quotient. It does not.
The global claim is valid only as a hybrid proof: classical orbit
classification plus exact repository-specific isotropy and information
algebra.

## Reproduction

    $env:PYTHONPATH = "src"
    python -m spin8_cayley_flag --output artifacts/spin8_cayley_flag_replay.json
    python -m unittest discover -s tests -p "test_spin8_publication_theorems.py" -v

The maintained artifact is
[spin8_cayley_flag_20260806.json](../artifacts/spin8_cayley_flag_20260806.json).

## Primary references

1. J. Berndt and H. Tamaru,
   [*Cohomogeneity one actions on noncompact symmetric spaces of rank one*](https://arxiv.org/abs/math/0505490),
   especially the discussion of the \(\operatorname{Spin}(7)\) action on
   \(G_4^+(\mathbb R^8)\) on published page 3436.
2. R. Harvey and H. B. Lawson Jr.,
   [*Calibrated geometries*](https://doi.org/10.1007/BF02392726).
3. M. G. Katz and S. Shnider,
   [*Cayley form, comass, and triality isomorphisms*](https://arxiv.org/abs/0801.0283).
