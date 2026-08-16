# Spin(3) -> Spin(8) -> Spin(9) -> Spin(10) -> Spin(11) -> Spin(12)

**Status:** passed algebraic implementation gate with an exact-matrix core and
separately labelled float64 algebraic checks.  This is a representation and
subgroup result, not a trained-model result.

**Executed:** 2026-08-16 17:19:37 +02:00 (`Africa/Johannesburg`).

**Artifact:**
[`spin_dirac_a5_ladder_20260816.json`](artifacts/spin_dirac_a5_ladder_20260816.json),
SHA-256 `00109bcff339cc139045c6bfef0274621ced47fbcc4749b1e44d3f4c0f264c68`.

> **Exact follow-up.** The float64 group enumeration and tangent-centralizer
> screen in this report have now been promoted to an exact
> `Q(sqrt(5))` group and relation-Jacobian certificate. See
> [`SPIN_DIRAC_A5_RIGIDITY_RESULTS.md`](SPIN_DIRAC_A5_RIGIDITY_RESULTS.md).
> This original artifact remains an independent numerical cross-check.

## What was built

[`spin_dirac_a5_ladder.py`](../spin_dirac_a5_ladder.py) begins with the
maintained octonionic real Spin(8) gamma system.  It appends chirality to obtain
the real 16-dimensional Spin(9) Clifford system already used by the theorem
line, then alternates graded Pauli doubling and chirality extension:

| Spin group | vector dimension | complex Dirac spinor | Weyl half, if even | embedded Spin(3) copies in Dirac / Weyl |
|---|---:|---:|---:|---:|
| Spin(3) | 3 | 2 | -- | 1 / -- |
| Spin(8) | 8 | 16 | 8 | 8 / 4 |
| Spin(9) | 9 | 16 | -- | 8 / -- |
| Spin(10) | 10 | 32 | 16 | 16 / 8 |
| Spin(11) | 11 | 32 | -- | 16 / -- |
| Spin(12) | 12 | 64 | 32 | 32 / 16 |

All gamma entries in this ladder are Gaussian integers.  The complete
Clifford anticommutators, Hermiticity, skew-Hermiticity of every spin
generator, spin--vector covariance, chirality involutions, and recursive
branching checks have exactly zero recorded residual.

The local Spin(8) convention is also the convention published in the
[`spin-triality-research`](https://github.com/xSolumx/spin-triality-research)
archive. For the classical octonion--Clifford--spinor background, see John
Baez's [*The Octonions*](https://arxiv.org/abs/math/0105155). These are
provenance and mathematical context; the acceptance values above come from the
checked local implementation and artifact.

## A5 versus its spin lift

The vector action uses the exact permutation presentation

\[
\langle a,b\mid a^2=b^3=(ab)^5=e\rangle
\]

and its generated permutation group has exactly 60 elements.  The
icosahedral action in `SO(3)` lifts to unit quaternions

\[
a=(0,0,0,1),\qquad
b=\tfrac12(1,\sqrt{2-\varphi},0,-\varphi),\qquad
\varphi=\tfrac{1+\sqrt5}{2}.
\]

Before projection, the correct relations are

\[
a^2=b^3=(ab)^5=-1.
\]

The float64 enumeration found 120 unit-quaternion elements and 60 projective
classes.  Its largest projected relation residual is
`1.22e-16`.  On every spinor module, the central sign is retained exactly as
`-I`; identifying it with `+I` would collapse the binary icosahedral group
`2.A5` back to its vector quotient `A5`.

The Clifford construction makes restriction to the embedded Spin(3) an
isotypic sum of its two-dimensional complex spinor. The character audit over
all 120 enumerated lifts independently confirms the corresponding `2.A5`
restriction. Its worst residual is `1.43e-14`, at Spin(12), below the frozen
`5e-11` threshold. The two chiral restrictions are checked separately in even
dimensions.

## What is exceptional and what continues

Spin(8) is the only rung where the vector and both chiral spinors are all
eight-dimensional.  That is the triality rung.  Spin(9)--Spin(12) continue the
Clifford and branching structure, not triality:

- Spin(9) has a 9D vector and a 16D spinor;
- Spin(10) has a 10D vector and two 16D Weyl spinors;
- Spin(11) has an 11D vector and a 32D spinor;
- Spin(12) has a 12D vector and two 32D Weyl spinors.

The ladder therefore supplies larger exact representation spaces and a stable
central-sign channel.  It does not, by itself, supply a better recurrence,
memory law, or language model.

## Algebraic-geometry bridge

The concrete next object is the representation space

\[
\operatorname{Hom}(2.A_5,\operatorname{Spin}(n))/\operatorname{Spin}(n),
\]

with the lifted `(2,3,5)` relations imposed as polynomial equations in a
Clifford matrix chart.  The current executable gate computes the infinitesimal
centralizer of the embedded vector action.  For every rung it finds exactly
the untouched `so(n-3)` block:

| n | centralizer dimension | conjugacy-orbit dimension |
|---:|---:|---:|
| 3 | 0 | 3 |
| 8 | 10 | 18 |
| 9 | 15 | 21 |
| 10 | 21 | 24 |
| 11 | 28 | 27 |
| 12 | 36 | 30 |

This table was originally a numerical tangent-rank certificate. The exact
follow-up now constructs the relation Jacobian over `Q(sqrt(5))`, proves that
its kernel equals the infinitesimal conjugacy image, and obtains `H1=0` on
every listed rung. A scheme-wide classification and degree-two
obstruction/derived certificate remain open.

## Nonclaims and ML decision

- No triality beyond Spin(8) is claimed.
- No result here upgrades the open unrestricted Spin(8) Dirac--Gram or global
  D-optimality problem.
- No training, checkpoint, throughput, or sequence accuracy was measured.
- The existing Pure Rotor/A5 benchmark remains the ML falsifier: it shows
  variable short-composition signal and no long-horizon A5 tracking.
- The next ML experiment should use this ladder only if it tests a specific
  hypothesis that the vector quotient cannot test: central-sign-sensitive
  `2.A5` prefix tracking with width- and compute-matched direct, complex,
  quaternion, rotor, and selective-SSM controls.

## Replay

From the repository root:

```powershell
.\.venv\Scripts\python.exe SSM-Models\spin_dirac_a5_ladder.py `
  --output SSM-Models\experiments\artifacts\spin_dirac_a5_ladder_20260816.json
.\.venv\Scripts\python.exe -m unittest discover -s SSM-Models `
  -p "test_spin_dirac_a5_ladder.py" -v
```

The four dedicated tests pass.
