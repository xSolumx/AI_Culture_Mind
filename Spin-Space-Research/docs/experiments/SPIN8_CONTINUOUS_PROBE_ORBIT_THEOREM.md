# Sharp continuous five-probe theorem for Spin(8) triality

**Date:** 2026-08-06  
**Status:** exact invariant/rank certificate plus compact principal-orbit theorem  
**Verifier:** `src/spin8_continuous_probe_orbits.py`  
**Artifact:** `artifacts/spin8_continuous_probe_orbits_20260806.json`  
**Preregistration:** none; the invariant proof was discovered after the exact
coordinate atlas

## Main theorem

Let Spin(8) act diagonally on ordered unit probes chosen from its vector and two
real chiral-spinor representations.

1. **Every four-probe sensor is insufficient.** Its stabilizer has dimension at
   least six if all probes use one representation, and at least three if the
   sensor uses multiple representations.
2. **The bounds are sharp.** The principal stabilizer Lie algebra is
   `spin(4)` in the single-view family and `su(2)` in every mixed family.
3. **Five mixed probes are generically sufficient.** Every mixed allocation has
   an open dense free stratum: a generic tuple has trivial global stabilizer and
   uniquely determines the shared Spin(8) action.
4. **Five probes in one representation remain insufficient.** Their generic
   stabilizer is `Spin(3)`, of dimension three.

Thus five is the sharp generic query count, and using at least two triality
views is necessary.

## Why four probes cannot work

The configuration space of four ordered unit probes has dimension

\[
4(8-1)=28,
\]

the same as Spin(8). Dimension counting alone therefore leaves open the
possibility of a discrete stabilizer. The missing ingredient is that every
allocation carries unavoidable continuous invariants.

Up to the outer triality permutation, the four allocation types are:

| allocation | independent invariants | count |
|---|---|---:|
| `(4,0,0)` | six pairwise Gram entries | 6 |
| `(3,1,0)` | three Gram entries in the repeated view | 3 |
| `(2,2,0)` | two Gram entries and one quartic triality contraction | 3 |
| `(2,1,1)` | one Gram entry and two triality scalars | 3 |

For `(2,2,0)`, the extra invariant is

\[
q=\left\langle\mu_-(v_1,p_1),\mu_-(v_2,p_2)\right\rangle,
\]

where `mu-: V x S+ -> S-` is triality contraction. For `(2,1,1)`, the two
extra invariants are

\[
\tau(v_1,p,n),\qquad\tau(v_2,p,n).
\]

The maintained certificate differentiates these functions on the product of
spheres at exact coordinate tuples. Their tangent-Jacobian ranks are exactly

\[
6,3,3,3.
\]

The exact Spin(8) action ranks at the same tuples are

\[
22,25,25,25,
\]

so every invariant bound is attained.

## The principal-orbit step

For a smooth compact-group action, one principal isotropy type occurs on an
open dense set, and every other isotropy group contains a conjugate of that
principal isotropy group. A modern source discussing the theorem and its
compact-group role is Wallach's
[principal orbit type paper](https://arxiv.org/abs/1811.07195).

The exact Jacobian ranks prove that the displayed invariants are functionally
independent, so principal orbits have codimension at least the invariant count.
The exact action ranks attain that bound, fixing the principal stabilizer
dimensions at six and three. Every special tuple has at least as large a
stabilizer.

This is the step that upgrades “rank 25 in our experiments” to:

> No continuous or coordinate arrangement of four unit probes can identify a
> shared Spin(8) triality action.

It rules out not only optimizer failure but also any clever alternative
four-probe placement.

## Exact stabilizer types

For every representative, the verifier reconstructs the annihilator Lie
algebra over the rationals, computes all brackets, and checks its derived
algebra, centre, and Killing form.

- `(4,0,0)`: dimension 6, centre 0, derived rank 6, negative-definite Killing
  form; compact type `A1 + A1`, hence `spin(4)`.
- Every mixed allocation: dimension 3, centre 0, derived rank 3,
  negative-definite Killing form; compact type `A1`, hence `su(2)`.

The result is invariant under permutation of the three triality
representations.

## Why five mixed probes generically work

The binary coordinate atlas provides a full triality-closure tuple for every
mixed allocation type:

```text
(4,1,0)  (3,2,0)  (3,1,1)  (2,2,1)
```

Each exact tuple generates all eight coordinate basis vectors in all three
representations. Any shared action fixing it fixes three complete bases and is
therefore globally the identity. These are global certificates, not only
rank-28 tangent checks.

The principal isotropy subgroup must be contained in every isotropy group,
including one of these trivial groups. Hence it is itself trivial. The free
stratum is open and dense in every mixed five-probe allocation.

By contrast, five coordinate vectors in one view have exact action rank 25 and
stabilizer `Spin(3)`. Multiple views are not just better conditioned; they are
structurally necessary at the five-query boundary.

## Plain-language version

Four probes appear to provide 28 numbers of directional information, exactly
matching the 28 unknown rotation controls. But some combinations of the probes
cannot change under any Spin(8) action. These hidden “relationship meters” use
up at least three degrees of freedom. As a result, a three-dimensional family
of different actions always gives the same four answers.

A fifth probe can break the remaining symmetry—but only if the probes look at
more than one of triality's three views. Almost every such mixed five-probe
arrangement uniquely identifies the action. Looking through only one view
still leaves ordinary rotations in the unseen three-dimensional complement.

## Scientific significance

This closes the largest mathematical gap in the active-sensing line:

- the earlier information matrix proved a local rank boundary for sampled
  frames;
- the explicit closure theorem ruled out remote ambiguity for one tuple;
- the binary atlas classified all coordinate tuples;
- the present invariant/principal-orbit argument proves universal four-probe
  insufficiency and generic global sufficiency for every mixed five-probe
  allocation.

The five-probe phenomenon is therefore a theorem about the Spin(8) action, not
an empirical peculiarity of the optimizer or chosen probe frame.

Classical binary spinor machinery remains relevant prior art; see Arizmendi and
Herrera's [binary encoding](https://arxiv.org/abs/1905.10613). The candidate
new contribution here is the sharp multiview sensor-identifiability theorem,
its invariant proof, and its exact computational certificate.

## Claim boundary

Proved:

- universal insufficiency of four ordered unit probes;
- exact principal stabilizer Lie algebras for all four allocation types;
- existence of a globally free point in every mixed five-probe allocation;
- an open dense globally free stratum in every mixed five-probe allocation;
- insufficiency of the single-view five-probe family.

Not proved:

- a complete list of every exceptional five-probe orbit and finite stabilizer;
- an optimal-conditioning theorem for the free five-probe stratum;
- robustness under noisy probes or approximate shared actions;
- a language-model advantage.

## Replay

```powershell
$env:PYTHONPATH='src'
python -m spin8_continuous_probe_orbits `
  --output artifacts/spin8_continuous_probe_orbits_20260806.json
python -m unittest discover -s tests `
  -p "test_spin8_continuous_probe_orbits.py" -v
```

The verifier rebuilds exact skew/triality invariance, differentiates all
allocation-specific invariants, recomputes action ranks and stabilizer Lie
types, and regenerates every mixed five-probe full closure. It does not trust
the artifact's stored `passed` field.
