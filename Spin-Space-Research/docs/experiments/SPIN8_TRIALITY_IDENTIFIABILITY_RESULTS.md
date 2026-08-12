# Spin(8) triality identifiability results

Date executed: 2026-08-03. Protocol frozen in
[SPIN8_TRIALITY_IDENTIFIABILITY_PREREGISTRATION.md](SPIN8_TRIALITY_IDENTIFIABILITY_PREREGISTRATION.md).
Raw artifact: [spin8_triality_identifiability_seeds0_2.json](../../artifacts/spin8_triality_identifiability_seeds0_2.json).

## Result

The preregistered strong-support outcome passed.

The infinitesimal equivariance matrix has shape 14336 by 512, numerical
nullity exactly one, second-smallest singular value 3.4641016151, and an
absolute cosine of 1.0 between its unique null vector and the fixed
octonionic triality tensor. Thus

[
\dim\operatorname{Hom}_{\mathrm{Spin}(8)}(S^+\otimes S^-,V)=1
]

in the implemented real representations, with the Clifford tensor spanning
that line.

The deliberately incomplete observation design has rank 16 in the
64-dimensional bilinear feature space. All learned families nevertheless fit
those observed cells below the frozen 1e-5 MSE requirement.

| Family | Parameters | Seeds | Final training MSE | Worst held-out mean cosine | Maximum held-out MSE |
|---|---:|---:|---:|---:|---:|
| exact triality | 0 | oracle | n/a | 1.0 | 2.73e-9 |
| invariant scalar times triality | 1 | 3 | 1.60e-7 | 1.0 | 1.57e-7 |
| unconstrained bilinear | 512 | 3 | 8.59e-7--1.16e-6 | -0.0814-- -0.00846 | 0.158--0.166 |
| two-layer MLP | 608 | 3 | 9.89e-8--4.10e-7 | -0.101--0.0458 | 0.236--0.357 |

Evaluation used unseen continuous source caps and either an unseen third
generator or mixed generators through length 512. The invariant model learned
scale 0.9988799691 in every seed; cosine remains exactly one because this
scalar affects amplitude but not direction.

## What the gate establishes

This is a direct separation between interpolation and law identification.
The generic tensor and MLP can memorize the rank-16 observations but cannot
determine the unseen completion rule. Spin(8) equivariance collapses the
512-coordinate tensor family to one identifiable direction, leaving only a
learned scalar.

The earlier masked-completion experiment had a full-rank design and therefore
showed a useful bilinear inductive bias, but not underdetermined symmetry
extrapolation. This experiment closes that specific gap.

## Boundaries

- The shared Spin(8) actions are supplied, not learned.
- The uniqueness statement is a numerical verification of standard
  representation theory, not a new theorem.
- This tests one cross-representation completion law, not multi-item memory,
  attention replacement, or language modeling.
- The exact float32 row has small transport accumulation error; the algebraic
  identity itself was separately checked in float64.
