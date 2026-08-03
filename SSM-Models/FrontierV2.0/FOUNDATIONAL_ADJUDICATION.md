# Frontier V2 foundational adjudication

This file preserves `question.md` as the speculative source and classifies its
four proposals against the code, mathematics, and current evidence.

## 1. Spontaneous gauge discovery in standard models

**Status: plausible empirical program, not established.**

The decoder-blind compiler could be applied to hidden trajectories from a
standard model, but an apparent finite action is not automatically an emergent
gauge symmetry. Required controls include held-out word identities, gauge
changes, intervention on the candidate subspace, comparison with shuffled
trajectories, and proof that the action predicts unseen compositions rather
than clusters labels already present in the prompt.

## 2. Syntactic-monoid duality theorem

**Status: false as stated.**

Zero loss does not force a continuous recurrence to be a homogeneous space
`G/H`, a group action, or a minimal realization. It may contain redundant,
non-observable, nonlinear, or task-specific state. A correct theorem needs at
least observational equivalence, reachability, minimality, and regularity
assumptions. Even then the discrete quotient does not uniquely determine the
continuous realization above it.

The exact congruence-lattice audit already demonstrates this identifiability
boundary: transition closure alone supports multiple quotients.

## 3. Infinite compact-group compiler

**Status: the advertised robust-memory conclusion is obstructed.**

Compact groups can contain abstract free subgroups, but generic infinite words
have dense images. Distinct words therefore become arbitrarily close. A fixed
finite-precision state plus a positive decoding margin cannot distinguish all
of them for unbounded length. Compact norm preservation avoids overflow; it
does not create unbounded robust information capacity.

An honest infinite-state direction needs growing precision, an external stack,
a noncompact/discrete state, a bounded-horizon claim, or a task that asks only
for a continuous invariant rather than exact word identity.

## 4. Triality as activation-free binding

**Status: corrected constructive path now implemented.**

The bilinear map `S+ x S- -> V` is real and exactly Spin(8)-equivariant. An
unrestricted feedback recurrence using it is nonlinear and is not closed under
the existing affine scan. The exact scan-compatible form is triangular:

1. scan independent `S+` and `S-` affine memories;
2. bind their prefix states pointwise with the triality tensor;
3. use the result as the drive of a downstream vector scan.

`spin8_triality_lift.py` proves the construction two ways: an 81D homogeneous
single-scan lift and a practical two-stage scan with a 24-scalar streaming
cache. It also exposes the obstruction: two-way feedback causes unbounded
polynomial-degree growth.

## The strongest combined frontier

```text
isotypic state factorization
    -> Schur-complete multiplicity transitions
    -> independent retention/write controls
    -> triangular triality binding across staged scans
    -> masked associative-recall falsifier
    -> only then language modeling.
```

This is narrower than the original frontier language, but every arrow is
mathematically closed and experimentally falsifiable.
