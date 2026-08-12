# Mathematical Writing Standard

This repository treats exposition as part of the scientific method. A formula
is not complete merely because it is correct: the reader must also be able to
tell what its symbols mean, where it applies, and what conclusion it supports.

## The four-part contract

Every central mathematical claim should provide four things close together.

1. **Status.** Label the claim as an exact theorem, a computer-assisted exact
   certificate, numerical falsification evidence, an empirical result, or an
   open conjecture.
2. **Domain.** State the quantified variables and their admissible set. For
   example, write \(0\leq y\leq1\) rather than leaving the interval implicit.
3. **Meaning.** Define every nonstandard symbol before or immediately after
   the displayed equation.
4. **Consequence.** Explain in one sentence what the equation establishes and
   what it does not establish.

The preferred local pattern is therefore:

> Let \(X\in\mathbb R^{4\times8}\) have unit-norm rows and let
> \(G=XX^{\mathsf T}\) be its Gram matrix. We conjecture that ...

The declaration fixes dimensions, normalization, and notation before the
reader is asked to interpret the claim.

## Claim language

Use the narrowest verb justified by the evidence.

| Evidence | Preferred wording |
|---|---|
| Symbolic identity checked in exact arithmetic | “proves” or “certifies exactly” |
| Exhaustion of a finite, explicitly bounded set | “proves exhaustively on this finite set” |
| Floating-point search with no violation | “found no counterexample” |
| Optimization across finitely many seeds | “passed in \(k/n\) seeds” |
| Plausible structural explanation | “suggests” or “is consistent with” |
| Unfinished global statement | “remains open” |

Do not promote a numerical survival result into a theorem, a constrained-family
theorem into a global theorem, or a decoder-level success into a mechanism
claim. A failed auxiliary certificate rejects that certificate, not the target
inequality, unless an actual counterexample to the target has been produced.

## Voice and intellectual attribution

The repository uses a hybrid academic voice. Authorship should remain visible
where researchers made choices; established statements should read as claims
about mathematics or evidence rather than as autobiographical announcements.

- **Abstracts, theorem statements, result summaries, and conclusions:** give
  agency to the theorem, proof, calculation, experiment, or present work.
  Prefer “the theorem establishes,” “the exact calculation yields,” “the
  experiment found,” or “this work proves.”
- **Introductions and discussions:** first-person plural is appropriate when it
  identifies an intellectual decision: “we chose this normalization,” “we
  compare four controls,” or “we interpret the failure as...”
- **Methods and reproducibility:** use active first person when it makes
  responsibility clearer, especially for exclusions, stopping rules, and
  deviations from a preregistration.
- **Historical reports:** preserve the original voice. If later evidence
  changes the interpretation, add a dated correction rather than silently
  converting the old narrative into present-day prose.

Avoid both extremes. Repeating “we proved” makes theorem statements sound
personal and provisional; pervasive passive voice can hide who chose a method
or changed a protocol. The preferred division is simple: authors own decisions,
while results stand on their stated premises.

## Formula style

- Use display mathematics for load-bearing equations and prose for their
  interpretation.
- Prefer \(\operatorname{Spin}(8)\), \(\mathrm{Cl}(p,q)\),
  \(\mathbb R^n\), \(\det\), \(\operatorname{tr}\), and
  \(X^{\mathsf T}\) consistently.
- Reserve lowercase letters for scalars and vectors, uppercase letters for
  matrices or operators, and call out deliberate exceptions.
- Introduce squared coordinates explicitly, for example
  \(z=c^2\in[0,1]\), before using them.
- State every sign premise retained during squaring. “Equivalent” is permitted
  only when both directions have been proved; otherwise say “implies” or
  “is necessary”.
- Distinguish exact equality \(=\) from numerical approximation \(\approx\),
  and give the tolerance and arithmetic type for computational comparisons.
- Avoid raw double-dollar display blocks and malformed generated-LaTeX
  fragments. Use `\[ ... \]` for display mathematics.

## Proof and computation

A computer-assisted proof must identify which layer is exact:

- exact reconstruction of a candidate polynomial;
- exact identity between that polynomial and the original determinant;
- exact positivity of the reconstructed polynomial;
- artifact replay and integrity checks.

These are different obligations. A verifier should recompute the claim it
accepts rather than trust a stored Boolean flag. Hashes establish artifact
identity, not mathematical truth.

When a large calculation is staged, document the dependency chain. Each stage
should record its inputs, output hash, arithmetic backend, resource limits, and
acceptance predicate. Numerical GPU sweeps belong before exact certification as
counterexample searches; they are not substitutes for the certificate.

## Explanatory layer

Every theorem-level result should include a short plain-language account that
preserves the logical boundary. Analogies are useful only when the precise
statement remains nearby. A good explanation answers:

- What objects are being compared?
- Why was this reduction attempted?
- What obstruction did it remove?
- What is now known exactly?
- What remains unknown?

The reader should never have to infer whether a displayed number is an exact
invariant, a floating-point observation, or a training metric.

## Repository-wide audit

Run the documentation checker before committing mathematical reports:

```powershell
python tools/audit_math_docs.py
```

It checks every Markdown file for deprecated promotional terminology, stale
paths, malformed generated-LaTeX fragments, and unbalanced display-math
delimiters. The checker enforces mechanical hygiene; human review remains
responsible for argument quality, scope, and explanatory clarity.
