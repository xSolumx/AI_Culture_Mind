# Documentation Audit: 2026-08-06

> **2026-08-10 addendum.** The counts and frontier statements below are a
> dated audit snapshot. The repository-wide refresh on 2026-08-10 added the
> Spin(8)/Spin(9) hierarchical-memory result, matched routing artifacts,
> co-moving FLA benchmark documentation, current-status pointers on historical
> records, and explicit memory/model/theorem non-transfer rules. Frozen
> preregistrations and their original observations remain unchanged.

## Scope

This pass covered every Markdown writeup in the maintained
`Spin-Space-Research` repository, the related `SSM-Models` archive, and the
original plus overhauled `SpinorModel` documentation. It had three distinct
layers:

1. an editorial rewrite of the current synthesis, plain-language explanation,
   research map, and inherited frontier note;
2. a mechanical audit of every Markdown file, including historical reports and
   preregistrations;
3. an artifact/provenance reconciliation and a semantic claim audit of the
   current manuscripts.

The historical layer is deliberately described as an audit rather than a literary
rewrite of every historical paragraph. Preserving older preregistrations and
negative results verbatim is part of the scientific record; silently
modernizing their claims would damage provenance.

## Editorial changes

The current synthesis now explains the complete result ladder:

- exact `8+8+8+4` Cayley block factorization;
- exact variable-Cayley one-edge positivity theorem;
- exact local stability of the second residual edge;
- exact reduction of the finite second edge to degree-six and degree-twelve
  polynomial gates;
- the precise boundary between exact proof and numerical counterexample search.

The formerly compressed frontier note was rewritten into four falsifiable
programs. Claims that compact dynamics could store arbitrary infinite-group
state without qualification were removed; the defensible target is computation
of selected invariants or quotients.

## Mathematical-writing contract

The new [Mathematical Writing Standard](MATHEMATICAL_WRITING_STANDARD.md)
requires every central equation to state:

- the status of the claim;
- the domain and quantifiers;
- the meaning of its symbols;
- the exact consequence and remaining limitation.

It also separates reconstruction, identity verification, positivity, artifact
integrity, and floating-point falsification as distinct evidence layers.

## Automated checks

The repository now contains `tools/audit_math_docs.py` and a maintained unit
test. The audit rejects:

- deprecated promotional terminology in filenames or prose;
- the stale pre-rename note path;
- raw double-dollar display delimiters;
- known malformed generated-LaTeX fragments;
- unbalanced `\[` and `\]` display delimiters;
- unbalanced `\(` and `\)` inline delimiters;
- unclosed fenced-code blocks;
- broken relative Markdown links.

At the time of this report, the combined audit passed all 294 maintained
Markdown files in the three scoped research trees, and all 451 relative links
resolved. All 37 unique external references were reachable or independently
resolved after one timeout. All 305 current JSON artifact hashes also matched. The
provenance snapshot contained 463 destinations: 441 remained byte-identical to
extraction, 22 had post-extraction Git history, and none was missing. The full
maintained theorem and regression suite now contains 188 tests; its bounded
rerun is recorded in the validation section below.

The detailed semantic findings and publication boundary are recorded in the
[Full Manuscript and Historical-Record Audit](MANUSCRIPT_AUDIT_2026-08-06.md).

## Validation

The complete 188-test suite passed in 366.3 seconds of test time and 375.8
seconds including resource supervision. CPU affinity was restricted to six
logical processors; peak process-tree resident memory was 4.074 GiB; the
15 GiB watchdog did not fire. The combined 294-document mechanical audit and
the complete 305-entry JSON artifact-manifest verifier also passed independently.

## Interpretation boundary

This audit proves mechanical consistency of the documents it scans. It does
not prove that every historical argument is mathematically correct. Exact
claims remain governed by their theorem harnesses, artifacts, independent
arithmetic replays, and explicitly recorded open gates. Editorial elegance is
being used to expose those boundaries more clearly, never to make them appear
stronger.
