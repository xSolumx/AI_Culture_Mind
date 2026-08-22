# Spin-Delta learned-router curriculum-transfer results

**Decision:** v2 passed every frozen autonomous-router and curriculum-transfer
condition.

## Provenance repair

The first cohort was rejected before quality summarization because independent
CUDA router-training replicas were not bitwise identical. V2 trained one
router once per initialization and cloned that exact in-memory checkpoint into
all three data-order cells and both core schedules. Every artifact passed the
single-execution identifier, state-digest, untouched-core, frozen-router, and
implementation-hash checks.

No v1 retrieval outcome entered the v2 protocol, seeds, or thresholds.

## Frozen nine-cell result

Sixteen-write accuracy was:

| Init | Data | Fixed | Curriculum | Change |
|---:|---:|---:|---:|---:|
| 617 | 641 | 99.95% | 99.90% | -0.05 points |
| 617 | 643 | 98.05% | 100.00% | +1.95 points |
| 617 | 647 | 94.63% | 99.56% | +4.93 points |
| 619 | 641 | 78.96% | 99.71% | +20.75 points |
| 619 | 643 | 99.66% | 99.85% | +0.20 points |
| 619 | 647 | 91.85% | 97.02% | +5.18 points |
| 631 | 641 | 93.12% | 99.22% | +6.10 points |
| 631 | 643 | 100.00% | 99.41% | -0.59 points |
| 631 | 647 | 99.27% | 99.51% | +0.24 points |

Aggregate retrieval was:

| Metric | Fixed depth | Curriculum |
|---|---:|---:|
| Mean accuracy, 8 writes | 94.60% | **99.31%** |
| Mean accuracy, 16 writes | 95.05% | **99.35%** |
| Mean accuracy, 32 writes | 94.78% | **99.41%** |
| Minimum accuracy, 8 writes | 76.90% | **96.00%** |
| Minimum accuracy, 16 writes | 78.96% | **97.02%** |
| Minimum accuracy, 32 writes | 77.25% | **96.88%** |
| Robustness threshold | failed | **passed** |

Mean paired 16-write improvement was `+0.0430230`, or +4.30 percentage
points. The worst paired change was -0.59 points, inside the frozen two-point
allowance. The curriculum's worst cell exceeded the fixed-depth worst cell by
18.07 points, far above the required two points.

## Factorial contraction

The learned-router fixed arm exposed a much larger basin split than the oracle
factorial:

- maximum fixed-data initialization range: 21.00 points;
- maximum fixed-init data-order range: 20.70 points.

The curriculum reduced these maxima to 2.54 and 2.83 points. Every curriculum
range remained below the frozen five-point sensitivity boundary.

## Router validity

The learned router achieved exactly 1.0 for write-event F1, query-event F1,
write-slot accuracy, and query-slot accuracy at 2, 3, 5, 8, 16, and 32 writes.
The same four metrics were exactly 1.0 over every phase-B training batch in
both schedules. Evaluation supplied token IDs only, and no oracle control
tensor entered the model.

This means the curriculum repair transfers through autonomous forward routing,
not merely through the oracle-control API. The control function is learned and
computed causally from tokens, then frozen.

## Information and compute accounting

Both arms used 800 core updates, 102,400 examples, identical cloned states,
and fresh core optimizer state. Fixed depth saw 2,662,400 tokens; the curriculum
saw 2,009,600, or 75.48% as many. Router commissioning added the same 100
updates to every descendant and is excluded from the paired core token counts.

## Consequence

The following chain is now empirically closed on the synthetic grammar:

1. the causal router can learn the complete event/slot grammar;
2. its hard forward controls remain exact at unseen composition depths;
3. a fixed-depth pristine core is still highly trajectory-sensitive;
4. the 2/3/5/8 information homotopy makes that learned-router core robust over
   a fresh 3x3 factorial.

The immediate per-slot reconstruction and optimizer interventions are no
longer justified as default repairs. The next decisive gate is **learning
autonomy**: train router and core jointly from retrieval loss under the same
short-to-long curriculum, with no router labels in either optimization or
forward execution. The current supervised-router result is the ceiling and
direct falsifier for that experiment.

## Boundaries and artifacts

Phase-A router labels remain privileged supervision, while phase-B labels are
audit-only. This is not self-supervised routing, natural-language quality, a
speed result, or a maintained v1.2 promotion.

Canonical v2 artifacts are in
[`artifacts/spin_delta_router_curriculum_transfer_v2/`](artifacts/spin_delta_router_curriculum_transfer_v2/).
The summary SHA-256 is
`9a6388df9994831083f2fbe3ceb44f84d20059ad4a57c20ae8676e67bfa2ea89`;
the summarizer reproduces it byte-for-byte from the nine cell files.
Ten focused curriculum/router tests pass, the complete maintained WSL/cu126
suite passes 117 tests, and Ruff passes on both transfer implementations and
their summarizers/tests.
