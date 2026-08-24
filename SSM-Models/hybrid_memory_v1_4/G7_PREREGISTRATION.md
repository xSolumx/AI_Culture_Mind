# Hybrid Memory v1.4.4 G7 Competence-Paced Validation

Status: frozen after G6 failure analysis and before model seeds 1693, 1697, and
1699 are run.

Date: 2026-08-25

## Trigger

G6 used the causal reverse-binding objective but advanced every seed after a
fixed number of updates. It failed the all-seed gate: seed 1643 reached 89.209%
L96 and 83.398% L512, while seeds 1657 and 1663 exceeded 94% at both lengths.

All seeds learned reverse binding, but the weak seed's last-batch retrieval
accuracies at the ends of the four phases were only 12.5%, 17.2%, 53.9%, and
87.1%. The fixed clock advanced before that seed had mastered the current
problem. G7 changes the training schedule, not the architecture or objective.

## Frozen model and objective

- Hybrid Memory v1.4.4 tied-address configuration;
- query retrieval cross entropy coefficient 1.0;
- reverse-key reconstruction at observed value events coefficient 0.25;
- no internal address, write-gate, state, or intermediate-logit labels;
- AdamW 0.003, weight decay 0.01, gradient clip 1.0, batch size 32;
- one optimizer state is retained continuously across phases and saved.

## Frozen competence pacing

Every 100 updates after the phase minimum, evaluate four fresh deterministic
batches. A phase is mastered after two consecutive probes reach at least 90%
exact query accuracy. Otherwise it trains to its fixed cap.

| Phase | Pairs | Queries | Length | Minimum | Maximum |
|---|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 16 | 300 | 1,200 |
| 2 | 4 | 4 | 24 | 300 | 1,200 |
| 3 | 8 | 8 | 48 | 400 | 1,600 |
| 4 | 16 | 16 | 96 | 1,200 | 2,400 |

Probe batches are never used for gradient updates. Updates used, probe scores,
retrieval labels, and reverse-binding labels are recorded per seed.

## Frozen seeds and final evaluation

- model seeds: 1693, 1697, 1699;
- training data seed: model seed plus 100,000;
- competence probe namespace: model seed plus 500,000;
- L96 evaluation seed: model seed plus 2,000,000;
- L512 evaluation seed: model seed plus 2,100,000;
- final evaluation: 16 batches of 32 per length;
- L96 has 8,192 decisions; L512 has 2,048.

## Gate

Pass only if every phase for every seed reaches competence before its cap and
every seed reaches at least 90% exact query accuracy at both L96 and L512. A
non-finite result, missing artifact, or weak seed fails the gate.

## Claim boundary

A pass validates robust synthetic associative learning from external causal
labels under a competence-paced, capped budget. It does not validate ordinary
label-free next-token learning, natural-language quality, upstream
superiority, or fused speed. Variable update counts must be reported and may
not be compared as if they were fixed-budget efficiency.

