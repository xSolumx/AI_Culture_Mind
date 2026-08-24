# Hybrid Memory v1.4.5 G10 Retention-Safe Fresh Validation

Status: frozen after the seed-1723 causal development replay passed and before
model seeds 1753, 1759, and 1777 are run.

Date: 2026-08-25

## Question

Does the retention-only intervention that repaired G9's exposed weak seed make
the full external causal learning schedule robust from random initialization
on an unseen cohort?

## Development evidence and fixed intervention

The G9 weak seed 1723 was exposed before this gate. An exact paired replay
changed only minimum/initial global retention from 0.90/0.995 to
0.999/0.9995. It improved L96 from 88.269% to 98.059% and L512 from 87.891%
to 96.289%. That supports a causal hypothesis but is not validation.

G10 fixes the resulting v1.4.5 candidate before fresh seeds:

- the complete v1.4.4 architecture, including tied orthogonal query/key
  projection, identity value/output paths, four heads, key dimension 32, and
  value dimension 16;
- minimum learned global retention 0.999;
- initial learned global retention 0.9995;
- all other configuration fields unchanged from G9.

The 0.999 hard floor has a 692.8-token half-life. It constrains global decay,
not the content-selective DeltaNet overwrite term.

## Frozen objective and schedule

- query retrieval cross entropy coefficient 1.0;
- causal reverse-key reconstruction at observed value events coefficient 0.25;
- no internal address, memory-state, write-gate, or intermediate-logit labels;
- AdamW, weight decay 0.01, gradient clip 1.0;
- one continuous optimizer across all phases.

| Phase | Pairs | Queries | Length | Batch | Updates | LR |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 2 | 2 | 16 | 32 | 1,200 | 0.003 |
| 2 | 4 | 4 | 24 | 32 | 1,200 | 0.003 |
| 3 | 8 | 8 | 48 | 32 | 1,400 | 0.003 |
| 4 | 16 | 16 | 96 | 32 | 1,300 | 0.003 |
| 5 | 16 | 4 | 512 | 16 | 600 | 0.001 |

This is the exact G9 schedule: 1,292,800 retrieval labels and 1,408,000
reverse-binding labels per seed.

## Frozen seeds and evaluation

- unseen model seeds: 1753, 1759, 1777;
- the G9 training and evaluation namespace formulas are unchanged;
- 16 batches of 32 sequences per seed and length;
- L96 has 8,192 decisions and L512 has 2,048 decisions per seed;
- retain model/optimizer states, phase traces, source commit, starting Git
  status, preregistration hash, and checkpoint hashes.

## Gate

Pass only if every seed reaches at least 90% exact query accuracy at both L96
and L512. A mean cannot hide a weak seed. Missing or non-finite evidence fails.

## Decision rule

- Pass: promote retention-safe v1.4.5 as the synthetic external-learning
  successor and proceed to a separately frozen ordinary next-token/natural-
  text screen.
- Fail: retain the intervention as exposed-seed development evidence and do
  not advance to natural text.

## Claim boundary

A pass validates fresh synthetic external causal-label learning for this
schedule. It does not establish ordinary label-free next-token learning,
natural-language quality, upstream superiority, or fused speed.
