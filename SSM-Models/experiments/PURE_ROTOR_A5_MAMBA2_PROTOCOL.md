# Pure Rotor v2.1 versus Mamba-2: A5 compositional tracking protocol

Protocol recorded **2026-08-16T16:17:11+02:00** (`Africa/Johannesburg`,
UTC+02:00), before the first cohort artifact from
[`benchmark_pure_rotor_a5.py`](../benchmark_pure_rotor_a5.py).

## Question

Does the maintained `pure_rotor_ssm` v2.1 model's learned rotor transport help
on a controlled non-commutative state-tracking task, relative to its own
identity-transport ablation and a small parameter-near Mamba-2 architecture?

This is deliberately a model comparison. It does not ask whether an oracle
Cl(3) action can represent `A5` (that older, write-free mechanism question is
separate), and it does not treat Mamba-2 as a complex diagonal SSM.

## Task and split

The target is every ordered prefix product in the alternating group `A5`. The
runner independently verifies the two-generator presentation

```text
<a, b | a^2 = b^3 = (ab)^5 = e>
```

using the repository's permutation convention, with `a = 13254` and
`b = 24315`. They generate all 60 elements. Inputs are the symbolic alphabet
`(a, b, b_inverse)`. Training forbids exactly the symbolic adjacent pair
`a -> b`; all evaluation strings contain it. The runner records both pair
counts and observed group-state coverage. By default it injects one legal
shortest witness per A5 state into the fixed shared training schedule, and
records a schedule hash plus an exact bounded-language reachability audit. A
missing-bigram result is therefore rejected if either the grammar or its finite
training schedule silently collapses the group.

Training uses length 16. Evaluations use lengths 2, 16, 64, and 128 and report
cross-entropy, all-prefix accuracy, and final-position accuracy. A length
result is functional evidence only: no learned action is declared an exact A5
representation without separate operator/relation diagnostics.

## Candidates and matching

The default small screening configuration contains:

| Candidate | Architecture | Raw parameters |
|---|---|---:|
| Pure Rotor | canonical v2.1, 9 Cl(3) channels and 3 blocks | 24,990 |
| Identity ablation | the same model, `max_rotor_angle=0` | 24,990 |
| Mamba-2 | Transformers `Mamba2ForCausalLM`, 3 layers, hidden width 32 | 25,604 |

The raw Pure Rotor/Mamba-2 gap is 2.4%. Pure Rotor and the identity ablation
share initial function and raw parameter tensors; the report also records the
ablation's disabled rotor-controller parameters, because raw equality is not
effective-capacity equality. Mamba-2 is initialized independently after the
same seed and therefore does not have the same initial function.

This is **parameter-near, not state-matched**. Mamba-2's SSM and convolution
caches are not the same object as the fixed `Cl(3,0)` layer state. The report
states this instead of presenting the table as a single universal fairness
match. On the present Windows checkout Mamba-2 uses the Hugging Face
Transformers implementation; no fused `mamba_ssm` throughput claim is allowed.

## Predeclared interpretation

- A rotor advantage must replicate across the requested seeds and be reported
  separately from the identity ablation and Mamba-2.
- If Pure Rotor does not exceed identity on held-out-pair-containing long
  sequences, the task supplies no evidence that its learned rotations matter.
- If Mamba-2 wins, that is evidence for this configured model/task pair, not a
  counterexample to the diagonal-SSM state-tracking theorem.
- If any candidate fits L16 but fails L64/L128, call it a finite-length fit;
  do not call it compositional generalization.
- Throughput is informative only for the recorded backend. A Transformers
  fallback cannot support a native Mamba-2 kernel comparison.

## Commands

Run the fast structural checks from the repository root:

```powershell
python -m pytest SSM-Models\test_pure_rotor_a5_benchmark.py -q
```

Run a one-seed, 200-step CUDA pilot with checkpoints:

```powershell
python SSM-Models\benchmark_pure_rotor_a5.py --device cuda --steps 200 `
  --batch-size 16 --validation-batches 4 --validation-batch-size 64 `
  --evaluation-microbatch-size 16 `
  --seeds 0 --checkpoint-directory SSM-Models\checkpoints\pure_rotor_a5_mamba2_pilot `
  --output SSM-Models\experiments\artifacts\pure_rotor_a5_mamba2_pilot_seed0.json
```

Do not treat that pilot as a completed cohort. The first interpretable screen
uses 1,000 steps and seeds `0,1,2`, retains all JSON outputs and checkpoint
hashes, and adds a dated results document before upgrading any programme
status. The current Windows Transformers fallback requires explicitly recorded
training/evaluation batch choices; a batch-256 run was interrupted before a
report, as recorded in [`PURE_ROTOR_A5_MAMBA2_EXECUTION_LOG.md`](PURE_ROTOR_A5_MAMBA2_EXECUTION_LOG.md).
