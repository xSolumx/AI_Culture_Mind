# Pure Rotor / Mamba-2 A5 execution log

Updated **2026-08-16T17:02:30+02:00** (`Africa/Johannesburg`, UTC+02:00).

## Completed structural smoke

[`artifacts/pure_rotor_a5_mamba2_smoke_seed0.json`](artifacts/pure_rotor_a5_mamba2_smoke_seed0.json)
is a one-step CUDA smoke. It has one 16-example evaluation batch at each of
lengths 2, 16, 64, and 128. It verifies the following runner contracts:

- `13254` and `24315` satisfy the requested `(2,3,5)` presentation and
  generate all 60 A5 elements;
- the training schedule excludes `a -> b`, while every evaluation string
  contains it;
- Pure Rotor and Mamba-2 have 24,990 and 25,604 raw parameters (2.4% gap);
- all candidates forward, backpropagate, and write checkpoints; and
- this machine uses the unfused Transformers Mamba-2 path with no importable
  `mamba_ssm` extension.

This is operational/provenance evidence only. One optimizer update, 16
training strings, and incomplete training-state coverage cannot compare model
quality or compositional generalization.

## Interrupted full-batch attempt

The original 200-step pilot started at **2026-08-16T16:22:13+02:00** with
batch 256 and validation batches of 256. Pure Rotor and its identity ablation
completed and wrote checkpoint files, but Transformers Mamba-2 consumed
7.7/8.0 GiB of GPU memory in the long evaluation path and did not reach its
checkpoint or report. The exact process was stopped at
**2026-08-16T16:32:41+02:00** to protect the active workstation.

There is deliberately no JSON report for that run. The two partial checkpoint
files under `checkpoints/pure_rotor_a5_mamba2_pilot/` are incomplete execution
residue, not comparable artifacts, and must not be summarized as a result.

## Correction before the next pilot

`benchmark_pure_rotor_a5.py` now records an
`evaluation_microbatch_size` and evaluates identical logical validation batches
in fixed microbatches. The new test confirms that splitting a Pure Rotor
evaluation batch preserves NLL and both accuracy metrics. This prevents an
unfused Mamba-2 evaluation batch from determining whether the scientific task
can run; it does not make a systems-throughput comparison valid.

## Corrected 200-step screen

The coverage-injected, microbatched three-seed screen completed after this log
was opened. Its result is recorded separately in
[`PURE_ROTOR_A5_MAMBA2_PILOT200_RESULTS.md`](PURE_ROTOR_A5_MAMBA2_PILOT200_RESULTS.md).
It is a valid pilot artifact, but its non-replicated seed-0 short-pair event is
not a model-quality conclusion.

The successor 1,000-update three-seed screen is recorded in
[`PURE_ROTOR_A5_MAMBA2_BUDGET1000_RESULTS.md`](PURE_ROTOR_A5_MAMBA2_BUDGET1000_RESULTS.md).
It finds a variable short-pair signal but no long-horizon tracking result.
