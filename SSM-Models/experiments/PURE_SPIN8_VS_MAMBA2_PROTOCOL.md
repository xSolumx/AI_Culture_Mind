# Pure Spin(8) versus Mamba-2 Triality-Transport Protocol

Protocol frozen: **2026-08-16T21:54:00+02:00**

## Task

Each continuous token is a 28-coordinate Spin(8) Lie-algebra increment. Most
tokens are Gaussian with standard deviation `0.15`; independently with
probability `0.10`, a token is replaced by a signed `2*pi` rotation in the
first coordinate plane. Those pulses act as identity in `8v` and minus
identity in `8s+` and `8s-`.

A fixed unit initial probe in each of the three triality representations is
transported by the ordered prefix product. Training receives only the terminal
24-scalar `(8v,8s+,8s-)` target. No intermediate state or local action is
supervised.

Use the frozen terminal-only composition-depth curriculum:

| updates | length |
|---:|---:|
| 1--250 | 2 |
| 251--500 | 4 |
| 501--750 | 8 |
| 751--1,000 | 16 |

Evaluate fresh fixed terminal schedules at lengths 16, 64, and 128 plus paired
length-128 examples differing only by an identity versus `2*pi` first token.

## Candidates

1. `maintained_pure_spin8`: `PureSpin8SSMLayer` v1.0 in transport-only mode,
   all three triality representations, 24 recurrent scalars, exponential
   action chart, no input normalization, and no bilinear readout coupling. Its
   `28 -> 28` controller begins at identity plus `N(0,0.05)` noise; the initial
   probes begin at the teacher probes plus `N(0,0.05)` noise.
2. `transformers_mamba2`: installed unfused one-layer Mamba-2 with hidden size
   32, state size 16, four heads of dimension 16, continuous input projection,
   and 24 outputs.
3. `delta_product_reference`: repository equation-faithful DeltaProduct layer,
   hidden size 32, four heads, four Householder factors, and 24 outputs.

Run seeds 0, 1, and 2. For a seed, all candidates receive byte-identical
training/evaluation tensors. AdamW uses learning rate `3e-3`, weight decay
`1e-4`, batch 32, and gradient clipping at 1.0. Save and hash every checkpoint.

## Frozen gates

For every seed:

- teacher self-replay maximum MSE below `1e-12`;
- maintained Pure Spin(8) L128 MSE below `1e-4`;
- maintained Pure Spin(8) L128 MSE below both learned references;
- maintained Pure Spin(8) paired center classification above `0.99`; and
- every learned metric finite and checkpoint reloadable/rehashable.

Mamba-2 and DeltaProduct have no absolute-accuracy gate. Their execution paths
are unfused architecture references, not throughput-matched production kernels.
No threshold, schedule, initialization distribution, or optimizer may change
after the run.

## Claim boundary

Passing would show that the maintained 24-state faithful Spin(8) transport
model learns and extrapolates this algebra-matched center-sensitive synthetic
task more effectively under the frozen budget. It would not establish generic
language-model superiority, triality-specific natural-task utility, parameter
matching, state matching, compute matching, or a global optimization theorem.
