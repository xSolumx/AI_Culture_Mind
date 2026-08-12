# Co-moving fused DeltaRule preregistration

- **Date frozen:** 2026-08-10, before CUDA result inspection
- **Programme:** Triality memory and Intertwiner SchurScans
- **Status:** prospective exact-compilation systems experiment

## Algebraic target

For row-oriented state and orthogonal supplied value actions,

\[
S_t=(I-\beta_t k_tk_t^\mathsf{T})S_{t-1}R_t^\mathsf{T}
    +\beta_t k_tv_t^\mathsf{T}.
\]

For invertible actions, with `P_t = R_t ... R_1` and
`Sbar_t = S_t P_t^{-T}`, this becomes

\[
\bar S_t=(I-\beta_t k_tk_t^\mathsf{T})\bar S_{t-1}
          +\beta_t k_t(P_t^{-1}v_t)^\mathsf{T},
\qquad
y_t=P_t(q_t^\mathsf{T}\bar S_t).
\]

For exactly orthogonal actions `P_t^{-1}=P_t^T`, recovering the cheaper
transpose formula.

The transformed recurrence is an ordinary DeltaRule and can call a maintained
fused implementation.  The cumulative action is not free: a streaming row
retains both the `8 x 8` DeltaRule state and `8 x 8` `P_t`, or 128 recurrent
scalars versus 64 for the native transported state.

## Rows

1. direct slots with native supplied transport, 64 state scalars;
2. local native transported DeltaRule, 64 state scalars;
3. local co-moving DeltaRule, 128 state scalars, correctness diagnostic;
4. official FLA `chunk_delta_rule` after co-moving compilation, 128 scalars;
5. official FLA `fused_recurrent_delta_rule` after compilation, 128 scalars.

All co-moving timings include the cumulative-action scan, value-frame change,
DeltaRule operator, physical-frame read, loss, and backward pass.  No row may
receive precomputed action prefixes.  FLA is an external systems tier and is
identified by package/version and source URL.

## Correctness and differentiation

Use noncommuting Spin(8) action words and irregular short lengths in float64.
Native and co-moving predictions must agree to `2e-10`; local chunk/recurrent
states must agree to `1e-9`.

Gradients are compared with respect to keys, values, queries, gates, initial
state where supported, and skew action coordinates exponentiated into
orthogonal matrices.  Ambient derivatives with respect to unconstrained action
matrix entries are outside the theorem and are not a valid parity gate.

## CUDA protocol

Hardware is the local RTX 2070 SUPER under the isolated WSL FLA environment.
Record OS, GPU, CUDA, PyTorch, Triton, `flash-linear-attention`, and `fla-core`
versions.  Use float32, batch `16`, key/value width `8`, lengths `256`, `1024`,
and `4096`, and a scalar loss depending on every output.

Tuning uses three independent processes on seed `101`, excluded from timing.
It chooses action-scan backend, local chunk size, FLA chunk size where exposed,
and any safe compile mode.  Freeze one configuration per row/length before
measurement.

Measurement uses three new independent processes, fixed seed `20260810`, 20
warmups and 100 synchronized iterations when runtime permits.  Record median,
interquartile range, minimum, and peak allocated memory separately for forward
and forward+backward.  Each process runs variants in a deterministic rotated
order.  Aggregate medians are computed only after all three raw artifacts
validate.

## Interpretation boundary

The comparison is not state matched: compiled FLA uses twice the logical
recurrent state.  A systems win establishes a useful exact compiler for
supplied orthogonal transport, not a Spin(8)-specific memory-capacity result.
Failure to beat direct slots at width eight does not falsify the algebra; it
may reflect action-scan and kernel-launch overhead.

## Compatibility addendum frozen before timing

An API capability smoke, before any timing result, found that the installed
official `ChunkDeltaRuleFunction` raises an assertion for float32 and requests
bfloat16; fp16 succeeds on the local Turing GPU.  The measured CUDA dtype is
therefore frozen to fp16.  Forward differences are accumulated and reported in
float32, while the independent local algebra and constrained-action gradient
gates remain float64.  Batch, lengths, seeds, rows, timing counts, and state
accounting are unchanged.  The rejected float32 attempt is a compatibility
observation, not a benchmark sample.

## Numerical-stability addendum frozen before valid tuning

The first 4096-token capability run rejected the naive fp16 transpose path at
`0.11816` relative forward error.  Rounded fp16 action matrices are not exactly
orthogonal, so the `P_t^T=P_t^{-1}` substitution accumulates error.  Those
failed samples are excluded before implementation selection.

The measured compiler is therefore the stronger invertible-action formula
above: accumulate `P_t` in float32, solve `P_t x_t=v_t` in float32, cast only
`x_t` into the required fp16 FLA operator, and apply the physical read frame in
float32.  All of these conversions, the prefix scan, and the solve are timed.
The logical streaming state remains the Delta state plus `P_t`, or 128 scalars;
the solve is computation, not unreported recurrent state.  A pre-timing
4096-token gate must compare this path to a float32 native reference and stay
within ordinary fp16 error before tuning resumes.
