# Continuous Octonion Associator-Tracking Protocol

Protocol frozen: **2026-08-16T21:02:38+02:00**

## Task

Each input token is an independently sampled unit octonion `u_t`. The target at
every prefix is the complete ordered action

`P_t = L_(u_t) ... L_(u_1)`

flattened to 64 real outputs. Supervising the complete operator, rather than
one acted-on vector, prevents an eight-scalar recurrence from hiding loss of
the associator.

Train at length 16. Evaluate fixed fresh schedules at lengths 16, 64, and 128.
Use batch 32, 300 optimizer updates, AdamW, learning rate `3e-3`, and seed 0.
All learned candidates receive identical float32 token/target batches. Hash the
training and evaluation schedules.

## Candidates

1. `exact_operator_oracle`: no learned parameters; the implemented operator
   lift on the true input.
2. `collapsed_octonion_ablation`: recurrently collapse the prefix to one raw
   octonion and emit `L_(prefix)`. This is the intentionally invalid identity
   `L_u L_v=L_(uv)` and is the direct associator falsifier.
3. `learned_octonion_operator`: a 72-parameter continuous `8 -> 8` encoder,
   unit normalization, and exact operator scan. Initialize the encoder as
   identity plus frozen seed-0 Gaussian noise with standard deviation 0.05.
4. `transformers_mamba2`: the installed unfused Transformers Mamba-2 path,
   continuous input projection, hidden size 32, one layer, state size 16, four
   heads of dimension 16, and a 64-output head.
5. `delta_product_reference`: the repository's equation-faithful unfused
   DeltaProduct layer at hidden size 32, four heads, four Householder updates,
   with continuous input/output projections.

Report parameters, recurrent-state scalars, training loss, wall time, peak CUDA
memory, and evaluation MSE/relative Frobenius error. Save and hash every learned
checkpoint. Do not call the unfused learned timings production comparisons.

## Gates and claim boundary

Required structural gates:

- oracle maximum MSE below `1e-12`;
- collapsed L128 MSE above `1e-2`;
- every learned run completes with finite loss/metrics and reloadable hashed
  checkpoint; and
- learned operator L128 MSE below the collapsed ablation.

Mamba-2 and DeltaProduct have no preregistered win gate; their values measure
what this small frozen budget discovers. One seed cannot establish model
superiority or sample efficiency. Success for the learned operator establishes
realizability and length extrapolation on this algebra-matched synthetic task,
not a natural-sequence breakthrough or a triality-specific advantage.
