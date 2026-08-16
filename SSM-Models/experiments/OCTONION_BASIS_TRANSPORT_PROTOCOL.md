# Haar-Basis Octonion Operator Replication Protocol

Protocol frozen: **2026-08-16T21:11:30+02:00**

## Question

Does the continuous associator-tracking result survive when neither the token
coordinates nor the recurrent-state coordinates expose the canonical Fano
basis?

For basis seed `s in {0,1,2}`, draw a deterministic Haar `Q_s in SO(8)`. Draw
canonical unit octonions `u_t`, but present

`x_t = Q_s u_t`

and supervise every complete transported prefix operator

`Y_t = Q_s L_(u_t) ... L_(u_1) Q_s^T`.

Training remains length 16, batch 32, 300 AdamW updates at learning rate
`3e-3`, and evaluation uses fixed fresh schedules at lengths 16, 64, and 128.
Each basis gets its own deterministic schedule and all learned candidates for
that basis see byte-identical inputs and targets. Record hashes for the basis,
training schedule, and evaluation schedule.

## Candidates

1. `exact_transported_oracle`: uses the hidden `Q_s`; zero parameters and a
   64-scalar operator state.
2. `fixed_canonical_operator`: treats `x_t` as if it were expressed in the
   canonical Fano basis; zero learned parameters. This isolates coordinate
   alignment.
3. `transported_collapsed_octonion`: uses the hidden basis but collapses the
   prefix to one raw octonion before lifting. This isolates associator loss
   independently of coordinate identification.
4. `learned_basis_operator`: learns one `SO(8)` gauge as the exponential of a
   28-parameter skew matrix, decodes tokens, scans canonical multiplication
   operators, and conjugates the prefix back to the learned state basis. It is
   initialized at the identity—not near the hidden Haar basis.
5. `dense_linear_operator`: learns the unrestricted linear leaf map
   `R^8 -> R^(8x8)` with 512 parameters and scans its matrices. Its weights are
   initialized to canonical octonion left multiplication, so it begins at the
   same fixed-canonical function as candidate 2. This is the decisive
   state-matched non-octonion control.
6. `transformers_mamba2` and `delta_product_reference`: the same unfused
   implementations and sizes as the first continuous pilot.

The learned gauge and dense operator both retain 64 recurrent scalars because
the target is a complete operator. Parameter, state, and kernel matching remain
separate questions.

## Frozen gates

For every basis seed:

- exact-oracle maximum MSE below `1e-12`;
- transported-collapsed L128 MSE above `1e-2`;
- every learned checkpoint rehashes/reloads and every metric is finite;
- learned-basis L128 MSE below `1e-3` and below the fixed-canonical control;
- dense-operator L128 MSE below `1e-3` and below the fixed-canonical control.

There is no preregistered win gate against Mamba-2 or DeltaProduct. A failed
gate is a negative result and must not be tuned away inside this protocol.

## Claim boundary

Passing all three bases would establish coordinate-transported realizability
and length extrapolation for this synthetic operator-identification task. It
would not establish a natural-sequence win, triality-specific utility, a
fused-kernel comparison, or superiority over an equally structured dense
operator scan. If the dense control matches the learned gauge, attribute the
gain primarily to scanning the correct associative operator object.
