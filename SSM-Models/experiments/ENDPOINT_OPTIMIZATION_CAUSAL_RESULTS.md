# Endpoint optimization causal audit: results

Date completed: 2026-08-03.

## Frozen-arm outcome

| Arm | Endpoint labels | Trigger | Endpoint behavior | Mechanism gate |
|---|---:|---:|---|---:|
| Fixed L16, 2k | 512,000 | none in seeds 0--2 | chance plateau | fail |
| Ordered curriculum | 512,000 | 10/10, steps 600--800 | 100% all formal gates | pass 10/10 |
| Shuffled same-length mixture | 512,000 | none by 1,500, seed 0 | final logged L16 batch 100%; no frozen evaluation checkpoint | fail |
| Fixed L16, 8k | 2,048,000 | none by 7,500, seed 0 | chance through step 8,000 | fail |
| Scrambled clean blocks | 512,000 | none by 1,500, seed 0 | L1/L2/L4 reach 100%; L16 remains near chance | fail |

This realizes the preregistered `CURR=pass, MIX=fail, F16-8k=fail` branch:
the tested monotonic short-to-long blocked curriculum is the only schedule
tested here that finds a faithful representation at this optimizer,
initialization, and budget. This does not yet establish that monotonicity is
necessary in general. The prospective `L8 -> L1 -> L16 -> L2 -> L4` control
also fails, showing that clean block separation alone is insufficient for this
seed and schedule.

## Fixed-L16 persistence details

The 8,000-step run stops at the prospectively fixed boundary. Its minimum
logged task loss is `4.0610` versus chance `log(60)=4.0943`; maximum logged
endpoint accuracy is `5.86%`; step-8,000 loss/accuracy are `4.0672` and
`1.56%`. No representation is detected during the eligible step-250--7,500
window.

The first 2,000 trajectory values reproduce the original fixed seed-0 run
exactly on every shared field (loss, task loss, accuracy, compile state, anchor,
and retraction diagnostics). The only schema difference is that the new
artifact explicitly logs `sequence_length=16`.

Thus quadrupling the endpoint-label budget does not rescue the symmetric
mixed basin in this experiment.

## Shuffled-mixture details

MIX uses the exact same generated batch multiset as CURR and only permutes its
order. It shows partial and eventually strong behavioral fitting—the final
logged L16 training batch is 100% at task loss `0.0130`—but no channel meets
the pre-registered faithful-representation trigger by step 1,500. Because the
mechanism gate fails, no dense or long checkpoint result is inferred from that
single training batch.

The contrast is important: short-word exposure can make the endpoint task
learnable without forcing optimization into a single compilable group action.
Ordering the same evidence from low to high composition depth does in the
tested schedule; whether any clean block order would suffice is not yet known.

## Mechanistic reading

The exact information audit explains why beginning in a high-information
short-word regime is a plausible mechanism. Near identity,
single-position endpoint information decays from 2 bits at L1 to 0.00128 bits
at L16, and empirical action gradients change from nearly collinear across
batches to mostly incoherent. Short-to-long continuation first identifies token
actions in a high-signal regime, then transports that solution through rising
composition depth. This is an information homotopy, not merely “easier
examples first.”
The scrambled-block control is required to distinguish that continuation-path
account from generic benefits of clean stage separation.

## Scope

- CURR includes L1 endpoint observations; it does not avoid all one-step
  transition information.
- MIX was tested for one diagnostic seed, not a reliability cohort. Its failure
  is sufficient for the frozen causal branch but not a population estimate.
- F16-8k was one prospectively stopped persistence control, not an impossibility
  proof over optimizers or compute; different optimizers, learning rates, or a
  much longer fixed-length run remain outside the tested boundary.
- Clean length blocks in a non-monotonic order were not part of these three
  original arms. The completed one-permutation control fails; it is not an
  exhaustive result over all block permutations.
- The separate endpoint query compiler remains exact and was not the failed
  component.

Artifacts: `endpoint_mixture_seed0.json` and
`endpoint_fixed_L16_8k_seed0.json`; see `ENDPOINT_BLOCK_ORDER_RESULTS.md` for
the subsequently frozen fourth control.
