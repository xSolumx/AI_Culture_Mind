# Pure Spin(8) endpoint observability protocol

Protocol frozen: **2026-08-17T06:14:38.9940486+02:00**

## Question

How much of the signed `(8v,8s+,8s-)` endpoint must be supervised for the
shared Pure Spin(8) router to identify one local action and transfer it to
hidden representations? Where is recovery merely difficult, and where is it
information-theoretically impossible because the input itself erased the lift?

The empirical arm changes only the endpoint readout mask of the already passed
endpoint-only continuous-observation task. It retains the same unique noisy
12-real inputs, seven active teacher coordinates, fixed initial state, L16
training, excluded adjacent center relation, candidates, optimizer, and
L16/L64/L128 evaluations.

## Exact observability arm

The maintained generators are multiplied by two, making every entry integral.
For each of `8v`, `8s+`, and `8s-`, SymPy computes the exact rational rank of
the infinitesimal action on the first `m=1,...,8` standard-basis probes:

`7, 13, 18, 22, 25, 27, 28, 28`.

Thus one state has a 21-dimensional infinitesimal stabilizer, while seven
independent probes remove it in every individual 8D representation. This is a
local Lie-algebra statement, not global group-element identifiability.

For the tested central element `z=-1`, the maintained actions certify

- `rho_v(z) = +I`;
- `rho_+(z) = -I`; and
- `rho_-(z) = -I`.

Consequently the vector endpoint is center-blind while either half-spin
endpoint sees the sign.

The separate quotient collision uses plane-0 coordinates `0` and `2*pi`.
Their full `8v` action matrices are identical, while their `8s+` targets
negate. With balanced hidden lifts, identical inputs therefore have opposite
unit-norm labels. The squared-loss Bayes predictor is zero, its per-scalar MSE
is exactly `1/8`, and hidden-lift accuracy is at most `1/2`. No architecture can
beat this without additional lift information or an asymmetric prior.

Certificate artifact:
`artifacts/pure_spin8_endpoint_observability_certificate.json`<br>
SHA-256:
`fa29a9d74a927993c17328b7dffb5f96c7f42f308b2e30450d4f714a9ce89a53`

## Empirical readout arm

Five masks are trained:

| Readout | Supervised endpoint scalars | Tested center visible? |
|---|---:|---:|
| `vector_only` | 8 | no |
| `positive_only` | 8 | yes |
| `negative_only` | 8 | yes |
| `spinor_pair` | 16 | yes |
| `full_triality` | 24 | yes |

For every mask, two models receive the identical precomputed schedule and the
same candidate-specific initialization:

- 930-parameter shared Pure Spin(8), with one coordinate action tied across all
  three representations; and
- 957-parameter independent `SO(8)^3`, with the same 24 recurrent scalars but
  separate action coordinates per representation.

Only selected endpoint blocks are sliced on CPU and transferred into the loss.
Hidden endpoint blocks, coordinates, and event labels never enter the model or
accelerator-side loss. A unit test requires exactly zero gradient at every
nonfinal time and every hidden representation block.

## Development evidence used to freeze gates

Seed 0 used 2,000 updates, batch 32, L16, 1,024,000 unique observations, and
64,000 endpoints. The independent family learns each supervised view, while
its hidden action heads remain unconstrained. The shared family transfers one
view into all three:

| Readout | Shared action RMSE `(8v,8s+,8s-)` | Independent action RMSE `(8v,8s+,8s-)` |
|---|---|---|
| vector only | `(0.05131, 0.05193, 0.05193)` | `(0.06984, 0.22114, 0.22051)` |
| positive only | `(0.01178, 0.01178, 0.01178)` | `(0.18170, 0.02650, 0.21959)` |
| negative only | `(0.01240, 0.01240, 0.01240)` | `(0.18149, 0.21708, 0.02431)` |
| spinor pair | `(0.01136, 0.01136, 0.01136)` | `(0.18144, 0.02386, 0.02408)` |
| full triality | `(0.01100, 0.01100, 0.01100)` | `(0.11703, 0.02432, 0.02500)` |

All shared spinor center rows are exact at L128. Under positive-only
supervision, shared early/late hidden-negative L128 MSE is `0.00923/0.01768`,
versus `0.24992/0.25312` independent. Negative-only is chirally symmetric.
Even vector-only supervision selects a consistent shared lift on this
injective coordinate chart: shared spinor L128 MSE is `0.04902--0.07600` and
center classification is exact. This last observation is empirical and does
not contradict the quotient collision, because the current input chart still
distinguishes local Lie coordinates and never presents the identical `0/2*pi`
collision.

The initial evaluator treated a `3.83e-8` float32 vector target discrepancy as
visible in some microbatches. Before protocol freeze, visibility was corrected
to an RMS threshold of `1e-5`; spinor gaps are `0.7071`. Training metrics and
checkpoints were unchanged. The independent validator regenerates every split
and records vector classification as `null`.

Development source SHA-256:
`37e8d28eb9a23d5ea831c2530caa9e00d7320dd91db39313d815c534c840f7d2`<br>
Validated replay SHA-256:
`4f9933c4aa4e748bf1c1a1ddfd257db4b4e6e9f38abbe1e441efc294d5982b95`

## Frozen fresh cohort and gates

Fresh seeds **1, 2, and 3** use 2,000 updates, batch 32, L16, and all five
readouts. Every seed must independently pass; no median rescue is allowed.

1. The exact certificate, split audits, schedule hashes, observation hashes,
   metric finiteness, checkpoint hashes, and strict state-dict reloads pass.
2. All vector evaluation targets are certified center-invisible and expose no
   classification score; both spinor targets are certified visible.
3. For every readout other than vector-only, shared action RMSE is at most
   `0.05` in all three views, every L128 per-view MSE is at most `0.07`, and
   both spinor center classifications are exactly one.
4. For vector-only, shared action RMSE is at most `0.09` in all views, every
   L128 per-view MSE is at most `0.14`, and both hidden spinor center
   classifications are exactly one.
5. For every readout and every view, shared action RMSE and both L128 MSEs are
   strictly lower than the independent family.
6. The independent control is capable in every supervised view: action RMSE is
   at most `0.16`, both L128 MSEs are at most `0.20`, and each supervised
   center-visible view has classification at least `0.95`.
7. All three training schedules, all three observation systems, and all 18
   evaluation schedules are distinct.

No mask, architecture, initialization, schedule, optimizer, step count,
threshold, or seed gate may change after this freeze.

## Commands

For each `SEED` in `1,2,3`:

```powershell
python benchmark_pure_spin8_endpoint_observability.py --seed SEED --steps 2000 --batch-size 32 --training-length 16 --evaluation-pairs 64 --evaluation-lengths 16,64,128 --evaluation-microbatch-size 32 --readouts vector_only,positive_only,negative_only,spinor_pair,full_triality --candidates shared_pure_spin8,independent_so8_triplet --device cuda --output experiments\artifacts\pure_spin8_endpoint_observability_validation_seedSEED.json --checkpoint-directory checkpoints\pure_spin8_endpoint_observability_validation
```

Then adjudicate with
[`validate_pure_spin8_endpoint_observability.py`](../validate_pure_spin8_endpoint_observability.py).

## Claim boundary

Passing would establish replicated cross-representation transfer from partial
signed endpoints for this injective seven-coordinate synthetic teacher family,
plus an exact impossibility boundary when quotient inputs erase the lift. It
would not prove global group identifiability from one representation, recovery
from physical unsigned observations, robustness to chart shift or unknown
initial state, all-28-coordinate trainability, natural-task utility, or generic
SSM superiority.

## Post-freeze outcome

The aggregate **failed** exactly as specified; it was not rescued by medians.
Seeds 1, 2, and 3 passed `37/40`, `39/40`, and `39/40` scientific gates. Every
seed failed the vector-only exact hidden-spinor-row gate, with minimum center
accuracy `0.984375`. Seed 1 additionally failed the independent positive-only
supervised-capability and center gates because that control remained near
chance. All integrity gates and all 30 strict checkpoint reloads passed.

The shared positive-only and negative-only rows nevertheless passed every
intrinsic action, all-view L128, and center gate in every seed. Their all-view
L128 MSE ranges are respectively `0.00974--0.02492` and
`0.00932--0.02520`. This stratified result does not change the failed aggregate
status. See
[`PURE_SPIN8_ENDPOINT_OBSERVABILITY_RESULTS.md`](PURE_SPIN8_ENDPOINT_OBSERVABILITY_RESULTS.md).

Failed aggregate SHA-256:
`baed378d569391e86c46df731cfc72db4f0c0a24d21883bb17a4604db9e5c987`
