# Resource-bounded FLINT and CUDA audit

**Date:** 2026-08-06  
**Machine:** Intel i7-9700K, 23.89 GiB RAM, RTX 2070 SUPER 8 GiB  
**Status:** exact arithmetic cross-check passed; numerical falsification cohort
passed; no new global theorem claimed

## Why two exact backends

Setting `SYMPY_GROUND_TYPES=flint` makes SymPy faster, but it does not create an
independent check: SymPy still controls the expression and calls FLINT below
the domain layer. The new cross-verifier instead serializes each rational
entry into native `python-flint` objects and asks FLINT to recompute:

- the balanced 28-by-28 information rank and determinant;
- its complete characteristic polynomial;
- the rank-25 finite-deformation boundary;
- the rank-25 and rank-7 weight-simplex boundaries;
- all coefficients of the degree-28 fixed-support determinant, reconstructed
  independently from 29 rational evaluations by solving a FLINT Vandermonde
  system.

Every quantity agrees exactly with SymPy. This is independent arithmetic
verification after the projectors have been constructed; it is not an
independent derivation of the Spin(8) representation matrices.

Harness: `src/spin8_flint_crosscheck.py`  
Artifact: `artifacts/spin8_flint_crosscheck_20260806.json`

## Frozen resource contract

`src/spin8_resource_limits.py` runs each expensive stage as a subprocess with:

- affinity restricted to six logical cores, leaving two i7-9700K cores free;
- all common BLAS/OpenMP pools capped at six threads;
- `SYMPY_GROUND_TYPES=flint` in the child environment;
- a process-tree RSS watchdog at 15 GiB, one GiB below the requested ceiling;
- peak RSS, elapsed time, affinity, command, and exit status recorded.

The watchdog terminates the process tree if the threshold is reached. Exact
work remains staged because monitoring is not a substitute for avoiding large
simultaneous intermediates.

## Ten-seed CUDA counterexample hunt

The numerical cohort uses the GPU only for falsification and conditioning:

| Measurement | Result |
|---|---:|
| Seeds | 10/10 passed |
| Dense interior samples | 860,160 |
| Gradient-refined starts | 1,680 |
| Samples or starts above `log(81/1024)` | 0 |
| Recovered normalized KW maximum | exactly `75.0` numerically |
| Worst learned fixed-support weight error | `7.30e-14` |
| Gaussian noise samples | 286,720 |
| Uphill samples above the exact local optimum | 0 |
| Worst float32 log-determinant discrepancy | `1.51e-5` |
| Peak CUDA allocation | 2,875.2 MiB |
| Peak process-tree RSS | 1.483 GiB |
| CPU affinity | cores 0--5 |

Each seed searches all 21 view allocations with 4,096 random interior frames,
then runs eight gradient starts per allocation for 250 steps. It also maps the
continuous Kiefer--Wolfowitz quadratic sensitivity on all three unit spheres,
learns the exact fixed-support reweighting, and profiles normalized query noise
from zero through standard deviation `0.05` in float64 and float32.

The strongest dense random value remained below the candidate optimum. The
gradient search repeatedly converged to the balanced value but never exceeded
it. This is strong counterexample-search evidence and a hardware-conditioning
profile, not a proof of global five-query optimality.

Harnesses:

- `src/spin8_gpu_design_audit.py`
- `src/spin8_gpu_design_cohort.py`

Artifacts:

- `artifacts/spin8_gpu_design_cohort_20260806.json`
- `artifacts/spin8_gpu_design_cohort_resource_20260806.json`

## Boundary sensitivity learned from the sweep

The float64 determinant decreased under every sampled Gaussian perturbation.
At noise standard deviation `0.05`, the smallest observed eigenvalue fell from
about `0.2373` to `0.1525`, while the worst condition number rose from about
`12.97` to `20.98`. This is controlled degradation rather than immediate rank
collapse. Float32 log-determinants differed from float64 by at most about
`1.51e-5` in this experiment, so theorem-boundary comparisons must continue to
use float64 for search and exact rational arithmetic for adjudication.

## Hierarchical box filtering: safe design

A GPU may prioritize boxes by sampled margin, Lipschitz estimates, or interval
surrogates. It must not certify or discard a box. The proof-safe architecture
is:

1. represent the target polynomial in an exact Bernstein basis;
2. use CUDA only to rank boxes most likely to contain small or negative values;
3. subdivide selected boxes by exact rational de Casteljau transforms;
4. let FLINT/SymPy certify a box only when every exact Bernstein control has
   the required sign;
5. retain boundary-adapted charts and factor known equality strata before
   subdivision;
6. stop before the 15 GiB watchdog rather than allowing box queues and exact
   coefficient tensors to coexist without bounds.

This scheduler is the recommended next performance project. It is not yet
wired into the million-control proof path, because an unvalidated GPU discard
rule would weaken rather than accelerate the mathematics.

## Replay

```powershell
$env:PYTHONPATH = "src"
python -m spin8_resource_limits --workers 6 --memory-gib 15 -- `
  python -m spin8_flint_crosscheck --threads 6 `
  --output artifacts/spin8_flint_crosscheck_replay.json

python -m spin8_resource_limits --workers 6 --memory-gib 15 -- `
  python -m spin8_gpu_design_cohort `
  --output artifacts/spin8_gpu_design_cohort_replay.json `
  --seeds 20260806 20260807 20260808 20260809 20260810 `
          20260811 20260812 20260813 20260814 20260815
```
