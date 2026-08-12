# Intertwiner SchurScan equivariant-identification results

- **Date executed:** 2026-08-10
- **Protocol:** [`INTERTWINER_SCHURSCAN_EQUIVARIANT_IDENTIFICATION_PREREGISTRATION.md`](INTERTWINER_SCHURSCAN_EQUIVARIANT_IDENTIFICATION_PREREGISTRATION.md)
- **Artifact:** [`intertwiner_schurscan_equivariant_identification_20260810.json`](../../artifacts/intertwiner_schurscan_equivariant_identification_20260810.json)
- **SHA-256:** `ccd0447de5626b8bedda3ba92d0ac1ccde5005b62799fb99cdd9f96612d5ba6c`

## Verdict

All frozen gates passed for all ten seeds in both the Spin(8) triality and
SO(3) cross-product families.

The unrestricted bilinear SchurScan interpolated every endpoint on the seen
coordinate subspace to float64 roundoff, but its length-8 group-orbit relative
squared error remained `0.6043--0.7224` for Spin(8) and `0.4699--0.5089` for
SO(3). Restricting the tensor to the complete one-dimensional intertwiner
family recovered the scalar exactly and extrapolated over every held-out group
orbit and length through 512 at roundoff. Giving the unrestricted tensor four
transformed copies of each training endpoint made its design full rank and
also restored roundoff-level extrapolation.

This establishes the controlled consequence of the equivariant hypothesis
class: it fills directions that the restricted observations do not identify.
It does not establish a triality-specific advantage because the matched
ordinary SO(3) cross-product intertwiner family shows the same effect. That row
is an experimental control inside this benchmark, not a standalone SO(3)
language model.

## Frozen-gate summary

| Diagnostic | Spin(8) triality | SO(3) cross product |
|---|---:|---:|
| passing seeds | 10/10 | 10/10 |
| recurrent state | 24 scalars | 9 scalars |
| restricted design rank | 16 / 64 | 4 / 9 |
| augmented design rank | 64 / 64 | 9 / 9 |
| maximum tensor-equivariance absolute error | `2.27e-13` | `4.80e-14` |
| maximum structured orbit relative squared error | `4.53e-30` | `7.61e-26` |
| restricted generic length-8 error range | `0.6043--0.7224` | `0.4699--0.5089` |
| maximum augmented-generic orbit error | `1.67e-29` | `2.46e-25` |
| minimum additive training error | `0.9038` | `2.3276` |
| maximum scan/recurrent/closed-form absolute error | `3.02e-14` | `1.07e-14` |

Every evaluation cell contains 128 fresh sequences. Evaluation lengths are 8,
32, 128, and 512. The downstream retention is `0.995`; hence length 512 is a
true contractive long-horizon endpoint, not an unweighted sum disguised as a
recurrence.

## What the controls show

### Same endpoints, different hypothesis class

The structured and restricted-generic rows receive the identical 64 length-8
endpoints. The Spin(8) restricted feature matrix has rank 16 inside a
64-dimensional tensor-product feature space; the SO(3) matrix has rank 4
inside a nine-dimensional space. The minimum-norm generic solution is therefore
unconstrained away from the seen coordinate block. Its nearly zero training
error cannot certify the missing group directions.

The structured row fits only one scalar multiplying the maintained tensor. In
both families that scalar is `1.0` in every seed. This is not representation
discovery: the complete intertwiner direction is supplied as an architectural
prior.

### Augmentation can buy the missing information

The augmented generic row receives four fresh shared group transforms per base
endpoint, or 256 labelled endpoints rather than 64. Its design is full rank in
every seed and it recovers the teacher to roundoff. Therefore the generic
family is expressive enough; the unaugmented failure is an identification and
support-extrapolation failure, not a recurrence or optimization defect.

The comparison is intentionally not label matched. It demonstrates an
architecture-versus-augmentation trade, not a universal sample-complexity
lower bound.

### The multiplicative drive is load-bearing

The additive row uses the same recurrent source and downstream state but
replaces the tensor-product feature by a linear function of `[u, v]`. Its
training relative squared error never falls below `0.9038` for Spin(8) or
`2.3276` for SO(3). The benchmark therefore cannot be solved by an additive
drive on the frozen random data.

## Scan contract

The fitted tensor changes only the pointwise drive. For both families, the
maintained ordered work-efficient scan, a sequential recurrence, and the
closed-form weighted endpoint agree below `3.1e-14`. Thus the empirical
identification result composes with the maintained SchurScan implementation;
it does not rely on a separate non-scan execution path.

## Claim boundary

Established:

- a known one-dimensional equivariant bilinear family gives exact orbit
  extrapolation from a proper source subspace in this deterministic teacher
  task;
- an unrestricted tensor can interpolate the seen endpoints yet fail on the
  held-out orbit;
- explicit group augmentation can make the unrestricted design full rank and
  close the gap;
- the effect is common to Spin(8) and SO(3).

Not established:

- a triality-specific advantage over other intertwiners;
- discovery of the group, representations, or intertwiner from raw data;
- superiority over direct slots, delta memories, fast weights, or learned
  generic bilinear recurrences on retrieval;
- robustness to noisy labels, representation misspecification, or naturalistic
  inputs;
- parameter-matched quality or production-throughput superiority.

The next scientific gate remains a matched learned-retrieval task. It should
compare a triality SchurScan, a non-triality equivariant control, an
unrestricted bilinear recurrence, direct slots, and modern delta/fast-weight
memories under equal recurrent state and visible training budgets.

## Replay

```powershell
$env:PYTHONPATH='src'
python -m intertwiner_schurscan_equivariant_identification `
  --output artifacts/intertwiner_schurscan_equivariant_identification_20260810.json
python -m pytest -q `
  tests/test_intertwiner_schurscan_equivariant_identification.py `
  tests/test_intertwiner_schurscan.py `
  tests/test_benchmark_intertwiner_schurscan.py
```
