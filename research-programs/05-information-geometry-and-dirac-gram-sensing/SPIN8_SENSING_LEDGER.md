# Spin(8) sensing and Cayley design

## Scientific object

One infinitesimal `Spin(8)` action is observed through unit probes in the
vector and two chiral-spinor representations. Each probe contributes a
rank-seven information projector on the 28-dimensional action algebra.

This program concerns identifiability, conditioning, and exact experimental
design. Correlated-frame Dirac--Gram inequalities form a separate program.

## Claim ledger

| Claim | Canonical source | Status |
|---|---|---|
| Generic five-probe boundary | [`SPIN8_CONTINUOUS_PROBE_ORBIT_THEOREM.md`](../../Spin-Space-Research/docs/experiments/SPIN8_CONTINUOUS_PROBE_ORBIT_THEOREM.md) | Exact on the principal mixed-probe stratum; exceptional five-tuples are not fully classified |
| Displayed free five-tuple | [`SPIN8_GLOBAL_FIVE_PROBE_THEOREM.md`](../../Spin-Space-Research/docs/experiments/SPIN8_GLOBAL_FIVE_PROBE_THEOREM.md) | Exact construction; not an all-orbit classification |
| Balanced Cayley spectrum | [`CAYLEY_INFORMATION_SPECTRUM.md`](../../Spin-Space-Research/docs/manuscripts/CAYLEY_INFORMATION_SPECTRUM.md) | Exact canonical-family theorem with a separately stated orbit-normal-form input |
| Strict local five-query optimum | [`SPIN8_FIVE_QUERY_LOCAL_GEOMETRY.md`](../../Spin-Space-Research/docs/experiments/SPIN8_FIVE_QUERY_LOCAL_GEOMETRY.md) | Exact local result modulo the Spin(8) orbit |
| Approximate-design correction | [`SPIN8_APPROXIMATE_DESIGN_CORRECTION.md`](../../Spin-Space-Research/docs/experiments/SPIN8_APPROXIMATE_DESIGN_CORRECTION.md) | Exact distinction between fractional designs and exact equal-five designs |

## Open claims

- global equal-five-query optimality across allocations and nonorthogonal
  frames;
- complete classification of exceptional mixed five-probe strata;
- robustness under noisy or quantized probes beyond the recorded experiments.

## Referee package

The compact independently readable package is
[`referee/cayley-information-spectrum`](../../Spin-Space-Research/referee/cayley-information-spectrum/README.md).
It deliberately distinguishes the exact canonical-family algebra from the
global orbit-normal-form premise.
