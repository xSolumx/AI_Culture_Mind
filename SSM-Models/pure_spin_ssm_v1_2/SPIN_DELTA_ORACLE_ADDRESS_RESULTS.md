# Spin-Delta oracle-address intervention results

**Decision:** oracle capacity passed; causal address/event inference bottleneck
passed.

## Frozen result

Both rows used identically initialized Spin-Delta models, the same parameters,
training batches, two-slot state, independent Spin transports, CUDA recurrence,
optimizer, and budget. The intervention supplied only causal write timing,
binary write/erase slots, unit erase strength, and the final query slot.

| Seed | Variant | W=8 | W=16 | W=32 |
|---:|---|---:|---:|---:|
| 431 | learned addresses | 78.83% | 79.52% | 77.66% |
| 431 | oracle addresses | **100.00%** | **99.98%** | **100.00%** |
| 433 | learned addresses | 81.81% | 81.25% | 82.30% |
| 433 | oracle addresses | **99.32%** | **99.29%** | **99.27%** |
| 439 | learned addresses | 98.12% | 97.27% | 97.78% |
| 439 | oracle addresses | **99.34%** | **99.41%** | **99.46%** |

Three-seed means were:

| Writes | Learned | Oracle | Oracle rescue |
|---:|---:|---:|---:|
| 8 | 86.25% | 99.55% | +13.30 points |
| 16 | 86.01% | 99.56% | +13.55 points |
| 32 | 85.91% | 99.58% | +13.66 points |

## Frozen decisions

Oracle capacity required every oracle seed to reach at least 95% at both 8 and
16 writes. The worst observed oracle accuracy in those rows was 99.29%, so the
capacity gate passed.

The bottleneck decision required a five-point 16-write rescue in at least two
seeds and on the three-seed mean. Improvements were +20.46, +18.04, and +2.15
points; two seeds cleared the individual boundary and the mean rescue was
+13.55 points. The causal address/event inference bottleneck gate passed.

The machine decision is preserved in
`artifacts/spin_delta_oracle/summary.json`.

## Interpretation

This separates the failed learned-address model from its recurrent machinery.
With exact causal slot information, the same Spin-Delta recurrence learns
near-perfect overwrite/retrieval in every seed and extrapolates from 8 training
writes to 32 writes without degradation. Therefore the two-slot state, value
drive, Spin transport, raw-CUDA backward, and readout jointly have sufficient
capacity for this task.

The failed autonomous model is bottlenecked upstream: it does not reliably
infer when to write, which slot to erase/write, and which slot to query. This
experiment intervenes on those quantities together, so it does not distinguish
event detection from address classification. Calling it purely an address
geometry result would be too strong.

The result does not rescue Shakespeare quality and does not establish an
autonomous memory model. Oracle slot labels are privileged causal side
information unavailable in ordinary language. The supported engineering
direction is a new, explicitly falsifiable router: discrete low-entropy
write/query events and slot identities, trained with an auxiliary consistency
objective or an internal self-supervised key protocol. It should leave the
verified recurrence and CUDA compiler unchanged.

## Systems diagnostics

Oracle intervention training took 19.08--20.58 seconds versus 18.16--18.78
seconds learned. Peak allocated memory was 131,561,472 versus 131,335,168
bytes. These fixed-order diagnostics are not a speed gate. The intervention's
purpose was causal localization, not throughput.
