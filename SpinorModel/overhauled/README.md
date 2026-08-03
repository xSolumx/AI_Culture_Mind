# Overhauled SpinorModel

This is a new implementation. The historical files in the parent
`SpinorModel` directory are intentionally untouched so the original experiment
and old checkpoints remain reproducible.

## Architecture

The model is a causal selective state-space language model over tensor-valued
Cl(3, 0) states. Each layer and multiplicity channel stores one eight-component
multivector. For token step `t`, the recurrent core applies

```text
h_t = a_t * (r_t h_(t-1) reverse(r_t)) + b_t
```

where `0 < a_t < 1`, `r_t` is a unit even rotor generated from the current
input, and `b_t` is a bounded selective-write innovation. The controller uses
grade invariants for scalar gates and an equivariant bivector source for rotor
direction. Rotation, retention/erasure, and writing are distinct controls.

Two transition tuples compose associatively:

```text
a_21 = a_2 a_1
r_21 = r_2 r_1
b_21 = a_2 * Ad(r_2, b_1) + b_2
```

The reference parallel backend uses an inclusive Hillis--Steele scan with
logarithmic dependency depth. Streaming inference retains exactly
`layers * channels * 8` scalars and processes one new token at a time. The
reference scan has `O(L log L)` work; a production implementation should fuse
the same operator into a work-efficient kernel.

The retention/write parameterization uses

```text
a_t = exp(-delta_t lambda)
b_t = (1 - a_t) * write_t * candidate_t
```

with a positive token-selective step size and sigmoid write gate. Large learned
step sizes implement erasure without silently shortening the initialized
half-life schedule. Since rotor transport is isometric and the innovation
coefficient is at most `1-a_t`, bounded candidates imply the usual
convex-recursion BIBO bound.

## Contracts

The tests require:

- exact Cl(3, 0) basis identities and reversion;
- nonzero rotor-controller gradient at identity;
- associative transition composition;
- parallel scan/recurrent equivalence;
- full/chunk/token model equivalence;
- padding as an identity state transition;
- causal outputs;
- constant recurrent-state size;
- prompt-once streaming generation;
- exact checkpoint round trips.

Run them from `SpinorModel`:

```powershell
python -m unittest discover -s overhauled -p "test_*.py" -v
```

Reproduce the local correctness/throughput diagnostic from the repository
root:

```powershell
python -m SpinorModel.overhauled.diagnostics --device cuda `
  --output SpinorModel/overhauled/diagnostics_rtx2070.json
```

Train the standalone word-level demonstration from the repository root:

```powershell
python -m SpinorModel.overhauled.train --epochs 20 --device auto `
  --output SpinorModel/overhauled/checkpoints/demo.pt
```

This is a rigorous research reference, not yet a throughput-optimized language
model. Claims about quality, scaling, or advantage over modern delta-rule and
attention baselines require matched downstream experiments.
