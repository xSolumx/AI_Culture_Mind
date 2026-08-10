# Overhauled SpinorModel

> **Historical reference.** This implementation remains reproducible and its
> tests remain useful, but it is no longer the maintained model frontier. The
> canonical JAX/PyTorch v2.1 implementation, hard mathematical contract, and
> transport-ablation runner live under
> [`../../SSM-Models/pure_rotor_ssm`](../../SSM-Models/pure_rotor_ssm/CONTRACT.md)
> and `../../SSM-Models/run_transport_ablation_v2.py`.

> A later co-moving FLA memory compiler and hierarchical router are documented
> in the theorem repository, but are not implemented here. Their memory-core
> timings are not model-level results for this historical reference.

This is a new implementation. The historical files in the parent
`SpinorModel` directory are intentionally untouched so the original experiment
and old checkpoints remain reproducible.

## Architecture

The model is a causal selective state-space language model over tensor-valued
Cl(3, 0) states. Each layer and multiplicity channel stores one eight-component
multivector. At token step \(t\), the recurrent core applies

\[
h_t=a_t r_t h_{t-1}\widetilde r_t+b_t,
\qquad 0<a_t<1,
\]

where \(\widetilde r_t\) denotes Clifford reversal, \(r_t\) is a unit even
rotor generated from the current input, and \(b_t\) is a bounded
selective-write innovation. The controller uses
grade invariants for scalar gates and an equivariant bivector source for rotor
direction. Rotation, retention/erasure, and writing are distinct controls.

Two transition tuples compose associatively:

\[
(a_2,r_2,b_2)\circ(a_1,r_1,b_1)
=\left(a_2a_1,\ r_2r_1,\ a_2\operatorname{Ad}_{r_2}(b_1)+b_2\right),
\]

where \(\operatorname{Ad}_{r}(h)=rh\widetilde r\).

The reference parallel backend uses an inclusive Hillis--Steele scan with
logarithmic dependency depth. Streaming inference retains exactly
`layers * channels * 8` scalars and processes one new token at a time. The
reference scan has `O(L log L)` work; a production implementation should fuse
the same operator into a work-efficient kernel.

The retention/write parameterization uses

\[
a_t=\exp(-\delta_t\lambda),
\qquad
b_t=(1-a_t)w_tu_t,
\]

where \(w_t\) is a sigmoid write gate and \(\delta_t\) is a positive,
token-selective step size. Large learned
step sizes implement erasure without silently shortening the initialized
half-life schedule. Since rotor transport is isometric and the innovation
coefficient is at most \(1-a_t\), bounded candidates imply the usual
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
