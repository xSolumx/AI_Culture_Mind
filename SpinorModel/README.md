# SpinorModel experiments

> **Lineage note.** The files directly under this directory are the preserved
> first tensor-based SpinorModel baseline. The additive
> [`overhauled`](overhauled/README.md) implementation is the 2026-08-03
> persistent-state reference that closed the original systems gaps. The
> maintained cross-backend successor is now
> [`../SSM-Models/pure_rotor_ssm`](../SSM-Models/pure_rotor_ssm/CONTRACT.md),
> with its v2.1 transport falsification ladder in `SSM-Models/experiments`.
> Commands below reproduce the historical baseline; they should not be read as
> the current architectural frontier.

> The 2026-08-10 hierarchical Spin(8)/Spin(9) memory result is likewise a
> separate memory-core programme. It has not been integrated into this
> historical model and must not be used to upgrade its claims.

`spinor_llm.py` and `geometric_layers.py` are the maintained historical
baseline.
They represent every GA(3, 0) multivector as a PyTorch tensor with final-axis
basis order `[1, e1, e2, e3, e12, e13, e23, e123]`. This removes the original
mix of raw tensors and version-sensitive Kingdon `MultiVector` objects.

`GALLM_1.py` and `GALLM_2.py` remain historical experiments. `GALLM_1.py` is a
conventional transformer baseline despite its name; `GALLM_2.py` explores
GA(2, 0) operators.

Install the local dependencies and run the smoke tests from the repository
root:

```powershell
python -m pip install -r SpinorModel\requirements.txt
python -m unittest discover -s SpinorModel -p "test.py" -v
```

Run the maintained tiny-corpus demo explicitly:

```powershell
python SpinorModel\spinor_llm.py --epochs 100
```
