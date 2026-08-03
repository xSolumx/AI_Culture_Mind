# SpinorModel experiments

`spinor_llm.py` and `geometric_layers.py` are the maintained implementation.
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
