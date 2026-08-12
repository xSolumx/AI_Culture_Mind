# Reproduction instructions

## Minimal independent replay

Requirements: Python 3.10 or newer. No third-party package is imported.

From this directory:

```text
python verify.py
```

The verifier reconstructs the exact certificate in memory and compares it with
`artifacts/certificate.json`. It exits nonzero if an identity fails or the
stored coefficient map differs.

To print the reconstructed artifact without trusting the stored file:

```text
python verify.py --emit
```

## Full representation-to-spectrum replay

From the repository root, with the project dependencies installed:

```text
set PYTHONPATH=src
python -m spin8_cayley_blocks
python -m spin8_cayley_flag
python -m spin8_cayley_criteria
python -m spin8_publication_flint_crosscheck
python -m unittest discover -s tests -p "test_spin8_publication_theorems.py"
```

PowerShell users can write the environment assignment as:

```powershell
$env:PYTHONPATH = "src"
```

The full replay constructs the rational triality information matrix and checks
the block and flag certificates. It has more dependencies and a larger trust
surface than the minimal verifier.

## Hash verification

On PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 verify.py, artifacts/certificate.json
```

Compare with [`SHA256SUMS`](SHA256SUMS). The hashes in
[`UPSTREAM_ARTIFACTS.sha256`](UPSTREAM_ARTIFACTS.sha256) identify the four
canonical full-repository artifacts used when this package was assembled.

## Expected resource use

The minimal verifier is a sub-second, single-process exact calculation and
requires negligible memory. It does not use the GPU and does not start any
training or benchmark job.
