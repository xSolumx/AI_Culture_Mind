# Repository map

This workspace has one Git owner and one scientific navigation authority.
There are no submodules, nested Git repositories, or pointer-only programme
directories in the maintained tree.

## Canonical top-level paths

| Path | Owns |
|---|---|
| [`research-programs/`](research-programs/README.md) | The seven claim charters, detailed evidence ledgers, status boundaries, and cross-program routing |
| [`Spin-Space-Research/`](Spin-Space-Research/) | Exact Spin/Clifford/Dirac mathematics, certificate generators, tests, published artifacts, manuscripts, and experiment reports |
| [`SSM-Models/`](SSM-Models/) | The maintained rotor and noncommutative state-space implementations |
| [`Spin8-SSM-Benchmark/`](Spin8-SSM-Benchmark/) | Controlled empirical model and systems benchmarks |
| [`SpinorModel/`](SpinorModel/) | Historical prototype lineage and its separately labelled overhaul |
| [`.private/`](.private/) | Ignored local audit streams and recoverable metadata backups; never publication evidence |

## One source for each kind of truth

- Claim status and programme ownership: `research-programs/` only.
- Exact code and executable acceptance gates: `Spin-Space-Research/src/` and
  `Spin-Space-Research/tests/`.
- Published machine evidence: `Spin-Space-Research/artifacts/`, with one entry
  per JSON file in `Spin-Space-Research/ARTIFACTS.sha256`.
- Machine-dependent watchdog records and resumable checkpoints:
  `Spin-Space-Research/runtime/`; ignored and never treated as theorem
  artifacts.
- Maintained rotor-model contract: `SSM-Models/`; `SpinorModel/` is historical.

Shared machinery may cross programme boundaries. Claims do not transfer
without an explicit theorem, reduction, or matched experiment.

## Historical provenance

`Spin-Space-Research/` was formerly the separate
`Spin8-Triality-Research` repository at
`https://github.com/xSolumx/spin8-triality-research.git`. At flattening, its
HEAD matched the root's pinned Gitlink commit
`c4b6310a3a9063f06042d387fb0d90973a10e6d1`. The complete nested `.git`
directory was moved to the ignored, recoverable local backup
`.private/git-backups/Spin-Space-Research.git/`; `CITATION.cff` retains the
historical remote URL.

The backed-up commit contains 961 tracked files. Of these, 955 remain at the
same relative path under `Spin-Space-Research/`. The six deliberate exceptions
are the old `programs/` navigation index and its five ledgers: their substantive
content now lives directly under the canonical root `research-programs/`
charters, and the obsolete nested programme index has been removed.

## Validation entry points

From `Spin-Space-Research/`:

```powershell
$env:PYTHONPATH = "src"
python -m spin8_gate_contracts
python tools/verify_artifact_manifest.py
python -m pytest -q
```

Expensive exact campaigns have narrower documented commands and resource
contracts in
[`Spin-Space-Research/docs/REPRODUCIBILITY.md`](Spin-Space-Research/docs/REPRODUCIBILITY.md).
