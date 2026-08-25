# Repository map

**Research author:** Hayden Austin

**Last reconciled:** 2026-08-25

This repository has one Git owner and one current scientific navigation
authority. The eight canonical claim charters are under
[research-programs](research-programs/README.md). Dated reports preserve
provenance; they do not override a current charter merely because they are
more detailed.

## Canonical top-level paths

| Path | Owns | Does not own |
|---|---|---|
| [research-programs](research-programs/README.md) | Current claim names, established/open status, dependencies, nonclaims, and cross-programme routing | Raw artifacts or duplicated result narratives |
| [Spin-Space-Research](Spin-Space-Research/README.md) | Exact Spin/Clifford/Dirac mathematics, certificate generators, tests, manuscripts, and hash-bound artifacts | Current model-quality promotion |
| [SSM-Models](SSM-Models/README.md) | Maintained/experimental recurrent models, trainers, checkpoints, model contracts, and model result reports | Global theorem status outside a model contract |
| [Spin8-SSM-Benchmark](Spin8-SSM-Benchmark/README.md) | Isolated controlled model and systems comparisons | Maintained-model identity or cross-programme theorem status |
| [SpinorModel](SpinorModel/README.md) | Historical prototype and its separately labelled overhaul | Current model frontier |
| [PUBLICATION_SCOPE.md](PUBLICATION_SCOPE.md) | Public/private and result-publication rules | Scientific proof of an individual claim |
| [AUTHORSHIP.md](AUTHORSHIP.md) | Repository-wide research-author and attribution policy | Third-party ownership overrides |
| `.private/` | Ignored local recovery/audit material | Public evidence of any kind |

## One source for each kind of truth

| Question | Canonical source |
|---|---|
| Who is the research author? | [AUTHORSHIP.md](AUTHORSHIP.md), [NOTICE](NOTICE), and [CITATION.cff](CITATION.cff) |
| What licence applies? | [Apache License 2.0](LICENSE), qualified by the attribution, prior-release, and third-party boundaries in [NOTICE](NOTICE) |
| What programmes exist and what do they currently claim? | [research-programs/README.md](research-programs/README.md) and the eight canonical charters |
| Which models are maintained, experimental, or historical? | [SSM-Models/MODEL_STATUS.md](SSM-Models/MODEL_STATUS.md) |
| What exact theorem/certificate documents exist? | [Spin-Space documentation guide](Spin-Space-Research/docs/README.md) and [experiment index](Spin-Space-Research/docs/EXPERIMENT_INDEX.md) |
| What is the controlling empirical result? | The model/result document beside its artifact; use the programme ledger to find it |
| What was known at a past date? | The dated audit, preregistration, result, or documentation-refresh file |
| What is public or intentionally local? | [PUBLICATION_SCOPE.md](PUBLICATION_SCOPE.md) |
| How should the work be cited? | [CITATION.md](CITATION.md) |

Shared code may support several programmes. A claim moves across programmes
only through an explicit theorem, reduction, or matched experiment.

## Documentation classes

1. **Maintained front doors:** root README/map/policies, programme charters,
   component READMEs, model status, and documentation indexes. These must track
   the current repository.
2. **Frozen prospective records:** preregistrations and amendments. They remain
   unchanged after their declared freeze except through an explicit correction
   or later amendment.
3. **Completed evidence:** result reports and machine artifacts. The report
   states the protocol, outcome, limitations, and validation path.
4. **Historical snapshots:** dated audits, older plans, superseded model docs,
   and extracted records. Their observations remain provenance; current status
   comes from a maintained front door.
5. **Private/runtime material:** ignored caches, logs, recovery data, and
   incomplete work. It is never public evidence.

Repository-wide authorship is declared centrally so frozen/hash-bound records
do not need byte-changing author insertions. Maintained front doors repeat
`Research author: Hayden Austin` explicitly.

## Active and historical directory names

`Spin-Space-Research` was formerly the nested
`Spin8-Triality-Research` repository. It was flattened into this Git owner on
2026-08-11. The historical remote and package names remain valid provenance
identifiers, but current local links must target `Spin-Space-Research`.

Legacy programme directories are retained where they contain historical
narrative or evidence routing. They are marked as legacy and link back to the
eight active charters; they are not alternate programme authorities.

## Artifact and result routing

- `Spin-Space-Research/ARTIFACTS.sha256` covers published exact-research
  machine artifacts. `PROVENANCE.json` records the original extraction and is
  not rewritten to make old files look newly authored.
- `SSM-Models/experiments/artifacts` and model-local `artifacts` directories
  hold empirical machine records. Their controlling Markdown result must state
  whether the run is complete, validated, historical, failed, or smoke-only.
- Model checkpoints are distributed only through an explicit documented
  exception. Ordinary checkpoints, datasets, logs, and profiler dumps remain
  ignored.
- The active `SSM-Models/hybrid_memory_v1_4` workspace controls its own frozen
  G-series records. External indexes may summarize and link to them but must
  not silently amend them.

## Validation entry points

Repository documentation:

```powershell
python tools/check_repository_docs.py
```

Exact theorem layer:

```powershell
Set-Location Spin-Space-Research
$env:PYTHONPATH = "src"
python tools/verify_artifact_manifest.py
python tools/audit_math_docs.py
```

Model validation is component-specific; use
[SSM-Models/MODEL_STATUS.md](SSM-Models/MODEL_STATUS.md) to reach the relevant
contract, tests, and result report.
