# Public release boundary

**Research author:** Hayden Austin

This repository publishes work that can be interpreted without access to a
private workstation or an unpublished conversation. Public inclusion requires
one of the following:

1. an exact result with a readable derivation and a replayable certificate;
2. a bounded numerical or empirical result with its protocol, artifact, and
   limitations stated together;
3. a historical record needed to explain a correction, negative result, or
   change of interpretation.

## Included with explicit caveats

Open problems, negative results, and theorem candidates may be public when
their status is unmistakable. They are not promoted by proximity to stronger
results. In particular, exact reconstruction is not exact positivity, local
optimality is not global optimality, and scan parity is not model superiority.

## Excluded from public commits

- credentials, tokens, private contact data, and machine-specific paths;
- model checkpoints or data whose redistribution rights are not established;
- temporary renders, caches, profiler traces, logs, and crash-recovery files;
- private reviewer conversations and unedited model-generated commentary;
- raw exploratory notebooks or proof-search dumps that have no stable replay
  contract;
- unpublished intermediate certificate grids when a smaller hashed proof
  object or deterministic generator is the appropriate public artifact.

Excluded material may remain in a local working tree. Its absence from Git is
intentional and must not be interpreted as evidence for a public claim.

## Integrity contract

Published machine-readable artifacts are listed in `ARTIFACTS.sha256`.
Historical extraction hashes remain in `PROVENANCE.json`. These manifests have
different purposes: the former describes current public artifacts; the latter
preserves the original extraction boundary.

For performance claims, the public artifact must also record frozen
implementation selection, warm-up and sample counts, process isolation,
software versions, hardware identity, peak-memory method, and every material
state-width mismatch. Ideal payload bytes and measured allocator peaks are
different quantities. A fused memory-core timing does not become a model-level
quality or end-to-end sparse-routing claim by proximity.
