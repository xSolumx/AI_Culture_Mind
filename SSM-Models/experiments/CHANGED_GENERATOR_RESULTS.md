# Changed-Generator Transfer Results

The procedure and gate were fixed in `CHANGED_GENERATOR_PREREGISTRATION.md`
before evaluation. The selected macro alphabet is the first qualifying pair in
repository order and its inverses:

- `12453`, `12534` (order 3);
- `23514`, `41253` (order 5).

Their compiled word lengths over the original token operators are 2, 2, 3, and
3. The pair generates all 60 A5 elements. Checkpoint hashes were recorded, no
parameters were changed, and execution was deterministic.

| family | seed | original role | full dense minimum | strongest channel minimum | best channel subset | subset minimum |
|---|---:|---|---:|---:|---:|---:|
| GA rotor | 1 | uniformly clean | 100.00% | 97.75% | 0+1 or all | 100.00% |
| GA rotor | 2 | original dense-pass trough | 92.77% | 91.99% | 0+1+2 | 99.41% |
| GA rotor | 6 | original dense failure | 76.76% | 90.33% | 2+3 | 97.17% |
| GA rotor | 9 | original dense failure | 94.92% | 94.73% | 0+2+3 | 99.80% |
| Householder | 0 | uniformly clean | 99.80% | 25.49% | not audited | not audited |
| Householder | 3 | holonomy-rescued drift | 70.80% | 1.56% | not audited | not audited |

The single-channel Householder diagnostic is not expected to be sufficient:
its learned representation is distributed across channels, unlike the GA
specialization found by the observability audit.

## Complete GA channel-subset audit

The initial four-seed result below is superseded by the deterministic ten-seed,
two-alphabet audit in `CHANNEL_ENSEMBLE_RESULTS.md`. It is retained to preserve
the sequence of interpretation changes.

The initial singleton ablation was insufficient. Each frozen four-channel GA
checkpoint was therefore evaluated on all 15 non-empty channel subsets. This
resolves a genuine contradiction in the first interpretation:

- Seed 1 is not `one signal + three nuisances`: the full model and channels
  `0+1` score 100.00%, while the oracle-like channel 0 alone scores 97.75%.
- Seed 2 contains a harmful channel 3 but a useful auxiliary channel 2. The
  best subset `0+1+2` raises the floor from 92.77% to 99.41%.
- Seed 6 contains a harmful channel 0. The causal channel 2 alone passes at
  90.33%, but the pair `2+3` is much stronger at 97.17%.
- Seed 9 benefits from channel 0 and is damaged by channel 1 in the full code.
  The subset `0+2+3` raises the floor from 94.92% to 99.80%.

Auxiliary-channel effects are therefore signed at the decoder boundary and dependent on
the path distribution. A channel that is at chance in isolation can still
shift the shared decoder boundary helpfully or harmfully when combined with
the causal irrep. Calling all weak channels `nuisance channels` was too strong.
The ten-seed follow-up earns the more precise description `stable decoder
ensemble around one dominant, approximately faithful 3D realization`.

The dominant GA channel remains mechanistically special: it is the only
single channel that tracks A5 well, and removing it collapses performance. But
single-channel sufficiency is not the same as universal dispensability of the
remaining channels.

## Family-level interpretation

Clean Householder also transfers, proving that compiled macro-actions do not
artificially privilege GA. But the Householder seed-3 model that holonomy
rescued on the original generators fails badly after the generating set
changes. Its original-language behavioral repair therefore did not produce a
comparably transferable representation.

This targeted experiment provides structural evidence favoring the compact GA
chart over two-Householder actions: the GA dominant subrepresentation survives
a changed generating set in all four audited seeds, whereas one of two
Householder models does not. The samples are selected and small, so this is not
a population-level GA advantage claim.

The result does not show inference of an operator from a novel symbol. Every
macro operator is compiled compositionally from frozen learned operators; that
is the stated zero-shot group-action test.

## Consequence

Universal post-training pruning is rejected by this audit: it would reduce
changed-generator robustness in seed 1 and discard useful correction channels
in seeds 2, 6, and 9. The next intervention must learn or certify a robust
observable projection across multiple path distributions. A four-channel
subset lattice is cheap enough to serve as an oracle target for a
reliability-aware gate, decoder-margin ensemble, or worst-distribution decoder
objective. Any learned gate must be compared against both the full code and the
dominant channel on original and changed generators; it does not get to select
subsets using test labels.

All ten GA seeds now have deterministic saved checkpoints. The two-alphabet
audit rejects universal pruning and geometric defect-cancellation while
showing selective decoder-boundary correction. Exact-representation rounding
remains necessary for an infinite-horizon guarantee. Spin(8) remains deferred
until the small model's redundancy and stability are understood.

The prospective bounded-gate follow-up now passes an untouched third macro
class `10/10`; see `ROBUST_CHANNEL_GATING_RESULTS.md`.

Raw reports: `changed_generator_channel_subset_lattice_a5_pair0_10seeds.json`
and `changed_generator_channel_subset_lattice_a5_pair1_10seeds.json`.
