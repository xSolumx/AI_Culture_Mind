# Spin(8) finest-congruence compiler preregistration

Date fixed: 2026-08-03, after the exploratory seeds 38/43/46 cardinality and
quotient-lattice audits. None of seeds 48--57 has been trained or inspected.

## Claim under test

The task cardinality need not be supplied if the learned recurrent state
contains a unique finest independently reproducible regular congruence. The
compiler may recover coarser quotient automata, but it must prove that every
other viable candidate is a deterministic homomorphic quotient of the chosen
action. It may not choose a cardinality merely because it is largest or
because Q8 is expected.

## Inputs and prohibitions

Discovery may use only recurrent states, uniformly sampled token strings, and
one-token state successors. It receives no decoder/logits, target labels,
Cayley table, inverse pairs, token-to-element map, identity label, group-aware
sampler, or state cardinality. The only coarse search bound is `2 <= k <= 12`;
the selected value is an outcome. Observer weights become accessible only
after the full anonymous action and torsor origin are frozen.

## Candidate and lattice gate

Each `k=2..12` is independently fitted on primary and audit corpora with the
already-fixed eight-restart deterministic k-means procedure. A viable action
requires:

- transition winner fraction >=0.99 and winner gap >=0.98;
- all token columns are permutations;
- regular group closure of order `k`;
- torsor-origin winner fraction >=0.99 and gap >=0.98;
- exact transition and origin agreement across corpora.

No Euclidean separation-ratio floor is used. Let `F` be the viable candidate
with greatest cardinality. It is accepted as the finest congruence only if it
is unique at that cardinality and every other viable action is, in both
corpora, a surjective quotient of `F` with per-fine-state mapping purity
>=0.99 and exact token-action intertwining. Otherwise the compiler refuses.

After selection, the existing shared regular-orbit Spin(8) retraction and
minimum-change observer transport are applied without gradient steps. The
same algebraic, numerical, negative-control, dense L15--L256, and long
L4095/L4096/L16383/L16384 gates remain.

## Prospective cohorts

- Seeds 38, 43, and 46 are development evidence only and are permanently
  excluded.
- Seed 48 is the prospective smoke. It must select `k=8`, pass every compiler
  gate, be post-hoc Q8-isomorphic, and reach 100% on every dense and long
  checkpoint.
- Only after seed 48 passes, seeds 49--57 form the untouched reliability
  cohort. The reliability gate is at least 8/9 complete passes. Uniform 9/9 is
  reported separately.

The frozen fixed-cardinality state-only result remains 7/9 regardless of this
experiment.
