# Spin(8) state quotient-lattice audit: results

Date completed: 2026-08-03. This audit is explanatory and does not repair the
frozen 7/9 state-only gate.

## Result

The central structural hypothesis passes, but the preregistered total-parity
detail is falsified.

For seeds 43 and 46, the viable two-state partition is an exact balanced
homomorphic quotient of the viable eight-state action:

- every eight-state cluster maps to one two-state cluster with purity `1.0`;
- the quotient fibres have sizes `[4, 4]`;
- the quotient map intertwines every token transition exactly (`1.0`);
- independently fitted primary and audit quotient maps agree after centroid
  alignment.

Seed 38 behaves as the intended negative reference: its `k=2` transition is
not viable, has only `0.5` intertwining fraction, and does not replicate.

The frozen hypothesis additionally required all four tokens to act by the
nonidentity two-state permutation. That is false in both 43 and 46: one
inverse-generator pair acts trivially and the other nontrivially. The quotient
is therefore not total word-length parity. It is one of Q8's index-two
character quotients, with a four-element cyclic kernel and quotient
`Q8 / C4 ~= C2`.

## Consequence

This establishes why closure alone cannot infer state cardinality: recurrent
geometry can simultaneously expose a full finite action and exact coarser
congruences. It also supplies the principled repair. The latent automaton is
the unique **finest reproducible congruence** if every other viable candidate
is an exact homomorphic quotient of it. The prospective finest-congruence
compiler freezes that rule before seeds 48--57 and never selects `k=8` merely
because Q8 is expected.

Artifact: `spin8_state_quotient_lattice_audit.json`.
