# Spin(8) state-only finite-action compiler: prospective contract

Date fixed: 2026-08-03, after the decoder-labeled table-blind compiler passed
seed 19 and untouched seeds 20--28, and before any state-only clustering result
was computed. Seed 20 may be used only as an implementation fixture. Seed 38
is the prospective smoke. Seeds 39--47 are reserved for an unchanged
reliability cohort if the smoke passes.

## Question

Can the finite endpoint algebra be recovered from recurrent-state geometry
alone, without using decoder logits, decoder predictions, target labels, a
group table, inverse pairs, identity labels, or a group-aware sampler during
discovery?

The compiler is told only that there are eight latent states and four input
tokens. State-cardinality inference is explicitly deferred to the next gate.

## Frozen state-only discovery

For each of two independent calibration corpora:

1. Uniformly sample 32 batches of 512 token strings at each of L15 and L16.
2. Extract the final four-channel, eight-real state and flatten it to 32 real
   coordinates. Do not call `decode` or access the output head.
3. Run deterministic Euclidean k-means with `k=8`, eight k-means++ restarts,
   at most 100 Lloyd iterations, and convergence tolerance `1e-10`. Select the
   minimum-inertia restart. No true labels initialize or score clusters.
4. Require every cluster to contain at least 256 paths and require minimum
   centroid separation divided by pooled within-cluster RMS to be at least 2.
5. From every clustered state, apply each of the four learned token actions,
   assign the successor to its nearest state centroid, and accumulate the 32
   anonymous transition votes.
6. Require at least 256 votes per edge, winner fraction at least 0.99, and
   winner-minus-runner-up fraction at least 0.98. Require every token column to
   be a permutation and the generated action to be regular of order eight.

The independent audit corpus is clustered from scratch with unrelated paths
and k-means seeds. Align its centroids to the discovery centroids by a Hungarian
minimum-distance assignment. After that purely geometric gauge alignment, all
32 inferred transitions must agree exactly. Audit transitions are not averaged
into the discovery result.

## Frozen exact compilation

Order discovery centroids by their recovered regular action. Project the full
8x8 orbit Gram matrix of each channel onto the commutant of that recovered
regular representation, compile the shared exact positive-chiral Spin(8)
actions, and transport the linear observer by the minimum-change section map.

The observer and logits become accessible only after the state clusters,
transitions, replication gate, and recovered group are frozen. They may be used
to preserve the learned readout on the exact section, but may not relabel,
merge, split, reject, or select clusters. There are no post-compilation gradient
steps and no independently normalized token actions.

## Frozen gates

In addition to every state-only discovery gate:

- post hoc abstract group isomorphism to Q8 must pass; the isomorphism search
  sees only the completed anonymous table and is never fed back;
- maximum centroid projection RMS <= 0.03;
- commutant residual <= 1e-10;
- Spin(8) action reconstruction <= 1e-5;
- recovered-table homomorphism RMS <= 1e-5;
- exact section rank equals eight;
- observer transport error <= 1e-5;
- streaming state error <= 1e-5 and logit error <= 1e-4;
- every dense L15--L256 and long L4095/L4096/L16383/L16384 central-pair
  checkpoint reaches at least 99% member and joint accuracy.

Outcomes are classified as clustering/coverage failure, transition
nondeterminism, independent-corpus disagreement, wrong abstract group, exact
compilation failure, behavioral failure, or all-gate pass. Thresholds and
classification may not change after seed 20 is inspected; only implementation
bugs that contradict this written algorithm may be corrected and recorded.

## Recorded implementation correction after seed 20

The first seed-20 fixture recovered two perfectly agreeing Q8-isomorphic
transition systems and compiled an exact action, but behavioral accuracy was
0%. The implementation had passed anonymous cluster index 0 as the group base.
Unlike the preceding decoder-labeled experiment, state-only cluster numbers
have no semantic ordering, so cluster 0 need not be the recurrent identity.
This was a gauge bug, not a failed threshold.

Before seed 38, the frozen recovery is corrected to choose as base state the
cluster centroid nearest the model's raw initial recurrent state. This uses
only state geometry, not the decoder, labels, or hidden table. The nearest and
runner-up initial-state distances and their gap are persisted. Both independent
corpora must choose identity centroids that map to one another under the frozen
Hungarian gauge alignment. The original failed seed-20 artifact remains
preserved; the corrected fixture receives a new filename.

The nearest-initial-state fixture still compiled a globally shifted exact
action and scored 0%. A read-only audit showed why: the endpoint curriculum
never supervises the empty word, and the learned initial state itself decodes
as class 7 in seed 20. Nearest-state anchoring is therefore not an identifiable
identity rule and is withdrawn before seed 38.

The final prospective origin rule uses the calibration words themselves. After
recovering the anonymous transition graph, replay every observed token word
from each of the eight possible base clusters. Choose the unique base whose
predicted endpoint cluster agrees with the state-only cluster assignment.
Require winner fraction >=0.99 and winner-minus-runner-up fraction >=0.98.
This resolves the torsor translation using only token strings, clusters, and
recovered transitions. It does not access the decoder or hidden group. The raw
initial-state nearest cluster and distances remain diagnostic only. Independent
corpora must select origins that align under the geometric cluster matching.

## Recorded multi-restart implementation correction

In the untouched cohort, seeds 44 and 45 encountered an empty cluster in one
of eight k-means++ restarts. The implementation incorrectly aborted the whole
multi-restart search. The written contract says to select the minimum-inertia
restart; therefore an invalid individual restart must be discarded, with
failure only if all eight restarts are invalid. This control-flow bug is fixed
without changing initialization, restart count, iterations, tolerance, or any
acceptance threshold. Seeds 43 and 46, whose best completed clusterings have
separation ratios below 2.0, remain frozen failures.

## Claim boundary

A pass is state-only finite-algebra recovery at fixed cardinality on a trained
Q8 endpoint task. It is not unsupervised state-number discovery, raw-language
structure learning, or evidence that Euclidean k-means is universally suitable
for recurrent manifolds. The next adversarial gates are unknown cardinality,
unequal cluster occupancy/noisy equivalence, and transfer to a group not used
to design the Q8 curriculum.
