# Research review: selective rotor SSMs after the A5 falsifier

Date: 2026-08-02

## Bottom line

The project now has a large, repeatable, theory-aligned result: on a two-
generator A5 word problem, the selective Cl(3,0) rotor reaches essentially
perfect length-16 tracking in all three seeds and averages 46.38% at length 32,
while the matched two-layer complex recurrence averages 54.16% at length 16
and only 2.39% at length 32 (chance is 1.67%).

It also has a decisive falsification: when one of the four possible ordered
generator pairs is removed from all training strings and required at
evaluation, both models fit the restricted training language but fail the
unseen pair. The rotor has learned a much better long-horizon representation
under complete local-transition coverage, but not an exact group homomorphism.

That is real progress. It is not yet a general sequence-model breakthrough.

## Corrections to the Gemini interpretation

1. Grade-specific decay does not create cross-grade leakage. Rotor conjugation
   preserves grade, and the implemented grade-diagonal damping also preserves
   grade. The two operators commute. The variant can anisotropically change the
   relative grade norms, but it cannot mix vector coefficients into bivectors.
   Its failure is empirical; "chaotic leakage" is not an explanation supported
   by the mathematics or recorded data.

2. The hybrid split was two complex and two GA channels at the default four-
   channel width, not four plus four. Each channel still contains eight real
   state values. The capacity-starvation interpretation is plausible but was
   not isolated from the doubled-kernel optimization and throughput penalty.

3. A Cl(3,0) rotor sandwich does not faithfully represent Q8 itself. Spin(3)
   is the unit quaternions, but the sandwich action factors through SO(3), so
   `R` and `-R` act identically. The full model can still distinguish Q8 through
   affine writes and surrounding nonlinear layers. Calling the action a
   perfect mirror of Q8 overstates the mechanism.

4. S3 does not automatically align with "3D permutations." It has faithful
   orthogonal realizations, but the empirical win does not reveal which one the
   network learned.

5. A chiral eight-real Spin(8) action and a general SO(8) action have the same
   local transition capacity. Triality permutes the vector and two half-spin
   representations of the same `so(8)` Lie algebra. A direct dense-skew versus
   chiral-generator experiment is still valuable, but it compares
   parameterization, numerical construction, and optimization—not two
   different 28-dimensional transition classes. A distinct triality claim
   requires the three coupled representations and a triality-invariant
   interaction.

## Recent work that materially changes the roadmap

- [The Expressive Limits of Diagonal SSMs for State-Tracking](https://arxiv.org/abs/2603.01959)
  proves that a single input-dependent complex diagonal SSM tracks a finite
  group at finite precision exactly when the group is Abelian. A k-layer stack
  is characterized by a subnormal series with Abelian factors. Consequently,
  no fixed-depth model in that class exactly tracks non-solvable A5.

- [DeltaProduct](https://arxiv.org/abs/2502.10297) constructs stable products
  of generalized Householder transitions and proves a two-layer DeltaNet can
  solve dihedral word problems. D4 therefore cannot separate rotors from
  modern low-rank noncommutative transitions.

- [Structured Sparse Transition Matrices](https://papers.neurips.cc/paper_files/paper/2025/file/77b830c18836a9b2e1395a4936dd687a-Paper-Conference.pdf)
  introduces PD-SSM, with permutation-diagonal transitions and optimal finite-
  automaton state size. It is a more relevant state-tracking baseline than an
  unconstrained dense matrix alone.

- [A Held-Out Transition-Pair Falsifier](https://arxiv.org/abs/2606.07254)
  demonstrates why length extrapolation with complete local transition
  coverage is insufficient. Its hard-projected model reaches million-token
  exactness, while softened variants collapse. This directly motivated the
  split now implemented here.

- [Sequential-Parallel Duality in Prefix-Scannable Models](https://arxiv.org/abs/2506.10918)
  formalizes affine actions as an associative monoid and confirms that
  associativity—not commutativity—is the relevant scan property. It also shows
  that more general non-associative aggregation can trade constant memory for
  logarithmic memory.

- [M2RNN](https://arxiv.org/abs/2603.14360) shows that nonlinear matrix-valued
  recurrent layers can improve state tracking and large-scale language models,
  but gives up the exact associative affine recurrence used here.

- [MuonSSM](https://arxiv.org/abs/2606.30461) conditions low-rank memory writes
  using a Newton-Schulz step and momentum rather than merely constraining the
  transition. This suggests that our affine write geometry deserves as much
  attention as the rotor.

- [When Does Recurrence Become an Algorithm?](https://arxiv.org/abs/2607.20594)
  finds that architecture and curriculum select which algorithm SGD learns on
  group word problems. This reinforces treating the current seed/phase
  transitions as optimization phenomena rather than capacity alone.

- [Exploring Triality Explicitly](https://arxiv.org/abs/2502.14016) constructs
  the three eight-dimensional D4 representations and maps between them. It is
  the appropriate mathematical starting point if the coupled triality model is
  eventually justified.

## New empirical results

### Held-out S3 pair

Training length was 16. The ordered pair `132 -> 213` appeared zero times in
256,000 training sequences and in every evaluation sequence.

| Family | L2 | L4 | L8 | L16 | L32 | L64 |
|---|---:|---:|---:|---:|---:|---:|
| complex | 100.00% | 91.80% | 41.82% | 33.74% | 20.80% | 16.26% |
| selective GA | 100.00% | 98.19% | 82.86% | 66.28% | 32.08% | 15.94% |

Both infer the unseen pair itself. GA preserves the resulting state much
longer, but both reach six-class chance at length 64. The GA loss break around
steps 650-700 coincides with a rise in second-layer spectral participation,
not angle collapse or exploding state norm.

### A5 with all 60 elements exposed as tokens

Both models fail and learn only the trivial first-position copy. This version
is not diagnostic: a tiny model must learn 60 separate input transitions.

### A5 generated by two elements

Inputs were restricted to the 3-cycle `23145` and 5-cycle `23451`, which were
verified to generate all 60 A5 elements. Every prefix remains supervised over
the full 60-state output space.

| Seed | Complex L16 | Complex L32 | GA L16 | GA L32 |
|---:|---:|---:|---:|---:|
| 0 | 66.33% | 2.22% | 99.85% | 52.34% |
| 1 | 51.71% | 2.56% | 99.93% | 45.73% |
| 2 | 44.43% | 2.39% | 99.98% | 41.06% |
| mean | 54.16% | 2.39% | 99.92% | 46.38% |

This is the strongest positive result. It repeats across every seed and aligns
with A5's faithful icosahedral SO(3) representation and the diagonal-SSM
expressivity theorem. Nevertheless, both families approach chance at length
64, so the learned action is approximate.

### Held-out A5 generator pair

The pair `23145 -> 23451` was absent from training and required at evaluation.
Both models achieved near-zero loss on their restricted training language but
0% on the length-2 unseen pair and near-chance longer-horizon accuracy. This
shows that the positive random-word result still uses complete local-bigram
coverage. It does not establish systematic operator composition.

## Chosen path

1. Add explicit mechanism metrics: affine homomorphism error, group-relation
   residuals, commutator separation, prototype margin, and state drift under
   identity words.
2. Build a pure norm-preserving group-action rotor with a nonzero initial orbit
   state and no generic affine write. This is a mechanism upper bound, not a
   language model. Test whether structural composition alone passes the A5
   held-out-pair split and survives length 64+.
3. Compare against products of Householder transitions and PD-SSM-style
   permutation-diagonal transitions under identical generator and held-out
   splits.
4. Reintroduce writes using a conditioned/normalized update inspired by
   MuonSSM, while retaining an associative affine scan where possible.
5. Move to MQAR, selective copy, and code-like partially observed automata only
   after a candidate passes the mechanism gate.
6. Defer Spin(8). If pursued, compare direct and chiral SO(8)
   parameterizations honestly, then isolate a genuinely triality-specific
   three-representation coupling.

The immediate opportunity is not higher dimension. It is converting the
rotor's strong approximate A5 representation into a stable, demonstrably
compositional operator.

## Continuation result

That mechanism program has now been executed. The write-free two-generator
split was found to collapse the training language; an inverse-augmented
60-state, 15/16-bigram split corrected it. Exact A5 rotor and Householder
actions pass through length 512, while data-only training learns a
decoder-stable but state-inexact quotient. At the ten-seed checkpoint, GA
converges at length 16 in 8/10 runs and passes the complete long-length gate in
5/10; two-Householder transitions converge in 10/10 and pass the complete gate
in 8/10. Dense GA trajectories distinguish two early non-convergence failures
from later composition-retention failures. A follow-up 2.2-radian rotor chart
cap removes the convergence failures (10/10) and raises mean L64/L128 accuracy
to 90.0%/85.2%, but leaves the strict long-length gate at 5/10; chart
conditioning is therefore necessary but not sufficient. Post-warmup mean and
tail Cayley-closure pilots also fail to improve long-word accuracy, directing
the next objective toward path-coherent state drift rather than one-step
aggregate closure. That objective now succeeds: a preregistered length-64
path-holonomy loss raises capped GA's ten-seed functional gate from 5/10 to
8/10 and mean L128 accuracy from 85.2% to 97.5%, while retaining 100% L16 in
every seed. Seven seeds also pass the stricter positive alternate-path margin
contract; the remaining failures expose length-selective phase aliasing. This
is explicitly not the original `1e-3` raw-homomorphism mechanism gate, which
remains 0/10 for data-trained runs. See
[MECHANISM_GATE_RESULTS.md](MECHANISM_GATE_RESULTS.md) for the pre-registered
criteria, per-seed results, normalized span decomposition, and revised
interpretation.
