# Latent Cayley Recovery and Joint Retraction Results

## Verdict

Every preregistered gate passed.

The learner was not given the A5 multiplication table or the mapping from its
four token IDs to A5 elements. From training tokens and prefix class labels
alone, every seed:

1. reconstructed all deterministic state-token transitions;
2. recognized each token as a permutation of 60 anonymous states;
3. closed those permutations into a regular group of order 60;
4. reconstructed a Cayley table and token mapping in an arbitrary latent
   gauge;
5. derived exact 3D irrep candidates from that recovered table;
6. selected a learned rotor channel and candidate automatically;
7. continued ambient-gradient training with joint conjugacy retraction; and
8. scored 100% through L16384 on a prospectively untouched generator family.

This removes the explicit Cayley-table object used by the previous compiler,
but not its information content. With dense coverage, canonical prefix labels
are informationally equivalent to the transition table. The operation is
therefore deterministic automata preprocessing, not SGD discovering an
uncertain algebra. It is table-blind supervised reconstruction, not
unsupervised learning.

## Prospective status

`LATENT_CAYLEY_RETRACTION_PREREGISTRATION.md` was written before the cohort was
trained or class 33 was evaluated. Class 33 introduces a fourth previously
unseen order-3 inverse pair, `14352`, `15324`. Training settings, recovery
rules, compiler thresholds, evaluator seeds, and the L16384 gate were fixed
before results. No seed was restarted and nothing was repaired manually.

## Table-blind recovery

From adjacent supervised prefixes the learner records

`(label_(t-1), token_t, label_t)`.

For each token these observations define a map over the 60 anonymous labels.
Recovery is accepted only if all 240 edges are observed without conflict, each
token map is a permutation, closure contains exactly 60 permutations, the
action is regular on an arbitrary base label, and the reconstructed product
table replays every edge exactly.

Choosing a base label supplies only a gauge between labels and generated
permutations. Group multiplication is then recovered by permutation
composition. The true A5 table remains in the external data generator and
held-out evaluator; it is not used by recovery, holonomy, representation
compilation, trigger selection, or retraction.

The implementation fixes base label 0 and a deterministic generator/closure
order. Thus the gauge is arbitrary only in the mathematical sense that another
base would yield an isomorphic relabeling; it is not randomly chosen per seed.

## Recovery and discovery

| seed | full table step | minimum edge count | trigger step | anchor | trigger RMS | homomorphism RMS |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 1 | 850 | 1 | 0.026103 | 1.58e-7 |
| 1 | 2 | 3 | 500 | 0 | 0.034212 | 1.55e-7 |
| 2 | 2 | 2 | 500 | 0 | 0.075047 | 2.21e-7 |
| 3 | 1 | 1 | 400 | 2 | 0.040262 | 1.43e-7 |
| 4 | 1 | 1 | 600 | 1 | 0.030302 | 1.61e-7 |
| 5 | 2 | 4 | 400 | 2 | 0.036565 | 1.29e-7 |
| 6 | 1 | 1 | 550 | 2 | 0.075186 | 8.03e-8 |
| 7 | 1 | 2 | 500 | 2 | 0.042233 | 2.61e-7 |
| 8 | 1 | 2 | 450 | 0 | 0.034329 | 2.07e-7 |
| 9 | 1 | 1 | 400 | 3 | 0.043498 | 2.33e-7 |

The table is complete after one batch in seven seeds and two batches in three.
One batch contains 3,840 adjacent transitions for only 240 possible edges, so
this rapid recovery reflects dense prefix supervision rather than inference
from sparse evidence. All closures have order 60 and replay every edge.

Final float32 diagnostics across the cohort:

- homomorphism RMS at most `2.61e-7`;
- homomorphism maximum at most `6.72e-7`;
- mixed-relator RMS at most `5.82e-7`;
- strict mechanism gate: **10/10**.

## Behavioral result

| evaluation | pass count | population floor | mean |
|---|---:|---:|---:|
| original dense L16-L256 | 10/10 | 100.00% | 100.00% |
| untouched class 33 dense L16-L256 | 10/10 | 100.00% | 100.00% |
| original L4096 | 10/10 | 100.00% | 100.00% |
| untouched class 33 L4096 | 10/10 | 100.00% | 100.00% |
| untouched class 33 L16384 | 10/10 | **100.00%** | **100.00%** |

Class 33 was not used for recovery, holonomy, compiler selection, decoder
training, or model selection.

## L16384 numerical control

Sequential float32 states were compared with one direct canonical action for
the final element. Per-seed RMS drift ranges from `7.76e-5` to `6.22e-3`; the
largest observed drift is `6.24e-3`. Accuracy remains 100% in every seed.
Finite-precision drift is therefore measurable but decoder-safe, unlike the
catastrophic coherent holonomy of the earlier approximate representations.

## Gate ledger

| preregistered gate | result |
|---|---:|
| complete conflict-free transition recovery | **10/10** |
| 60-element regular closure and exact edge replay | **10/10** |
| automatic representation trigger by step 1500 | **10/10** |
| strict float32 mechanism thresholds | **10/10** |
| original and class-33 dense gate | **10/10; 100% floor** |
| original and class-33 L4096 gate | **10/10; 100% floor** |
| class-33 L16384 gate | **10/10; 100% floor** |
| nonzero ambient updates and tangent motion | **10/10** |

## What has now been established

The experimental chain separates four claims:

1. SGD finds useful approximate noncommutative rotor dynamics, but long words
   drift.
2. A certified oracle replacement proves mixed relations are the missing
   structure.
3. A supplied group table is enough to compile exact candidates and select the
   learned channel automatically.
4. Exact prefix-transition equivalence is enough to reconstruct the group
   table itself and complete the same compilation without a supplied algebra.

The fourth claim is the new result.

## Adversarial follow-up

Claude's review correctly required stronger checks before accepting “A5
recovered” or an independently arbitrary gauge. The prospective post-hoc audit
in `LATENT_CAYLEY_ADVERSARIAL_AUDIT.md` found:

- first-batch edge coverage is already `99.17-100%`;
- every recovered group is noncommutative and simple;
- conjugacy-class sizes are exactly `1, 12, 12, 15, 20` in all seeds;
- the deterministic recovered-to-evaluator map is an exact A5 table
  isomorphism in all seeds;
- there is one unique non-identity gauge across the cohort, with only label 0
  fixed, rather than ten independently varying gauges;
- original recovery elements are disjoint from class 33 and no training-token
  permutation equals a class-33 permutation;
- deleting a single recovered edge makes the current algorithm refuse
  recovery in all seeds; it does not infer missing transitions; and
- every checkpoint remains 100% at all 13 multiples of 1024 from L4096 through
  L16384.

The perfect behavioral sweep is therefore real, but the reduced-coverage
control is a clean negative result. The next gate is genuinely partial algebra
completion.

## Honest boundary

This is not unsupervised discovery from raw language. Every prefix has an exact
one-of-60 label; transitions are deterministic, noise-free, bijective, and
dense; the state count is known; and the tokens generate a finite regular
group. Under those conditions Cayley recovery is exact automata extraction.
The next gate must weaken supervision rather than merely extend length again.

## Result statement

> Dense exact prefix-equivalence data mechanically determines a latent finite
> transition group. A rotor recurrence can compile exact irreducible representations
> from the recovered regular action, autonomously compile a learned channel,
> and preserve exact joint dynamics under continued gradient training. Across
> ten seeds this table-blind pipeline attains 100% composition at L16384 on an
> untouched fourth generator family.

## Next gates

1. Hide state-token edges and complete only what associativity, bijectivity,
   and observed word equality force.
2. Remove prefix labels and infer equivalence from whole-word endpoint labels.
3. Add noisy or merged labels and uncertainty-aware algebra recovery.
4. Repeat across Q8, D4, S3, and groups without faithful real 3D irreps,
   selecting dimension from the recovered regular spectrum.
5. Relax inverses and regularity to compile finite semigroups and automata.

## Artifacts

- `latent_group_discovery.py`
- `train_self_compiling_retraction.py --table-blind`
- `LATENT_CAYLEY_RETRACTION_PREREGISTRATION.md`
- `latent_cayley_retraction_10seeds.json`
- `latent_cayley_adversarial_audit_10seeds.json`
- `latent_cayley_checkpoints/self_compiling_retraction_seed{0..9}.pt`
