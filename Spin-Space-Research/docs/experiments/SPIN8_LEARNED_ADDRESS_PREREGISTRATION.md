# Spin(8) learned-address gate preregistration

Date frozen: 2026-08-03, before implementation or results.
Action baseline commit: `8b86e7a`.

## Pre-reliability smoke amendment

A seed-0, 50-step-per-stage implementation smoke passed the joint row.  Before
running the frozen ten-seed cohort, this raised a necessary attribution check:
a sufficiently cold random Sinkhorn matrix can itself approach a permutation.
An `triality_joint_untrained` row is therefore added prospectively, using the
identical initialization and evaluation temperature but no endpoint updates.
Strong support now also requires every trained joint seed to beat that row at
length 2048 and no more than two untrained seeds to pass.  The smoke is not
part of the reliability cohort.

The untrained seed-0 control was then evaluated: it retained one rounded
collision, mean row entropy 1.5047, and length-2048 mean cosine 0.5304.  Thus
the parameterization alone did not pass, but this negative control remains in
the executable cohort rather than being inferred from one seed.

## Question

Can endpoint supervision recover a collision-free, token-dependent address
family while every recurrent update remains state-independent and therefore
associatively scannable?

This gate removes the address oracle only.  The Spin(8) action family and the
triality tensor remain supplied exactly.  Learned actions and learned
addresses must not be conflated in this experiment.

## Identifiability stress

There are eight logical keys and eight latent memory slots.  Training episodes
contain exactly one logical key, sampled uniformly across episodes.  A key is
written, transported by supplied noncommuting Spin(8) actions, overwritten,
and queried, but no training episode ever contains two distinct logical keys.

Consequently each key can select any one of the eight slots and obtain zero
training error.  Independent per-key normalization has no training evidence
that prevents two keys selecting the same latent slot.  Evaluation sequences
mix all eight keys and are therefore a prospective collision falsifier.

The global slot permutation is deliberately latent.  Recovery means a
collision-free permutation and correct retrieval; matching one privileged
slot labeling is not required.

## Variants

1. `triality_oracle`: supplied one-hot address and triality-bound memory;
2. `triality_independent`: independently row-softmaxed key addresses;
3. `triality_joint`: one jointly normalized Sinkhorn family over all keys and
   slots;
4. `direct_joint`: the same joint router and the same eight-by-eight recurrent
   state width, but values are stored directly and transported in the negative
   chiral representation rather than triality-bound and transported in the
   vector representation.
5. `triality_joint_untrained`: the joint row at initialization, evaluated
   without optimization as an attribution control.

Both learned routers receive the same initialization scale, temperature,
optimizer, training batches, loss, and discreteness penalty.  The direct
baseline is mandatory: parity means the contribution is joint latent-slot
identifiability, not a Spin(8)-specific memory advantage.

## Recurrent contract

For a token-dependent address row `w_t` and bound write `b_t`,

\[
  M_t = (1-w_t)\odot (A_t M_{t-1}) + w_t\odot b_t.
\]

`w_t`, `A_t`, and `b_t` depend on the current input event but never on the
running state.  Every step is therefore an affine map.  The implementation
must verify transition associativity and logarithmic-depth prefix-scan parity
against token-by-token recurrence.

The triality row stores
`bind(current_positive_key, current_negative_value)` and transports memory by
the supplied vector action.  It queries by exact triality unbinding.  The
direct row stores the negative value itself and transports it by the supplied
negative action.

## Optimization and evaluation

- seeds 0 through 9;
- training curriculum lengths 8, 16, and 32;
- Adam on router logits only;
- no mixed-key training episodes;
- float64 final evaluation;
- local GPU when it improves throughput;
- prospective mixed-key evaluation lengths 32, 128, 512, and 2048;
- at least 256 query events per reported length and seed;
- a dense diagnostic sweep at lengths 16, 32, 64, 128, 256, 512, 1024, and
  2048 for any learned joint row promoted as passing.

Report mean and minimum query cosine, relative squared error, rounded-slot
collisions, row/column stochastic residuals, router entropy, recurrent norm
drift, and parallel/recurrent error.  Report the whole ten-seed distribution,
not only its mean.

## Frozen gates

A learned joint seed passes only if:

- its rounded address family has zero collisions;
- maximum row- and column-sum residual is below `1e-6`;
- mean query cosine is at least `0.995` at every named evaluation length;
- minimum per-length mean query cosine is at least `0.98`;
- mean relative squared retrieval error is below `1e-3` at every named length;
- parallel/recurrent maximum error is below `1e-9` in float64;
- final recurrent state size is independent of sequence length.

The reliability gate is at least 8 of 10 passing seeds for `triality_joint`.
Strong support for joint normalization additionally requires it to have fewer
rounded-slot collisions and higher length-2048 cosine than
`triality_independent` in at least 8 of 10 seeds.

The oracle must pass numerically.  If `direct_joint` matches or exceeds
`triality_joint`, the result is evidence for joint family normalization but
not for a triality advantage.  A Spin(8)-specific advantage may be claimed
only if triality wins prospectively on state efficiency, extrapolation,
sample efficiency, or transport structure under an otherwise fair control.

## Interpretation boundary

Passing establishes discovery of a globally consistent latent address family
from single-key endpoint supervision and transfer to unseen multi-key
sequences.  It does not establish learned Spin(8) actions, content-dependent
state feedback, language-model utility, or superiority over delta-rule/fast-
weight memories.  Those remain separate gates.
