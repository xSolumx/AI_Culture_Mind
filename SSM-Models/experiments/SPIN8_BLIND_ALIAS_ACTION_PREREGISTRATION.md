# Joint blind-action and continuous-alias gate preregistration

Date frozen: 2026-08-03, before implementation or results.
Baseline commit: `fede409`.

## Question

Can one endpoint-trained model simultaneously infer continuous semantic
addresses and complete an entirely unsupplied negative-chiral action family,
then extrapolate both mechanisms to unseen aliases, unseen value directions,
and action compositions through length 2048?

This is the first gate capable of supporting a Spin(8)-specific result. The
previous direct-slot controls were given the exact negative-chiral transport;
this experiment removes that oracle.

## Teacher and observations

Each seed samples four noncommuting token actions from hidden Spin(8)
bivectors. The learner receives:

- columns e0--e4 of each vector action;
- columns e0--e4 of each positive-chiral action;
- no action-matrix column from the negative-chiral representation;
- no teacher bivectors, logarithms, or unobserved columns;
- fresh paired write/query aliases, never a logical key ID;
- one-step retrieval endpoints whose negative values lie in one fixed hidden
  two-dimensional calibration subspace.

Training contains no multi-token action word and no negative value outside
that rank-2 subspace. Evaluation uses full-dimensional negative values and
random words at lengths 16, 32, 64, 128, 256, 512, 1024, and 2048.

## Prospective rank contract

Before optimization, every seed must satisfy:

- shared Spin(8) Jacobian rank 28 from the partial vector/positive columns;
- independent vector Stiefel observation rank 25;
- independent positive Stiefel observation rank 25;
- independent negative rank-2 endpoint observation rank 13.

The independent family therefore has rank 63 in 84 tangent coordinates and 21
unobserved degrees of freedom per token family. Invalid design seeds must be
resampled before training, never discarded after outcomes are known.

## Models

1. `oracle_triality`: exact actions and exact logical addresses;
2. `oracle_action_learned_alias`: exact actions with jointly balanced
   continuous-alias encoders;
3. `joint_triality`: one 28-coordinate token tangent shared by vector,
   positive, and negative representations, optimized jointly with the alias
   encoders;
4. `independent_binding`: separate 28-coordinate SO(8) tangents for each
   representation, evaluated through triality bind/vector transport/unbind;
5. `independent_direct`: the same independently learned family and aliases,
   storing values directly and transporting them with its learned negative
   action;
6. `direct_negative_oracle`: jointly learned aliases but supplied exact
   negative transport, retained only as a capacity ceiling;
7. `joint_action_oracle_alias`: jointly learned actions with supplied logical
   addresses, isolating action completion.

The joint and independent learned rows receive the same partial action labels,
alias batches, rank-2 negative values, target endpoints, optimizer schedule,
and training budget. The independent row has three times as many action
coordinates; failure cannot be attributed to lower parameter count.

## Joint endpoint objective

For a training pair `(x_write, x_query)`, positive key `p`, negative value
`n` in the calibration subspace, and one token `a`, the model must predict

\[
 \rho_-(a)n
\]

by both:

1. binding `p,n`, writing by the inferred address, transporting memory with
   the learned vector action, transforming the query key with the learned
   positive action, and unbinding;
2. direct learned negative-action transport.

The target is an endpoint vector, not a supplied negative action matrix. The
same loss aligns write/query aliases and the three action views. An unlabeled
marginal-balance term prevents semantic slot collisions. No class-specific
slot target is supplied.

## Scan boundary

Alias encoding and action construction depend only on current inputs. Once
constructed, every slot/direct update is affine in recurrent state. No
Sinkhorn iteration or state-dependent router runs inside the recurrence.
Parallel-prefix and recurrent execution must agree below `1e-9` with exactly
64 streaming-state scalars.

## Optimization

- seeds 0 through 9;
- fixed alias radii 0.05, 0.10, 0.15;
- validation radius 0.22 and untouched test radius 0.35;
- float64 action construction and final evaluation;
- one jointly optimized model per learned action family;
- fixed Adam schedule followed by deterministic LBFGS refinement;
- no per-seed restart, checkpoint choice, or hyperparameter change.

## Evaluation

Report for every seed and model:

- partial-observation MSE and rank audit;
- calibration-subspace and orthogonal-complement one-step action cosine;
- full-dimensional vector/positive/negative action cosine;
- triality equivariance and noncommutative commutator separation;
- write/query alias collisions, agreement, entropy, and marginal residual;
- full-dimensional mixed-key retrieval over the dense length sweep;
- scan parity, log-norm drift, and state size;
- exact training endpoint loss, including controls that fail extrapolation.

Every seed/length cell must contain at least 256 queries.

## Frozen gates

A `joint_triality` seed passes only if:

- partial action MSE is below `1e-8`;
- test write/query aliases have zero collisions, 100% center alignment, and at
  least 99% untouched-alias agreement;
- negative-action mean cosine is at least 0.9999 on the full space and the
  calibration complement;
- full-dimensional retrieval mean cosine is at least 0.995 at every dense
  length and individual cosine never falls below 0.98;
- mean relative squared retrieval error is below `1e-3` at every length;
- triality residual is below `1e-8`, commutator is at least 90% of oracle,
  scan error is below `1e-9`, and state size is 64.

Reliability requires at least 8/10 passing seeds.

Strong Spin(8)-specific support additionally requires:

- independent controls fit partial columns and rank-2 training endpoints below
  `1e-6` in all seeds;
- joint triality beats both independent binding and independent direct on
  complement negative cosine in every seed;
- joint triality beats both at length 2048 in at least 8/10 seeds;
- exact-action learned-alias and learned-action oracle-alias decompositions
  each pass at least 8/10.

## Prospective correction after the seed-0 causal smoke

This correction was written after inspecting seed 0 and before running seeds
1--9. It changes the interpretation of one control, not the observation design,
optimizer, evaluation data, or model parameters.

The `independent_binding` retrieval path transports a bound vector with the
learned vector action and transports its key with the learned positive-spinor
action. It therefore does not consume the learned negative-spinor action at
inference. Seed 0 made this bypass visible: binding retrieval remained nearly
exact even though the independently learned negative action was wrong on the
held-out calibration complement. Consequently, length-2048 binding accuracy is
not a valid behavioral test of negative-action completion.

The corrected causal interpretation is frozen as follows:

- `independent_binding` remains a required bypass ablation. Its retrieval may
  match the joint model, but it must not be described as completing the hidden
  negative action unless its negative complement and triality gates also pass;
- the behavioral extrapolation comparison is `joint_triality` versus
  `independent_direct`, because both paths must produce the transported negative
  value that was observed only on the rank-2 calibration plane;
- a per-seed behavioral win requires joint L2048 mean cosine at least `0.995`
  and independent-direct L2048 mean cosine at most `0.90`;
- a per-seed completion win requires joint complement cosine at least `0.9999`
  and a gap of at least `0.05` over the independent negative action;
- reliability requires at least 8/10 behavioral wins and completion wins in
  all ten seeds, while all independent controls still fit their supplied
  training evidence below `1e-6`.

The original strict-greater-than length comparison is retired because numerical
differences between two effectively exact binding paths are not a meaningful
effect size. Both the original criterion and this correction remain in the
record so the change cannot be mistaken for an unseen post-hoc rewrite.

## Post-cohort matched-path correction

The first ten-seed cohort was completed before this issue was noticed. During
the independent artifact audit, `joint_triality` was found to use the binding
path and therefore, like `independent_binding`, not to consume its learned
negative action during retrieval. Comparing that row with
`independent_direct` was a path mismatch. The first cohort artifact was
invalidated as final behavioral evidence before the result was frozen.

A `joint_direct` row is added without changing training, data, seeds,
hyperparameters, or thresholds. It uses the jointly learned negative action in
the same direct recurrence and evaluation stream as `independent_direct`.
The corrected gate requires:

- `joint_triality` and `joint_direct` each pass in at least 8/10 seeds;
- the L2048 behavioral win compares `joint_direct` with
  `independent_direct` under the already frozen `0.995`/`0.90` thresholds;
- `joint_triality` remains the binding-model result, and both binding rows
  remain explicit bypass diagnostics.

The complete cohort is rerun from the top. This is a post-cohort correction,
not a prospective one, and is labelled accordingly.

## Interpretation boundary

Passing demonstrates completion under a deliberately rank-deficient synthetic
observation design. Spin(8), the triality tensor, the number of semantic
classes, balanced frequency, and orthogonal alias centers remain architectural
priors. It is not language understanding or proof of advantage over full
Gated DeltaNet-2/EDA/Q-Delta. Those comparisons follow only after this gate.
