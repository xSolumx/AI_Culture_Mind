# Pure Spin(8) Latent-Increment Validation Results

Protocol frozen: **2026-08-17T03:34:00+02:00**<br>
Fresh validation executed: **2026-08-17T03:35--03:39+02:00**

## Outcome

The maintained Pure Spin(8) transport layer passes every frozen gate in fresh
seeds 1, 2, and 3. It identifies token-local actions without receiving the
teacher's 28 Lie coordinates, then composes the never-trained ordered pair
`a_half_center,a_half_center` into the nontrivial center element.

Every forced length-128 center and identity row is classified correctly in all
three seeds and both relation positions. Across those six splits,
post-relation MSE is `2.66e-5`--`2.87e-4` with median `1.13e-4`. Median Mamba-2
and GRU errors are `0.13857` and `0.13361`, or 1,223x and 1,179x the Pure
Spin(8) median under this parameter-near synthetic protocol.

This closes the specific supplied-coordinate objection to the earlier
triality-transport benchmark. It does not establish state-, compute-, or
prior-matched superiority.

## Frozen cohort

All candidates receive identical token IDs, every-prefix 24-scalar targets,
training schedules, update count, and optimizer settings within a seed. The
parameter counts are 892 Pure Spin(8), 891 Mamba-2, 887 GRU, and 874 token-only
MLP, a maximum spread of 2.02%. Their persistent state sizes are 24, 160, 9,
and 0 scalars respectively; state is explicitly not matched.

Training contains every ordered token pair except `(a,a)`. Evaluation pairs
that relation with `(b,b_inverse)` inside byte-identical contexts. The teacher
contract verifies that the two rows have equal vector targets and opposite
half-spin targets after the relation.

## Fresh results

| Seed | Action RMSE | Early L128 post MSE | Late L128 post MSE | Center/identity rows |
|---:|---:|---:|---:|---:|
| 1 | `1.586e-3` | `2.657e-5` | `5.047e-5` | `256/256` |
| 2 | `3.146e-3` | `9.726e-5` | `1.842e-4` | `256/256` |
| 3 | `3.887e-3` | `1.294e-4` | `2.872e-4` | `256/256` |

The row count is 64 center plus 64 identity examples in each of two L128
splits per seed. Center classification, center-row correctness, and
identity-row correctness are all exactly 1.0.

Learned local relation residuals:

| Seed | `a^2` vector `I` RMSE | `a^2` spinor `-I` RMSE | `b_inverse*b` `I` RMSE |
|---:|---:|---:|---:|
| 1 | `4.240e-4` | `4.240e-4` | `5.624e-4` |
| 2 | `6.818e-4` | `6.818e-4` | `1.207e-3` |
| 3 | `5.852e-4` | `5.852e-4` | `1.748e-3` |

The learned coordinates themselves are not compared because exponential
coordinates are non-unique; the faithful actions and their relations are the
identified objects.

## Controls and interpretation

At length 128, Mamba-2 and GRU center classification remains near chance. The
token-only ablation also remains near chance and cannot transport arbitrary
prefix context. These controls reject memorize-the-current-token as an
explanation and show that the algebra-matched recurrent prior is decisive at
this small budget.

What the result establishes:

- the model does not need supplied teacher coordinates for this finite latent
  dictionary;
- every-prefix supervision can identify legal local Spin(8) actions without
  exposing the held-out relation; and
- associative action composition carries the unseen central sign from length
  16 training through length 128 in three fresh seeds.

What it does not establish:

- that Mamba-2 or GRU cannot solve the task at another width, state size,
  optimizer, or budget;
- a natural-data or generic language-model advantage;
- a state- or compute-matched comparison;
- triality necessity versus every other faithful Spin(8) realization; or
- any unrestricted Dirac--Gram/D-optimality theorem.

## Reproducibility

The independent validator strictly rehashes and reloads all 12 checkpoints,
requires all three seeds to pass, and verifies distinct training and evaluation
schedules. Aggregate artifact:
`artifacts/pure_spin8_latent_increment_validation_seeds1_3.json`<br>
SHA-256:
`c3d49145fb710c43aa087262212e4005f887995ffb67f399a49afb57e8ae51a2`

Source artifacts:

- seed 1: `a06260a56df6817742ec25ddf2267a448f695bda4ff45295dafc0e41859acbe2`;
- seed 2: `fc70fbaffbcc39f609817942d829b4132e6cd3c127e3f7be4cc6e7710fcda15c`;
- seed 3: `d6117a55235a2508bbf4a8a9cd88d23361d4ee7b9b8747678f6a40a1be98f6b3`.

Maintained-model checkpoint hashes:

- seed 1: `b578270d0d0737a190d41b0363d9a7ee16611db28e737e9caab455e1cb4962fa`;
- seed 2: `2239856eae20126bc0fb88619e4a3dba3c1c81fa7973010e82600b209385db11`;
- seed 3: `7934b658ed176e4539a9b3f97bf4c57884945f491ef913b50e1b1ee17d02a03f`.

Protocol:
[`PURE_SPIN8_LATENT_INCREMENT_PROTOCOL.md`](PURE_SPIN8_LATENT_INCREMENT_PROTOCOL.md)<br>
Runner:
[`benchmark_pure_spin8_latent_increment.py`](../benchmark_pure_spin8_latent_increment.py)<br>
Validator:
[`validate_pure_spin8_latent_increment.py`](../validate_pure_spin8_latent_increment.py)

## Next falsifier

Replace the finite symbolic dictionary with noisy continuous observations from
which increments must be inferred online, then add separately state-matched and
measured-compute-matched controls. The compiled dictionary path is useful for
deployment, but it must not be mistaken for solving that continuous inference
problem.
