# Self-Compiling Joint Retraction Results

## Verdict

Every preregistered gate passed.

The system now performs the complete sequence that the oracle-rounding result
left open:

1. train four rotor channels with independent unconstrained token parameters;
2. detect an approximate faithful channel from its learned actions;
3. construct both exact 3D irreducible candidates from the A5 multiplication
   table alone, without character values or supplied icosahedral matrices;
4. select the nearest candidate and global conjugation automatically;
5. continue ordinary ambient-gradient training, but jointly retract all four
   anchor-token actions through one shared conjugation after every step; and
6. decode only from the resulting exact recurrent channel.

All ten seeds self-compiled. Every seed scored 100% at every dense L16-L256
length and at L4096 on both the original alphabet and the prospectively
untouched changed-generator class 22.

This is the first data-trained result in the project to pass both the strict
mechanism gate and the long-horizon changed-generator behavioral gate in all
ten seeds.

## Prospective status

`SELF_COMPILING_RETRACTION_PREREGISTRATION.md` was written before the cohort
was trained or class 22 was evaluated. Classes 0, 1, 2, and 11 had already been
observed. Class 22 was reserved structurally because it introduces a third
order-3 inverse pair, `13542`, `15243`; it was not selected by accuracy.

The compiler seed, trigger thresholds, training schedule, class-22 index,
evaluation batches, and gates were unchanged after the first result.

An execution smoke exposed one implementation error before the formal cohort:
the initial code fed float32 rotor-conversion output back as the next exact
reference. That preserves conversion error instead of remaining on the exact
orbit. The formal cohort keeps the group-table-derived float64 family as the
persistent reference and uses float32 only for model execution. No scientific
threshold or model-selection rule changed.

## The mathematical construction

Let `L_g` and `R_g` be the exact left- and right-regular permutation actions of
a finite group. They commute:

`L_h R_g = R_g L_h`.

Choose deterministic generic coefficients with `c_g = c_(g^-1)` and form the
self-adjoint right-group-algebra element

`C = sum_g c_g R_g`.

Because `C` commutes with every `L_h`, each eigenspace of `C` is invariant
under the left action. For a generic commutant element, multiplicity directions
split while the irrep dimension remains as the eigenvalue multiplicity. If
`U` is an orthonormal basis of a three-dimensional eigenspace, then

`rho_U(h) = U^T L_h U`

is an exact three-dimensional representation. The A5 regular action yields
two distinct 3D character vectors, recovering both inequivalent irreps without
inserting `phi`, `phi'`, a character projector, or geometric oracle matrices.
Equivalent multiplicity copies are deduplicated by the character vectors that
the construction itself produces.

For learned token matrices `T_s`, every candidate family is fitted with one
global `Q in SO(3)`:

`min_Q sum_s ||T_s - Q rho(s^-1) Q^T||_F^2`.

The nearest well-separated candidate is selected. No class-22 sequence or
label participates.

## Joint tangent retraction

After compilation, AdamW still produces independent ambient token updates.
For the current exact family `A_s`, the retractor solves the three-parameter
least-squares tangent problem

`min_Omega sum_s ||T_s - (A_s + [Omega, A_s])||_F^2`,

where `Omega` is skew-symmetric. It then updates every token with the same
conjugation:

`A_s <- exp(Omega) A_s exp(-Omega)`.

Consequently all cyclic and mixed relations remain jointly exact. This is not
independent token normalization under another name: four nominally separate
ambient updates are projected onto one shared three-dimensional conjugacy
orbit.

## Discovery and exactness

| seed | trigger step | anchor | trigger RMS | runner-up RMS | mean ambient update RMS | mean tangent norm | final homomorphism RMS |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 850 | 1 | 0.026103 | 0.538740 | 3.65e-5 | 2.61e-5 | 1.14e-7 |
| 1 | 500 | 0 | 0.034212 | 0.536507 | 6.11e-5 | 4.08e-5 | 1.55e-7 |
| 2 | 500 | 0 | 0.075047 | 0.537793 | 1.08e-4 | 6.56e-5 | 2.08e-7 |
| 3 | 400 | 2 | 0.040262 | 0.538447 | 1.01e-4 | 4.97e-5 | 1.46e-7 |
| 4 | 600 | 1 | 0.030302 | 0.539174 | 6.60e-5 | 3.10e-5 | 1.61e-7 |
| 5 | 400 | 2 | 0.036565 | 0.539282 | 7.83e-5 | 3.42e-5 | 1.57e-7 |
| 6 | 550 | 2 | 0.075186 | 0.539648 | 9.71e-5 | 5.89e-5 | 1.76e-7 |
| 7 | 500 | 2 | 0.042233 | 0.539646 | 9.52e-5 | 3.84e-5 | 1.48e-7 |
| 8 | 450 | 0 | 0.034329 | 0.539046 | 7.39e-5 | 4.05e-5 | 2.07e-7 |
| 9 | 400 | 3 | 0.043498 | 0.537050 | 7.92e-5 | 4.78e-5 | 1.98e-7 |

The trigger occurs at steps 400-850 and selects four different channel
indices across the cohort. Trigger commutators range from `1.336` to `1.392`.
The wrong-irrep fit remains about `0.537-0.540`, leaving a large discrete
selection margin. There are no retries or manual channel choices.

Post-trigger ambient updates and tangent motion are nonzero in every seed, so
the result is not merely a frozen exact initialization. The evidence does not
yet show that this tangent motion is necessary for final accuracy; it shows
that continued training is compatible with exact joint closure.

Final float32 diagnostics across all seeds:

- vector homomorphism RMS: at most `2.08e-7`;
- maximum pairwise homomorphism residual: at most `4.82e-7`;
- maximum mixed-relator RMS: at most `5.29e-7`;
- strict mechanism gate: **10/10**.

## Untouched behavioral result

| evaluation | pass count | population floor | mean |
|---|---:|---:|---:|
| original, dense L16-L256 | 10/10 | 100.00% | 100.00% |
| untouched class 22, dense L16-L256 | 10/10 | 100.00% | 100.00% |
| original, L4096 | 10/10 | 100.00% | 100.00% |
| untouched class 22, L4096 | 10/10 | 100.00% | 100.00% |

The earlier oracle-structured experiment obtained a `96.88%` class-11 L4096
population floor with a frozen decoder ensemble. Here the decoder is trained
after autonomous compilation to use the exact anchor alone; no auxiliary
channel is needed at evaluation.

## Floating-point control

For every L4096 example, the evaluator compares the sequential float32 state
against a direct canonical group action applied once to the same initial
state. On untouched class 22:

- worst seed RMS path drift: `8.67e-4`;
- maximum observed path drift: `8.76e-4`;
- classification errors: zero.

This quantifies the numerical execution floor Claude flagged. Float32
accumulation is real but decoder-safe here. It cannot explain the earlier
learned-action collapse to `19.34-79.88%`: the same recurrence implementation,
length, dtype, and hardware reaches 100% once mixed relations are exact.

## Gate ledger

| preregistered gate | result |
|---|---:|
| autonomous trigger by step 1500 | **10/10** |
| float32 exact mechanism thresholds | **10/10** |
| untouched class-22 dense floor at least 90% | **10/10; floor 100%** |
| untouched class-22 L4096 at least 90% | **10/10; floor 100%** |
| original L4096 at least 90% | **10/10; floor 100%** |
| nonzero ambient and tangent updates | **10/10** |

## Honest boundary

This is autonomous representation *selection and compilation*, not discovery
of an unknown algebra from raw tokens. The system is still supplied:

- the finite A5 multiplication table; and
- the mapping from the four training tokens to elements of that table.

It is not supplied an irrep branch, character values, exact low-dimensional
matrices, an anchor channel, a global alignment, or any class-22 data. The
result therefore closes the oracle-matrix gap from the previous experiment but
does not close the Cayley-table gap emphasized in Claude's review.

The `max_rotor_angle = 2.2` cap also makes the lower-angle 3D branch accessible
and excludes the exact 144-degree token action of the other branch. The
compiler reconstructs and compares both representations, but the architecture
still biases which one SGD can approach.

## Result statement

> A recurrent rotor model can discover an approximate noncommutative action by
> ordinary SGD, automatically compile the nearest exact irrep from a supplied
> finite-group table, and continue training through independent ambient
> gradients followed by a shared joint retraction. This produces exact
> constant-state streaming and 100% ten-seed composition at L4096 on an
> untouched generator family, without supplied representation matrices,
> character values, branch choice, channel choice, or auxiliary decoder code.

The new architectural primitive is a **self-compiling recurrent
representation**: search freely in an overparameterized ambient space, detect
a stable algebraic object, compile it into an exact minimal state mechanism,
then optimize only through geometry-preserving joint retractions.

## Next gate

The next genuinely harder problem is to remove the supplied Cayley table.
Candidate routes are:

1. learn a latent multiplication table from word-equivalence constraints and
   require associativity before invoking the regular-representation compiler;
2. build the commutant from empirically discovered closed paths rather than
   named group elements;
3. jointly infer discrete state aliases, token permutations, and invariant
   subspaces using train/validation/test word splits; and
4. repeat on non-isomorphic finite groups before lifting the same
   search-compile-retract principle to Spin(8) selective dynamics.

This result clears the mechanistic gate that was blocking larger geometry. It
does not yet establish a language-model advantage, but it supplies a concrete
way to prevent learned noncommutative recurrent transitions from drifting off
their discovered algebraic manifold.

The Cayley-table gate is now complete under exact prefix supervision; see
`LATENT_CAYLEY_RETRACTION_RESULTS.md`. All ten seeds reconstruct a latent
60-element regular permutation group from observed transitions alone and pass
an untouched L16384 gate at 100%. The remaining problem is algebra recovery
from incomplete, endpoint-only, or noisy equivalence evidence.

## Artifacts

- `representation_retraction.py`
- `train_self_compiling_retraction.py`
- `SELF_COMPILING_RETRACTION_PREREGISTRATION.md`
- `self_compiling_retraction_10seeds.json`
- `self_compiling_checkpoints/self_compiling_retraction_seed{0..9}.pt`
