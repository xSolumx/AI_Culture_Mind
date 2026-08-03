# Joint Exact A5 Rounding Results

## Result

The preregistered breakthrough gate passed.

Across all ten deterministic GA checkpoints, replacing only the dominant
anchor channel by one globally aligned exact A5 action:

- passes the untouched changed-order-3 alphabet at every dense length from
  L16 through L256;
- passes L4096 in all ten seeds, with a population floor of `96.88%`;
- reduces vector-action homomorphism RMS to `1.20e-7-2.32e-7` in float32;
- preserves or improves every seed's untouched dense floor; and
- leaves the learned initial state, auxiliary channel actions, decoder, bias,
  logit scale, and previously selected soft gates unchanged.

The experiment therefore separates two claims that short-context accuracy had
previously conflated. The learned system discovers a close, useful A5 action,
but its small coherent defects are not safe under indefinite composition. A
joint projection onto one exact representation removes that failure while the
frozen learned decoder ensemble remains useful.

## Prospective status

`JOINT_A5_ROUNDING_PREREGISTRATION.md` was written after the irrep/Lie-closure
audit and before any checkpoint was evaluated on generator class 11. Classes
0, 1, and 2 were already observed regression sets. Class 11 was the untouched
primary distribution and changes the order-3 inverse pair to `13425`, `14235`.

No hyperparameter or gate was changed after class-11 evaluation.

## Controlled variants

All variants use the same frozen soft decoder gates and identical evaluation
batches.

1. `learned` retains the checkpoint's learned anchor actions.
2. `independent_angle` retains every learned rotation axis but snaps each
   token angle to its exact labeled cyclic order. This fixes the individual
   cyclic relations without enforcing mixed relations.
3. `joint_exact` globally aligns the exact branch-0 real 3D A5
   anti-representation to all four learned token actions, then replaces the
   anchor actions jointly. The aligned SO(3) actions are converted back through
   the Cl(3) rotor chart, preserving a coherent vector/bivector rotor action.

This is an oracle-structured mechanistic intervention: the branch and token
group elements are known. It is not a claim that the unmodified optimizer
already reaches an exact representation, nor that the projection has yet been
learned without group labels.

## Untouched changed-order-3 result

| seed | learned dense floor | angle-only dense floor | joint-exact dense floor | learned L4096 | angle-only L4096 | joint-exact L4096 |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 96.09% | 100.00% | 99.90% | 19.34% | 99.61% | 100.00% |
| 1 | 100.00% | 100.00% | 100.00% | 70.31% | 56.05% | 100.00% |
| 2 | 97.07% | 100.00% | 100.00% | 27.54% | 99.22% | 100.00% |
| 3 | 99.80% | 100.00% | 100.00% | 57.03% | 78.91% | 100.00% |
| 4 | 96.88% | 99.90% | 99.90% | 30.86% | 97.27% | 99.80% |
| 5 | 100.00% | 100.00% | 100.00% | 79.88% | 100.00% | 100.00% |
| 6 | 97.95% | 99.51% | 99.51% | 42.58% | 96.09% | 96.88% |
| 7 | 100.00% | 100.00% | 100.00% | 56.25% | 100.00% | 100.00% |
| 8 | 99.02% | 100.00% | 100.00% | 38.48% | 73.83% | 100.00% |
| 9 | 98.14% | 100.00% | 100.00% | 36.91% | 95.90% | 100.00% |

Dense evaluation uses two deterministic batches of 512 at every multiple of
16 from L16 through L256. L4096 uses one deterministic batch of 512.

At the dense horizon, all three variants pass all ten seeds. That apparent
parity is misleading. At L4096:

| variant | seeds at least 90% | population floor | mean accuracy |
|---|---:|---:|---:|
| learned | 0/10 | 19.34% | 45.92% |
| independent angle | 7/10 | 56.05% | 89.69% |
| joint exact | **10/10** | **96.88%** | **99.67%** |

The learned variant fails every seed at L4096 despite a dense floor of at
least `96.09%`. Independent angle rounding looks essentially solved through
L256, but three seeds still fail at L4096. Seed 1 is especially diagnostic:
angle snapping lowers L4096 accuracy from `70.31%` to `56.05%` even though it
makes every individual cyclic relation numerically exact. Correct token orders
are therefore insufficient; the mixed group relations are causal.

## Mechanism result

| diagnostic across ten seeds | learned | independent angle | joint exact |
|---|---:|---:|---:|
| vector homomorphism RMS | `0.00696-0.01732` | `0.000527-0.004825` | **`1.20e-7-2.32e-7`** |
| maximum mixed-relator RMS | nonzero | up to `0.01133` | **up to `4.91e-7`** |
| strict mechanism gate | 0/10 | not credited | **10/10** |

`joint_exact` passes the preregistered float32 gate of homomorphism RMS at most
`1e-5` and maximum at most `1e-4`. Its remaining residual is numerical rotor
conversion and matrix arithmetic, not learned holonomy. In exact arithmetic,
the constructed representation composes exactly.

This does not retroactively make the raw learned model pass the original
mechanism gate. The correct gate hierarchy is now:

- learned raw full-operator `1e-3` gate: **0/10**;
- learned dense behavioral gate through L256: **10/10** on class 11;
- learned L4096 gate on class 11: **0/10**;
- post-training, oracle-structured joint-exact anchor projection: **10/10** on
  both the strict mechanism gate and class-11 L4096 behavior.

## Regression alphabets

The intervention does not trade away the already-observed distributions.

| alphabet | learned dense floor | angle-only dense floor | joint-exact dense floor |
|---|---:|---:|---:|
| original | 99.12% | 99.80% | 99.80% |
| class 0 | 97.56% | 99.51% | 99.71% |
| class 1 | 99.32% | 99.71% | 99.71% |
| class 2 | 99.80% | 98.44% | 99.90% |
| untouched class 11 | 96.09% | 99.51% | 99.51% |

On the original alphabet at L4096, learned passes only `2/10` with a `46.29%`
population floor. Independent angle and joint exact pass `10/10`, with floors
of `97.27%` and `99.61%` respectively. On the more stringent untouched class
11, only joint exact is uniformly reliable.

## What was falsified

1. **Short dense sweeps certify the mechanism.** They do not. All variants
   pass through L256 while their L4096 reliability differs radically.
2. **Exact generator orders are enough.** They are not. Independent angle
   rounding eliminates cyclic residuals but leaves mixed-relator error and
   long-horizon failures.
3. **Orthogonality prevents destructive drift.** It prevents norm explosion,
   not coherent orientation error. The logged maximum norm error remains small
   while classification collapses.
4. **The auxiliary ensemble is the source of long drift.** It is held fixed.
   Replacing one anchor channel jointly repairs the failure, identifying the
   approximate group action as the cause.

## Breakthrough statement

The strongest justified claim is:

> A capped Cl(3) rotor recurrence trained only through finite words discovers,
> in every seed, a dominant channel close enough to a faithful icosahedral A5
> action that one global representation projection preserves its frozen
> learned decoder ensemble. Joint exact projection—not independent cyclic
> rounding—then converts near-perfect finite-horizon behavior into uniform
> 10-seed L4096 composition on an untouched changed-generator alphabet.

This is a mechanistic breakthrough, not yet a general sequence-model
breakthrough. It gives a concrete architectural target: learn in the tangent
space, but periodically or finally retract a whole token-action family onto a
shared representation manifold. The constraint must be joint across tokens;
per-token normalization cannot enforce the mixed relations.

## Next research gate

The next experiment should remove the oracle from the projection while
retaining its joint structure:

1. parameterize one shared near-exact representation plus a learned global
   conjugation, rather than four independent token rotors;
2. infer or optimize the discrete irrep branch without class-11 labels;
3. train with a differentiable group-consistency/retraction step and compare it
   against the current unconstrained rotor chart at equal budget;
4. repeat the L4096 test on new groups and generator sets before returning to
   Spin(8) or language modeling.

The design lesson for the future Spin(8) model is precise: noncommutative
associativity gives an efficient scan, but infinite-horizon reliability needs
the learned token transitions to remain on one coherent representation
manifold, not merely inside the ambient spin group.

This next gate is now complete. `SELF_COMPILING_RETRACTION_RESULTS.md` removes
the supplied exact matrices, character values, branch choice, and anchor
choice. A regular-representation compiler plus per-step joint tangent
retraction passes the strict mechanism and untouched L4096 gates in all ten
seeds. The remaining oracle is the finite multiplication table itself.

## Artifacts

- `joint_a5_rounding.py`
- `JOINT_A5_ROUNDING_PREREGISTRATION.md`
- `joint_a5_rounding_10seeds.json`
