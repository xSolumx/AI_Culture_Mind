# Final-only octonion-law recovery results

Executed: **2026-08-16**

## Outcome

The fixed-length L16 experiment is a recorded negative result: the
28-parameter hidden-basis model converged in only one of three Haar bases and
the 512-parameter dense leaf map converged in none. Replacing the single depth
with a frozen L2 -> L4 -> L8 -> L16 curriculum changes the optimization result:
all nine structured models and all nine dense controls pass the L128 `1e-3`
gate.

The stronger conclusion is not that the law is uniquely identified. Every
curriculum depth is even, so terminal supervision is invariant under replacing
every learned leaf operator by its negative. A post-protocol odd-length audit
finds four recovered gauges in `G2` and five in `-G2`. At held-out L17, the
negative-coset models have ordinary MSE approximately `0.5`, but all nine have
parity-corrected MSE at most `5.26e-12`. This exactly explains why the earlier
unsigned-`G2` audit failed.

## Frozen cohorts

The first protocol trains at L16 for 1,000 updates with terminal supervision
only. Its structured L128 outcome is 3/9 successful runs, all in basis 1. Its
dense L128 outcome is 0/9. The artifact intentionally retains
`all_required_checks_passed: false`.

The curriculum protocol uses 250 updates at each of L2, L4, L8, and L16. L128
MSE ranges are:

| Candidate | basis 0 | basis 1 | basis 2 |
|---|---:|---:|---:|
| 28-parameter hidden basis | `1.11e-11`--`1.37e-11` | `1.29e-11`--`1.81e-11` | `1.02e-11`--`3.12e-11` |
| 512-parameter dense leaf | `1.83e-6`--`2.16e-6` | `2.00e-7`--`2.35e-7` | `9.84e-5`--`1.38e-4` |
| Transformers Mamba-2 | `0.12580` | `0.12658` | `0.12607` |
| DeltaProduct reference | `0.12523` | `0.12575` | `0.12537` |

The structured checkpoints continue to L1024 at MSE at most `2.32e-10`.
The curriculum artifact still correctly records a top-level failure: basis
0's float32 exact oracle scored `2.204e-12` against a frozen `<2e-12` gate.
That numerical threshold miss is not edited away.

## Signed-`G2` theorem exposed by the audit

Let `Q` be the true hidden Haar basis, `Qhat` the recovered basis, and
`A = Qhat^T Q`. The nine learned solutions satisfy, to maximum residual
`3.70e-6`,

`A e0 = s e0`, and `A L_u A^T = s L_(A u)`, where `s` is `+1` or `-1`.

Thus each recovered leaf is `s` times the target leaf and a length-`L`
terminal product differs by `s^L`. Even-only data cannot distinguish the two
cosets. The L17 diagnostic is a direct falsifier and observes the predicted
sign in every run.

This is an empirical identification result plus an exact algebraic explanation
of the observed symmetry. It is not a global convergence theorem.

## Artifacts

- Fixed L16 cohort:
  `artifacts/octonion_final_only_replication1000.json`, SHA-256
  `7def2ca25ce7b04f11f282c06c16dea51ccf86d2a18eae1a4db679fa4d9e8f4a`.
- Curriculum cohort:
  `artifacts/octonion_final_only_curriculum1000.json`, SHA-256
  `4ddabe92d532361e146ee0c7c156237a30061c0c46299f0cb29e3b4951def322`.
- Original unsigned-`G2`/L1024 audit, retained as failed:
  `artifacts/octonion_final_only_curriculum_audit.json`, SHA-256
  `fcee07b7dcbe7ee651a4777de0c88c93d4411fde63caf79b1492853d6d896bec`.
- Corrected signed-parity audit:
  `artifacts/octonion_final_only_parity_gauge_audit.json`, SHA-256
  `32545cb8174bfd84cf0cb2d1a84202570af34dce20ff4f0e48dd9573118c8d30`.

Protocols:
[`OCTONION_FINAL_ONLY_PROTOCOL.md`](OCTONION_FINAL_ONLY_PROTOCOL.md) and
[`OCTONION_FINAL_ONLY_CURRICULUM_PROTOCOL.md`](OCTONION_FINAL_ONLY_CURRICULUM_PROTOCOL.md).

## Claim boundary and next falsifier

The result establishes realizability, a robust depth curriculum, long-length
operator extrapolation, and the exact parity non-identifiability of the frozen
even-only task. It does not establish natural-task performance, generic SSM
superiority, or unique recovery of the octonion law.

The next protocol must mix odd and even training depths, hold out one odd and
one even long depth, and leave local Lie coordinates latent. That removes the
`-G2` ambiguity by construction and prevents the task from reducing to a
supplied algebra-coordinate scan.
