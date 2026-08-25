# Shared-family representation learning

**Research author:** Hayden Austin

## Scientific claim

Relational constraints imposed on a complete family of actions can remove
degrees of freedom that remain invisible when the actions are fitted or
normalized independently.

## Evidence map

| Result family | Canonical document | Status boundary |
|---|---|---|
| Joint A5 rounding | [`JOINT_A5_ROUNDING_RESULTS.md`](../../Spin-Space-Research/docs/experiments/JOINT_A5_ROUNDING_RESULTS.md) | Controlled finite-group result |
| Self-compiling retraction | [`SELF_COMPILING_RETRACTION_RESULTS.md`](../../Spin-Space-Research/docs/experiments/SELF_COMPILING_RETRACTION_RESULTS.md) | Compiler and optimization result |
| Latent Cayley retraction | [`LATENT_CAYLEY_RETRACTION_RESULTS.md`](../../Spin-Space-Research/docs/experiments/LATENT_CAYLEY_RETRACTION_RESULTS.md) | Shared-family recovery; inspect supplied-information contract |
| Partial Cayley completion | [`PARTIAL_CAYLEY_RETRACTION_RESULTS.md`](../../Spin-Space-Research/docs/experiments/PARTIAL_CAYLEY_RETRACTION_RESULTS.md) | Exact/empirical completion on stated masks |
| Inverse-cover identifiability | [`INVERSE_COVER_IDENTIFIABILITY_THEOREM.md`](../../Spin-Space-Research/docs/experiments/INVERSE_COVER_IDENTIFIABILITY_THEOREM.md) | Combinatorial theorem for its fixed setup |
| Endpoint curriculum | [`ENDPOINT_CURRICULUM_RESULTS.md`](../../Spin-Space-Research/docs/experiments/ENDPOINT_CURRICULUM_RESULTS.md) | Optimization result; not a uniqueness theorem |
| Zero-query compiler | [`ENDPOINT_MANIFOLD_COMPILER_RESULTS.md`](../../Spin-Space-Research/docs/experiments/ENDPOINT_MANIFOLD_COMPILER_RESULTS.md) | Latent-state structure recovery |
| Q8 state compilers | [`SPIN8_STATE_ONLY_COMPILER_RESULTS.md`](../../Spin-Space-Research/docs/experiments/SPIN8_STATE_ONLY_COMPILER_RESULTS.md) | State-only supervision contract |

## Nonclaims

- The mechanism is not uniquely geometric: direct structured controls can
  exhibit the same shared-family completion principle.
- Fixed-token operators make some held-out-bigram tests structurally easy; the
  meaningful question is whether optimization found a faithful action family.
- Oracle-best channel subsets are analysis tools, not deployable selectors.
- No result in this ledger establishes natural-language quality.

## Standalone paper

The cleanest paper is a general shared-family identifiability and optimization
study. Spin(8), A5, and Q8 should be examples rather than the title claim.
