# G15 Spin-Dirac prospective amendment

**Frozen:** 2026-08-25, after the historical-model audit and before any G15
training outcome was produced or inspected

This amendment strengthens
[`G15_SPIN_DIRAC_PREREGISTRATION.md`](G15_SPIN_DIRAC_PREREGISTRATION.md).
It does not change a result: no G15 training result exists at this freeze.
It supersedes the optional-arm language in the original frozen-ablation
section; every other original gate and nonclaim remains binding.

## Mandatory fourth primary arm

Identity transport with Clifford readout (`I+C`) is mandatory. The complete
within-family comparison is therefore:

| Arm | Transport | Second read sector |
|---|---|---|
| `I` | identity | identity copy |
| `I+C` | identity | fixed Clifford map |
| `C` | four commuting SO(2) planes | fixed Clifford map |
| `S` | full factorized Spin(8) | fixed Clifford map |

`I` versus `I+C` isolates the readout, `I+C` versus `C` isolates commuting
transport, and `I+C` versus `S` isolates full Spin transport. Query, key,
value, decay, erase, write, shell, active state, parameter tensors, optimizer,
tokens, and episode routing remain identical. The three-arm promotion rule in
the original preregistration is extended: `S` must also beat `I+C` by the
stated symmetry-task margin and remain inside the stated generic-task
non-inferiority margins.

## Additional integrity gates

Before G15A training:

1. verify all four Spin(8) central signatures through the vector, positive,
   and negative representations. Direct backend actions must match their
   expected `+I/-I` character to `1e-10` maximum absolute error in float64; a
   bounded-chart model test may reach the same elements by a multi-step product
   and must remain within `1e-8`;
2. rerun the symmetry task under frozen inner conjugations by
   `h in Spin(8)`: transform every carrier action as
   `rho_r(h) T_a rho_r(h)^-1`, or equivalently transform coordinates by
   `Ad_h`, while transforming inputs, outputs, and the Clifford tensor
   consistently. This is not an arbitrary `O(28)` coordinate rotation.
   Float64 predictions must agree within `1e-9` absolute and relative error in
   every seed. Outer `D4` triality permutations are a separate diagnostic;
3. verify that the default semantic transition is contractive: value drive is
   mapped smoothly into the unit ball, the symmetric addressed erase has
   operator norm at most one, transport is orthogonal, and retention has a
   strict configured upper bound below one;
4. run a 4,096-step bounded-input state-growth falsifier in float32 and the
   training dtype. Per head, record Frobenius norm, largest singular value,
   ratio to the analytic ceiling, and readout/RMSNorm statistics. For
   `M_t=L_t M_(t-1) R_t^T+U_t`, the binding ceiling is
   `r_max^T ||M_0||_F + (1-r_max^T)/(1-r_max)` because `||L_t||_2 <= r_max`,
   `||R_t||_2=1`, and `||U_t||_F<1`. No finite-conditioning claim follows from
   the loose default asymptotic ceiling of `1e6`;
5. compare mapped action-coordinate updates under the declared scalar-second-
   moment optimizer group and an SGD covariance diagnostic for eight steps in
   float64. Relative mapped-parameter and update discrepancies must each remain
   at or below `1e-10` in every seed. A chart-dependent AdamW effect cannot be
   counted as evidence for Spin structure;
6. verify causal observability at delayed scored positions in float64. With a
   `1e-4` normalized negative-gradient controller perturbation, both coordinate
   and query paths must change the later read by at least `1e-6` relative norm
   and reduce the scored target loss by at least `1e-8` in every seed. A
   same-token effect or merely nonzero gradient does not pass.

The unbounded-value mode may be retained only as an explicitly named ablation.
It is ineligible for the primary arms and carries no input-independent state
bound claim.

## Conditional coupling controls

If `S` passes G15A, two further parameter-matched diagnostics are required
before making a triality-specific claim:

- `S+identity-read`: full Spin transport with the identity-copy second read;
- `S-broken`: full marginal orthogonal action complexity, but a frozen
  non-automorphic signed permutation of the 28 coordinates is applied to one
  carrier, breaking the shared triality lift.

If `S` does not beat `S-broken`, the strongest allowed interpretation is a
benefit from richer two-sided orthogonal transport, not triality coupling.

## Name and claim boundary

`SpinDirac` is shorthand for Spin transport plus a fixed Clifford map. It is
not a geometric Dirac differential operator and carries no spectrum, zero-mode,
or index-theorem claim. The positive half-spin representation has image
`SO(8)` and is blind to one central `Z2`; observing one carrier's orthogonal
action therefore does not establish the coupled Spin(8) lift or triality
structure. Any positive result must survive the coupled-representation,
central-sign, fixed-readout, and inner-conjugation controls above.
