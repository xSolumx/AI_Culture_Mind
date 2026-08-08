# v2.1 related-work interpretation: noncommutative probability

Date: 2026-08-06. Status: interpretive note, not part of the frozen
transport-ablation decision protocol.

## Useful direction

The maintained SSM is a narrow class of contractive rotor-affine processes,

```text
h_t = d_t Ad(q_t)(h_(t-1)) + b_t,
```

not a proposed theory of all geometric or noncommutative neural networks. For
this class, ordered action statistics, perturbation bounds, controllability,
observability, representation decomposition, scan closure, and streaming
sufficiency are coherent mathematical targets. This structural-toolkit framing
is more defensible than a universal geometric-algebra advantage claim.

Noncommutative probability suggests useful exploratory diagnostics beyond
validation loss. The cleanest immediate observable is the action commutator

```text
||U_(t+1) U_t - U_t U_(t+1)||,
```

which is identically zero for identity, diagonal, commuting-phase, and one
fixed-action transports, but may be nonzero for input-selective quaternion,
rotor, and SO(8) actions. Higher ordered matrix moments can then test whether
that noncommutativity carries predictive information. These diagnostics would
measure mechanism activity; causal necessity still requires the retrained and
post-training interventions in the v2.1 protocol.

## Corrections and assumptions

1. The current rotor SSM state is the full eight-coordinate
   `Cl(3,0)` algebra. `Spin(8)` and its chiral representations belong to a
   separate research branch. Writing the present state algebra as
   `Cl+(8)` conflates the projects and changes the dimension radically.
2. Scalar-part and normalized-matrix-trace functionals are tracial for two
   factors in the relevant finite-dimensional representations. Consequently,
   `phi(h_s h_t)` versus `phi(h_t h_s)` is not generally an order detector for
   those choices. Use a non-tracial state, a commutator acting on a probe, or
   at least three ordered factors.
3. A commuting transport family does not force cumulants of the complete
   learned state process to vanish. Nonlinear input-dependent writes, gates,
   and residual mixing can generate nontrivial moments in every family.
   Vanishing or separation claims require derivation under explicit input and
   controller assumptions.
4. Pointwise `0 < d_t < 1` proves the finite-horizon state bound but not a
   uniform contraction rate or stationary distribution. Stationarity and
   uniqueness results need assumptions such as stationary driving and
   `d_t <= rho < 1` (or an appropriate negative Lyapunov exponent).
5. The `V`, `S+`, and `S-` triality representations form separate
   representation spaces, not automatically tensor-product subsystems of one
   density operator. Partial traces among them are undefined until a specific
   tensor factorization and state construction are supplied.
6. Free-probability language needs an actual asymptotic regime, such as
   channel width or random-matrix dimension tending to infinity. Finite
   noncommutative probability is meaningful here; asymptotic freeness is not
   automatic.

## Narrow follow-up, after the frozen cohort

- record adjacent and longer-range action-commutator norms;
- compare third-order ordered traces under token order reversal;
- condition those statistics on prediction error and intervention damage;
- derive observability/controllability Gramians first for fixed uniformly
  contractive transports, then state the extra assumptions needed for
  selective nonlinear controls;
- keep any Spin(8) marginal-reconstruction or moduli-space programme separate
  from claims about the Cl(3,0) language model.

These are prospective theory and diagnostic directions. They do not alter or
retroactively strengthen the v2.1 empirical endpoints.
