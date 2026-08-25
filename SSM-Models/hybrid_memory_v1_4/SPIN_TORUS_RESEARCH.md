# Spin, torus, and content-addressed memory

**Decision date:** 2026-08-25
**Status:** exact fixed-frame transports implemented; the complete G15A chain
now supports a learned full 28-generator shared vector/positive chart on
held-out frames and global-center words under oracle edit timing. The fixed
Clifford second read remains unsupported and moving-frame geometry remains
research-only.

## Outcome

The Spin path is retained, but geometry is not being asked to invent the
memory law. The primary candidate is

\[
\boxed{\text{content address} + \text{independent erase/write}
+ \text{Spin transport} + \text{fixed Clifford read}}
\]

with periodic local attention retained by the hybrid shell. Each head stores
an $8\times8$ association matrix rather than the old 24-scalar transported
value cache. Identity, fixed commuting transport, and full Spin transport all
share the same state shape and edit law, so geometry can be falsified without
changing the basic retrieval mechanism.

[`spin_dirac_memory.py`](spin_dirac_memory.py) implements this candidate.
[`G15_SPIN_DIRAC_PREREGISTRATION.md`](G15_SPIN_DIRAC_PREREGISTRATION.md) and
its prospective
[`amendment`](G15_SPIN_DIRAC_AMENDMENT_2026-08-25.md) define the learning and
integrity gates; the prospective
[`edit-law amendment`](G15_SPIN_DIRAC_EDIT_LAW_AMENDMENT_2026-08-25.md)
records the scalar-gate covariance repair. [`G15_SPIN_DIRAC_RESULTS.md`](G15_SPIN_DIRAC_RESULTS.md)
separates contracts already exercised from experiments that have not run.

## Exact fixed-torus relationship

The implemented commuting transport is the maximal torus

\[
T^4=SO(2)_{01}\times SO(2)_{23}\times SO(2)_{45}\times SO(2)_{67}
\subset U(4)\subset SO(8).
\]

The octonionic almost-complex six-sphere satisfies

\[
S^6\cong G_2/SU(3),\qquad SU(3)\subset G_2\subset SO(7)\subset SO(8).
\]

In a compatible basis, the $SU(3)$ maximal torus is the rank-two slice

\[
(\theta_1,\theta_2,\theta_3,\theta_4)
=(\alpha,\beta,-\alpha-\beta,0).
\]

The `su3_torus` mode implements exactly these two constraints. The tests
verify zero total phase, a fixed fourth plane, and no active coordinates
outside the three stated planes. This is an (S^6)-tangent prior; it is not an
(S^6) state or a moving (G_2) frame.

Under this $SU(3)\subset Spin(8)$, the three triality carriers restrict as

\[
8_v,8_{s+},8_{s-}\downarrow SU(3)
\cong 1\oplus1\oplus3\oplus\bar 3.
\]

That common decomposition makes shared torus coordinates economical. It does
not multiply state capacity: the carriers are representation-coupled views of
the same action.

## Fixed frame versus the octonionic sphere

The standard (G_2)-invariant almost-complex structure varies with the point:

\[
J_x(v)=x\times v,\qquad x\in S^6,\;v\in T_xS^6.
\]

A point selects an $SU(3)$ stabilizer, not a unique maximal torus. A complete
moving construction therefore needs both the base point and flag/gauge state,
for example

\[
x_t\in S^6,\qquad f_t\in SU(3)/T^2,\qquad \theta_t\in T^2.
\]

No such state is implemented here. Calling the fixed `su3_torus` arm a moving
octonionic memory would be false. The standard structure is strict nearly
Kahler and nonintegrable; see
[Butruille](https://arxiv.org/abs/math/0612655) and
[Foscolo--Haskins](https://arxiv.org/abs/1501.07838).

## The actual state-space dual

For one head, the recurrence is

\[
M_t=L_tM_{t-1}R_t^\top+U_t,
\qquad U_t=w_t k_t v_t^\top.
\]

Vectorization gives

\[
\operatorname{vec}(M_t)
=(R_t\otimes L_t)\operatorname{vec}(M_{t-1})
+\operatorname{vec}(U_t).
\]

This is a 64-state linear time-varying SSM and a content-addressed fast-weight
memory. Unrolling produces a transported causal kernel, while the factor
triple `(L, R, U)` composes associatively for a two-sided prefix scan. These
are legitimate recurrent, scan, and kernel descriptions of the same system.
It is not a scalar 1-semiseparable Mamba-2 recurrence, and the representation
factorization is not a new independent "State-Space Triality" theorem.

## Why the primary edit gates are scalar per head

An earlier draft used channelwise diagonal retention and channelwise
Hadamard erase/write gates. Those operations pick a preferred basis and fail
the prospective inner-conjugation gate. The primary `equivariant_scalar` mode
therefore keeps erase, write, and retention independent but scalar per head:

\[
E_t=I-b_t k_tk_t^\top,\quad
L_t=E_t\,r_t\,\rho_v(g_t),\quad
R_t=\rho_{s+}(g_t),\quad
U_t=w_tk_tv_t^\top.
\]

Under a shared frame change $h\in Spin(8)$, the complete edit transforms as

\[
M\mapsto\rho_v(h)M\rho_{s+}(h)^\top.
\]

The exact float64 contract covers the transported state and both Clifford-
coupled reads. The richer `channelwise` gate mode remains available only as an
explicit non-equivariant ablation; it is ineligible for a symmetry claim.

## Transport ladder and promotion rules

1. `identity`: content-addressed edit without transport.
2. `commuting_so2`: four fixed phase planes, the cheapest complex baseline.
3. `su3_torus`: the constrained rank-two $SU(3)$ slice.
4. `spin8`: full factorized shared vector/half-spin transport.
5. `broken_spin8`: orthogonal marginal complexity with one carrier's lift
   deliberately broken; a conditional control for triality-specific claims.
6. A bank of conjugate tori: promising but not implemented until an oracle
   moving-frame task proves the need.
7. Moving (G_2/SU(3)) plus flag state: mathematically complete and presently
   unjustified for generic text.

Do not average group matrices, since a convex average generally leaves the
group. Future routing must use hard selection, ordered products, or a tangent
combination followed by an exact group map.

The decisive progression remains mechanism and delayed observability first,
then generic associative memory, then multi-seed natural text, then
parameter/compute/state matching. If full Spin transport cannot beat identity,
fixed torus, and broken-coupling controls on a task with an observable moving
frame, the exceptional-geometry route is rejected before expensive language
training.

G15A-F now supplies that observable-frame control without promoting the model.
Its four-probe tangent map has full rank, and learned S beats identity, fixed
torus, and broken coupling in every frozen row. S still misses every absolute
precision gate, so the result supports an identifiable shared vector/positive
transport mechanism but not a solved controller. Chart-error decomposition and
a separately frozen learning repair precede generic association or geometry.

G15A-R subsequently supplies that learning repair. The original global
rotation-covariant optimizer on random compositions, with a 600-step staged-LR
schedule, learned the chart to roughly `1e-7` error and passed fresh
I/C/S/S-broken confirmation. Because the longer fixed-LR control also passed,
decay is a precision improvement rather than a demonstrated necessity. The
supported advance is learned shared vector/positive transport under
multi-probe oracle frames and oracle edit timing.

G15A-S then expands the hidden dictionary from eight to all 28 planes and
evaluates it on probe banks disjoint from training. It passes all three fresh
seeds at roughly `1e-6` held-out-frame error and at most `2.36e-5` direct-
carrier error on unseen 2-pi/4-pi loops, volume-center words, other center
cosets, and loop-plus-primitive continuations. This closes the local spanning
and center-compatibility question for the hard-coded shared vector/positive
lift. G15B address/write/query learning now precedes any torus-bank or moving-
frame implementation.

## Claim ledger

**Exact in code/tests:** two-sided affine scan algebra; content addressing;
bounded value drive; contractive primary erase/retention; fixed $T^4$ and
constrained $SU(3)\;T^2$; shared triality actions; Clifford equivariance;
inner-conjugation covariance of the scalar edit law; explicit broken-coupling
control.

**Empirical but local:** supplied-coordinate G15A separation; G15A-R learned
primitive-coordinate composition; G15A-S learned all 28 signed directions and
passed held-out-frame/global-center transfer under oracle edit timing; native
runtime and finite contract probes.

**Open:** learned benefit from Spin transport, sparse conjugate tori, moving
(G_2) frames, generic language improvement, long-range factual recall, and a
matched scaling advantage.
