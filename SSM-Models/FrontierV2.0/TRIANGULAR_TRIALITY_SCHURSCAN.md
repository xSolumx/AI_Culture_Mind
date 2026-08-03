# Triangular Triality SchurScan

## Core architecture

Use representation-theoretic state coordinates rather than treating Clifford
grades as unrelated channels. For each irreducible type `lambda`, factor state
as multiplicity space tensor representation space. The within-stream token
action is

```text
M_t,lambda tensor rho_lambda(g_t),
```

which composes exactly and gives the SchurScan monoid. Between streams, use the
Spin(8) triality tensor only along an acyclic dependency graph:

```text
S+ scan ----\
              triality bind ---> V drive ---> V scan
S- scan ----/
```

Every box is an affine associative scan. The binding between stages is
pointwise and activation-free. Parallel depth is a constant number of
`O(log N)` stages; recurrent cache is `8+8+8=24` scalars per channel.

## Why this is better than the original proposal

- It states exactly where nonlinearity lives.
- It does not pretend a bilinear feedback recurrence is affine.
- It preserves exact scan/streaming equivalence.
- It uses shared Spin(8) structure across `8s`, `8c`, and `8v`, unlike a single
  chiral stream that is merely another SO(8) chart.
- It exposes a formal no-go boundary: recurrent cycles cause unbounded degree
  unless additional closure exists.

## Evidence already passed

- Complete Cl(3) isotypic centralizer: dimension 8 versus old rank 4.
- SchurScan affine associativity: `1.11e-16` error.
- SchurScan length-17 parallel/recurrent parity: below `9e-16`.
- Triality lifted one-step closure: `8.88e-16`.
- Triality shared-action equivariance: `4.44e-15`.
- Practical staged scan parity: `4.27e-14`.

## Binding and memory gates now passed

The first masked cross-representation completion gate passes through length
512. A learned bilinear tensor rediscovers the supplied triality law, whereas
matched linear and MLP controls fail. The original training design was
full-rank, so a second preregistered gate deliberately reduced the bilinear
feature rank to 16 of 64. The infinitesimal equivariance system has a unique
invariant tensor direction; a one-parameter invariant model generalizes at
cosine 1.0 in 3/3 seeds, while unconstrained bilinear and MLP models fit the
observed cells below 1.2e-6 MSE and collapse off-support.

The multi-item capacity boundary is now explicit. One 8D triality vector is an
exact single-pair store, not a high-capacity VSA. Orthogonal multiplicity codes
give exact retrieval for (K\le H); for (K>H), unit-norm tight frames attain
the optimal average linear crosstalk bound ((K-H)/H). An eight-slot
token-selective overwrite recurrence then combines hard slot replacement,
shared Spin(8) transport, and an associative affine scan. Its length-128
parallel/recurrent and symbolic-oracle errors remain below 2.3e-15.

See the masked-completion, identifiability, and coded-memory result documents
under the experiments folder.

## Next empirical gate

Addresses, query keys, and Spin(8) actions are still supplied. The next honest
task must learn slot allocation and query routing under a frozen state budget,
then compare against:

- a same-width direct slot memory with no triality;
- diagonal/complex selective SSM memory;
- bilinear fast weights or linear attention;
- Householder/DeltaProduct transport;
- nonlinear cleanup outside the recurrent scan.

The exact rows should be used as mechanism oracles, not mixed into a claim
that attention has already been replaced.
