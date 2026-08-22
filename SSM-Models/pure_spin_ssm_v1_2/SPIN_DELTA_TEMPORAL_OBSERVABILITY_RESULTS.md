# Spin-Delta temporal observability results

**Decision:** the one-block structural identity, stack-induced observability,
and hard-slot dead-path decisions passed. Grammar-aligned credit failed. The
frozen final-path magnitude threshold also failed in the weakest seed.

## Structural result

In one Spin-Delta block, the state scan is computed before the query read. The
query changes the block output but not that block's transition. With loss only
at the final position, every non-final query-event and query-slot derivative
was exactly zero in all three paths and all three seeds.

This matches the implemented dependency graph:

\[
s_{1:T}=\operatorname{scan}(A_{1:T},b_{1:T}),\qquad
y_t=\operatorname{read}(s_t,q_t),\qquad L=L(y_T).
\]

For one block, \(q_t\) has no path to \(L\) when \(t<T\), hence
\(\partial L/\partial q_t=0\).

With two blocks, the first block's earlier read enters the second block through
the residual stream. The indirect path

\[
q_t^{(1)}\longrightarrow y_t^{(1)}\longrightarrow
s_T^{(2)}\longrightarrow L
\]

made non-final soft-event and slot derivatives nonzero in every seed.

| Seed | Non-final event max | Non-final slot max | Final event mean | Final slot mean |
| ---: | ---: | ---: | ---: | ---: |
| 761 | 2.234e-8 | 8.199e-9 | 7.271e-8 | 2.749e-8 |
| 769 | 2.637e-8 | 1.041e-8 | 2.315e-8 | 1.465e-8 |
| 773 | 1.498e-8 | 1.625e-8 | 2.495e-8 | 2.335e-8 |

These are two-block soft-query-event values. Stack-induced observability passed
its frozen `1e-10` threshold.

## Observability is not identification

For an event target, gradient descent is grammar-aligned when it decreases
non-final event logits and increases the final event logit. For the final slot,
it is aligned when it increases the correct-minus-wrong logit margin.

| Seed | Non-final event-off aligned mass | Final event-on aligned mass | Correct-slot descent margin |
| ---: | ---: | ---: | ---: |
| 761 | 47.25% | 38.90% | -1.376e-9 |
| 769 | 44.23% | 41.53% | +2.862e-9 |
| 773 | 48.22% | 52.94% | -3.630e-9 |

None approached the frozen 90% alignment requirement, and the slot direction
had the correct sign in only one seed. Depth therefore creates a nonzero
Jacobian without creating a reliable semantic teaching signal. This explains
how continuation can move query-slot parameters while failing to recover the
controller or improve retrieval robustly.

## Hard dead path and magnitude boundary

All initialized hard query events were forward-zero. Consequently, every
query-slot gradient was exactly zero at every position in both one- and
two-block models. The hard-slot dead-path decision passed.

The qualitative final-path topology was present, but its prospective magnitude
decision failed: one seed's one-block hard event gradient was only `1.328e-9`,
below `1e-8`, and one soft final-slot value was `9.006e-9`. This is reported as
a failed frozen gate rather than relaxing the threshold after measurement.

Immediate authoritative routing produced much larger slot gradients and made
the two-block indirect path clear, but it removes the event head entirely and
was already rejected as the higher-perturbation intervention. It is not
silently promoted here.

## Architectural consequence

The present task supervises a final answer, not a six-field tokenwise grammar.
Requiring that grammar to emerge as a uniquely named latent factorization is
not justified by the loss. More event annealing cannot fix a gradient whose
direction is not aligned with the desired event semantics.

The lowest-complexity next architecture is therefore an **event-free query
address**:

1. keep selective write events and the Spin-Delta state transition;
2. make the query address a continuously live simplex distribution at every
   model output position;
3. remove the binary query-event/internal-query switch;
4. train the address only through the ordinary output loss;
5. assess retrieval and address gauge, but do not demand recovery of an
   unsupervised binary event label that the task does not identify.

For language modeling, every position already requests an output, so an
always-live memory read is the natural contract. The binary event belongs on
selective state mutation, not necessarily on read access. This is an
architecture proposal supported by the audit, not yet a quality result.

## Boundaries and replay

The exact zero is a dependency result for the implemented one-block model with
final-only loss. The two-block values are finite initialization measurements,
not a theorem for arbitrary stacks or trained parameters.

Canonical artifacts are in
[`artifacts/spin_delta_temporal_observability/`](artifacts/spin_delta_temporal_observability/).
The raw SHA-256 is
`57b7b7e78c3bf0b2b2f0cd3194ae1bbe5e9e37fe8eb4bd7298e1ac36d89fb6a4`;
the deterministic summary SHA-256 is
`4ef3ed6233daac75baeb4a6b9b3b5f3498876a4d8e9bcd187b613cd7ad06d137`.
The summary replayed byte-for-byte under WSL2, PyTorch `2.10.0+cu126`, CUDA
12.6, and the RTX 2070 SUPER.
