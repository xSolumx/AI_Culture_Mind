# Spin-Delta query gradient-topology results

**Decision:** the predicted hard-gate dead path is reproduced, both isolated
repairs restore query-slot credit, and soft query-event continuation is the
frozen lower-perturbation repair for the next training gate.

## Exact local mechanism

For routed slot vector \(s(z_s)\), internal query \(i\), and event value
\(e(z_e)\), the core receives

\[
q=e s+(1-e)i.
\]

Consequently,

\[
\frac{\partial L}{\partial z_s}
=e\,J_s^{\mathsf T}\frac{\partial L}{\partial q}.
\]

The current straight-through binary event has hard forward value \(e=0\) at
its initialized logit \(-3\). Its backward surrogate can train the event
logit, but the multiplicative forward zero still kills the routed slot
gradient exactly. The float64 local probe measured:

| Path | Event value | Event-logit gradient | Slot-gradient norm |
| --- | ---: | ---: | ---: |
| Hard fallback | 0 | -5.491e-3 | **0** |
| Soft event | 0.0474259 | -3.301e-3 | 1.725e-3 |
| Authoritative query | 1 | 0 | 4.484e-1 |

This is a structural gradient identity for the audited query equation, not a
claim about all hard routers.

## Full-model result

Three fresh, bitwise-paired two-layer Spin-Delta initializations were probed on
the raw-CUDA recurrence with batch 128 and two writes. No optimizer update was
taken.

| Seed | Hard slot grad | Hard event grad | Soft slot grad | Soft logit change | Authoritative slot grad | Authoritative logit change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 691 | 0 | 2.646e-6 | 1.395e-6 | 8.023e-5 | 1.871 | 0.3270 |
| 701 | 0 | 1.878e-6 | 1.646e-6 | 7.477e-5 | 0.6816 | 0.2731 |
| 709 | 0 | 1.944e-6 | 2.337e-6 | 9.260e-5 | 1.605 | 0.2630 |

Every hard final query event was zero. Every hard query-slot gradient was
exactly zero while every hard query-event gradient exceeded the frozen
\(10^{-8}\) threshold. Both repair paths restored finite nonzero slot
gradients in every seed.

Soft continuation changed the initial logits by only
`7.477e-5`--`9.260e-5`; immediate authority changed them by
`0.2630`--`0.3270`. The soft intervention was therefore between `2,840x` and
`4,076x` smaller in every paired seed. It passes the prospective selector.

## Interpretation

The label-free failure is now localized more sharply. It is not merely that a
hard decision is difficult to optimize. At initialization, the present query
factorization gives the event head a first-order learning signal while giving
the conditional slot head none. Retrieval can then improve through the
internal-query fallback without ever activating explicit query-address
learning.

Forcing the routed query to be authoritative also repairs credit assignment,
but it abruptly replaces the core's initialized internal query. Soft event
continuation restores the missing derivative while remaining close to the
current function, so it is the cleaner first intervention.

## Frozen next gate

The next experiment must change only the query-event training path:

1. begin with the sigmoid query-event probability in the forward pass;
2. anneal a prospectively fixed temperature or soft/hard mixture toward the
   existing straight-through hard event;
3. retain hard straight-through categorical slots, the internal-query
   fallback, write controls, core, curriculum, token budget, and evaluation;
4. compare against the current hard-event curriculum on paired initialization
   and minibatch-order seeds;
5. require both retrieval robustness and gauge-correct query-event/slot
   identification, rather than treating either alone as success.

The schedule and decision thresholds must be committed before training. No
authoritative-query change or auxiliary router labels may be mixed into this
first repair gate.

## Boundaries and replay

This audit proves a local derivative identity and records finite full-model
initialization probes. It does **not** establish that soft continuation trains
a better model, survives annealing, generalizes to natural data, or improves
throughput.

Canonical artifacts are in
[`artifacts/spin_delta_query_gradient_topology/`](artifacts/spin_delta_query_gradient_topology/).
The raw artifact SHA-256 is
`cc012cc278179d0b21367aeddf684893461acd3b93166e11908a7869b99c5c23`;
the deterministic summary SHA-256 is
`6326679336bf10fdc6a9567deed1306fa21491f49231136bd356931acb31d707`.
The summarizer reproduced the summary byte-for-byte under WSL2, PyTorch
`2.10.0+cu126`, CUDA 12.6, and the RTX 2070 SUPER.
