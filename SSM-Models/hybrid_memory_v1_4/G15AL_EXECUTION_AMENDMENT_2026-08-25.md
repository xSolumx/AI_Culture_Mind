# G15A-L exact event-sparse execution amendment

**Frozen:** 2026-08-25, after the non-evidentiary smoke exposed dense identity-
filler cost and before any quality execution or quality metric

This amendment changes no seed, task example, token mapping, action, optimizer,
budget, metric, or threshold in
[`G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md`](G15AL_LEARNED_COORDINATE_PROTOCOL_2026-08-25.md).
It replaces redundant dense evaluation of identity filler with the exact
closed form of the same recurrence.

After the initial write, every filler position has zero erase/write and
identity transport. Retention is the same scalar `r=0.999999` at every
position. If the ordered nonidentity event products are `V` and `P` in the
vector and positive carriers, then at the final position

\[
M_{L-1}=r^{L-1}Vkv^TP^T.
\]

For the oracle transported query `q=V_*k`, the exact predicted positive read
is

\[
r^{L-1}\langle q,Vk\rangle Pv,
\]

and the S-teacher target is `r^(L-1) P_* v`. The event-sparse runner therefore
evaluates the actual SpinDirac action backend only at the two to sixteen
nonidentity action positions, composes those matrices in causal order, and
applies the analytic filler scalar. It is algebraically identical to the dense
recurrence, not a compatibility approximation or surrogate kernel.

Before quality, tests must compare event-sparse and dense `forward_controls`
reads on deterministic episodes in float64 to maximum absolute error at most
`1e-10`. Quality artifacts record both protocol hashes and the execution path.

The smoke result remains non-evidentiary. Its metrics do not alter the frozen
quality adjudication.
