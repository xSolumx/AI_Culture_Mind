# Q8 endpoint mixing and parity audit

Date completed: 2026-08-03, before learned Q8 center-fidelity training.

## Structural result

The proposed alphabet `{i,-i,j,-j}` does not define an aperiodic walk over all
eight Q8 elements. Word parity is a strict coset invariant:

- odd lengths reach exactly `{i,-i,j,-j}`;
- even lengths reach exactly `{1,-1,k,-k}`.

Accordingly, the 32-state `(group element, previous token)` Markov kernel has
largest and second-largest eigenvalue modulus both equal to one. Fixed-length
endpoint entropy approaches two bits, not `log2(8)=3` bits. Any even-only
L16/L64/L4096 evaluation would test only half the group while appearing long.

This is why the final curriculum and gate use matched odd/even lengths. It is
a coverage correction, not a tuned optimization trick.

## Exact position information

| Length | Mean `I(token_position; endpoint)` |
|---:|---:|
| 1 | 2.000 bits |
| 2 | 0.07975 bits |
| 4 | 0.000381 bits |
| 8 | `3.08e-8` bits |
| 16 | `9.54e-16` bits |
| 32 | `8.42e-16` bits |

Individual-token endpoint information mixes away even though the chain retains
the global parity class.

## Gradient qualification

Unlike the A5 identity audit, the complete Q8 action gradient remains highly
coherent at long length. That headline is partly common-mode, so the audit also
projects away the mean gradient shared by all four token actions. The spinor
token-contrast signal/RMS ratios at L1/L2/L4/L8/L16/L32 are
`0.995/0.787/0.459/0.681/0.791/0.877`; the sandwich values are
`0.995/0.772/0.382/0.582/0.698/0.814`. The 2π and 2.2 sandwich charts have the
same normalized ratios at identity and differ only in gradient scale.

Therefore the Q8 curriculum is not justified by claiming the same monotone
gradient-cancellation mechanism as A5. Its rigorous motivations are:

1. expose both parity cosets;
2. begin with exact high-information token actions;
3. increase composition depth while retaining adjacent odd/even lengths.

The learned experiment must decide whether that is sufficient. The balanced
central-pair falsifier remains the capacity test.

Artifact: `q8_endpoint_credit_audit.json`.
