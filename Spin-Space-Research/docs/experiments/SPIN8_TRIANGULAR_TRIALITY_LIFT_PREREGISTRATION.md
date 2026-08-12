# Spin(8) triangular triality lift preregistration

Date frozen: 2026-08-03, before execution.

## Question

Can the bilinear triality binding `S+ x S- -> V` be used between recurrent
states without forfeiting an exact associative training scan?

## Frozen distinction

An unrestricted bilinear recurrent update is not affine and does not compose
inside the current fixed-dimensional affine tuple. There is, however, an exact
finite lift when the two spinor streams are triangular: they evolve
independently by affine maps, and their binding does not feed back into either
source stream.

Carry the homogeneous lifted state

```text
y = [1, s+, s-, vec(s+ tensor s-)]          # dimension 81
```

for affine source updates

```text
s+' = P s+ + p
s-' = N s- + n.
```

Then

```text
s+' tensor s-'
 = (P tensor N)(s+ tensor s-)
 + (P s+) tensor n
 + p tensor (N s-)
 + p tensor n,
```

which is linear in the lifted state. Ordinary matrix multiplication therefore
gives an associative prefix scan. The fixed triality tensor reads the vector
binding from the 64D tensor block.

## Frozen gates

1. One lifted step must match direct affine spinor updates and their outer
   product to float64 `1e-12`.
2. Products of lifted matrices must match recurrent execution at every prefix
   of a non-power-of-two length 17 to `1e-11`.
3. The triality readout from the lifted tensor block must match direct
   `s-^T rho_i s+` binding to `1e-12`.
4. Under one shared random Spin(8) action, bound vectors must transform by the
   vector representation to `1e-11`.
5. A degree audit must explicitly distinguish the triangular closure (degree
   at most 2) from feedback into both spinor streams (unbounded degree growth).
6. Passing proves scan-compatible triangular binding only. It does not prove
   attention replacement, associative recall, or language quality.
