# Spin(8) dynamic-slot recurrence preregistration

Date frozen: 2026-08-03, before implementation results.

## Motivation

Orthogonal multiplicity codes prove exact simultaneous retrieval for at most
(H) supplied slots, but a static sum is not an overwritable dynamic memory.
This gate adds token-selective replacement without placing a nonlinearity
inside the recurrent loop.

## Transition

Represent state in one fixed orthonormal multiplicity basis:

[
M_t[h]=r_t[h],V_tM_{t-1}[h]+B_t[h],
]

where (r_t[h]\in[0,1]), (V_t\in SO(8)) is the shared vector action, and
(B_t\in\mathbb R^{H\times8}). For a hard write to slot (q),
(r_t[q]=0), (B_t[q]=\operatorname{bind}(p_t,n_t)), while unaddressed slots
have retention one and zero drive.

Composition is

[
(r_2,V_2,B_2)\circ(r_1,V_1,B_1)=
(r_2r_1, V_2V_1, B_2+r_2\odot V_2B_1).
]

It is closed and associative because every retention is diagonal in the same
multiplicity basis.

## Test

- eight slots, length 128, float64;
- random shared Spin(8) transport at every position;
- random hard overwrite address and fresh unit chiral key/value per position;
- compare serial recurrence, logarithmic-depth associative scan, and a direct
  symbolic oracle that transports retained keys/values and replaces the
  addressed pair;
- query all final slots by exact triality unbinding.

## Frozen gates

- transition associativity below (10^{-10});
- full parallel/recurrent state parity below (10^{-10});
- recurrent state versus directly rebuilt oracle memory below (10^{-10});
- final retrieval of every overwritten value below (10^{-10}).

## Boundary

Passing establishes exact addressed writes, overwrites, shared geometric
transport, and constant-state streaming. Addresses and query keys remain
supplied. Learned content addressing, approximate over-capacity retrieval,
gradient behavior, and downstream utility require separate experiments.
