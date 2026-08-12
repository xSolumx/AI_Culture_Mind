# Spin(8) Q8 joint orbit-family retraction result

Date: 2026-08-03

This is a seed-0 mechanistic smoke result. It passed the prospective protocol
in `SPIN8_Q8_JOINT_RETRACTION_PREREGISTRATION.md`; it is not yet a multi-seed
reliability claim.

## Unconstrained tangent training

`pure_spin8_positive` was added to the write-free eight-real-state harness.
Every token/channel learned an unconstrained 28-coordinate tangent update,
exponentiated through the fixed positive-chiral generators. The existing
parity-complete 2,000-step Q8 curriculum ran on the local RTX 2070 SUPER in
`28.66` seconds.

The raw model learned a real but imperfect mechanism:

- 100% central-pair member and joint accuracy through base length 32;
- 100% decoded one-step Cayley-edge accuracy;
- strong noncommutativity (`1.1536` commutator separation);
- central-state separation remained nonzero;
- but raw full homomorphism RMS was `0.4767` and orbit RMS was `0.2066`;
- accuracy collapsed to 0% at base lengths 127 and 128.

This is accumulated representation drift, not a capacity failure.

## Joint family retraction

The canonical orbit exposed the mechanism directly. In the three useful
channels, four singular values were near `1.36--1.45`; the remaining four were
at most `0.159` and often much smaller. The learned eight-real state had formed
an approximate faithful four-real quaternionic orbit inside the chiral space,
with an underconstrained complement.

The retraction therefore did not minimize full-space matrix error. It:

1. formed all four central-pair differences from all eight canonical states;
2. recovered one shared `8 x 4` frame by rectangular polar decomposition;
3. induced every token action jointly from the exact Q8 multiplication law on
   that frame;
4. fixed the orthogonal complement;
5. mapped the complete exact family through the positive-spinor Lie basis;
6. changed the initial orbit state but left the trained decoder frozen.

No token was normalized or rounded independently.

## Prospective gate result

| Metric | Raw | Jointly retracted |
|---|---:|---:|
| Full homomorphism RMS | `0.476714` | `4.286e-7` |
| Orbit homomorphism RMS | `0.206601` | `4.529e-7` |
| Decoded Cayley-edge accuracy | `100%` | `100%` |
| Reachable orbit rank | not exact | `4` |
| L127 joint central-pair accuracy | `0%` | `100%` |
| L128 joint central-pair accuracy | `0%` | `100%` |
| Dense L15--L256 minimum | `0%` | `100%` |
| Spin(8) exponential reconstruction maximum | -- | `5.283e-7` |
| Streaming-state residual | `0` | `0` |
| Streaming-logit residual | `2.861e-6` | `2.861e-6` |

The frame projection was small in the three useful channels (`0.0193--0.0277`
RMS) and large in the previously diagnosed nuisance channel (`0.4666`). The
decoder nevertheless remained perfect, matching the earlier conclusion that
the fourth channel was not required for the learned Q8 code.

## Long-horizon result

The untouched long audit achieved 100% pair-member and 100% joint accuracy at
base lengths `4,095`, `4,096`, `16,383`, and `16,384`. These lengths are far
beyond the L15/L16 final curriculum stage. The exact family projection removes
the drift instead of merely moving its first visible failure.

## What is new

The result joins three ideas into one working mechanism:

- unconstrained tangent optimization finds a useful low-dimensional orbit;
- the learned orbit reveals its minimal faithful representation dimension;
- one global algebraic retraction converts the complete token family into an
  exact center-faithful representation without retraining or decoder repair.

This is stronger than independent matrix rounding and more precise than
full-space nearest-representation fitting: it identifies and retracts the
observable minimal realization while explicitly fixing unused state slack.

## Next gate

Freeze the retraction and active-rank rule, then run fresh seeds without
changing thresholds. Compare raw and retracted reliability against the
parameter-matched quaternion-spinor and capable Householder baselines. Only
after that cohort should a triality-coupled write experiment begin.
