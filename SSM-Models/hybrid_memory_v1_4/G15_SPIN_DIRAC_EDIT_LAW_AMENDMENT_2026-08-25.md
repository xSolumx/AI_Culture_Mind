# G15 Spin-Dirac edit-law amendment

**Frozen:** 2026-08-25, after the inner-conjugation implementation audit and
before any G15 training outcome was produced or inspected

This is a second prospective amendment to
[`G15_SPIN_DIRAC_PREREGISTRATION.md`](G15_SPIN_DIRAC_PREREGISTRATION.md). The
first [`integrity/control amendment`](G15_SPIN_DIRAC_AMENDMENT_2026-08-25.md)
remains binding.

## Exposed defect

The first implementation draft used channelwise diagonal retention and
channelwise Hadamard erase/write gates. Although that law is a valid GDN2-like
fast-weight update in a fixed basis, it is not covariant under a general shared
Spin(8) frame change. A diagonal operator does not remain diagonal after
conjugation, and elementwise gates do not transform as carrier scalars.

That defect would make the prospective inner-conjugation gate impossible for
reasons unrelated to learned transport. It is corrected before training rather
than hidden as an optimization failure.

## Binding primary law

All primary I, I+C, C, and S arms use `gate_mode="equivariant_scalar"`.
Erase, write, and retention remain independent learned controls, but each is a
scalar per head and token:

\[
E_t=I-b_tk_tk_t^\top,
\quad L_t=E_tr_t\rho_v(g_t),
\quad R_t=\rho_{s+}(g_t),
\quad U_t=w_tk_tv_t^\top.
\]

The address key transforms in $8_v$, the value in $8_{s+}$, and the state
as $M\mapsto\rho_v(h)M\rho_{s+}(h)^\top$. This makes the complete edit and
both fixed-Clifford reads covariant under a shared inner frame change.

All four primary arms retain identical state shape, control tensor shapes,
trainable parameter tensors, shell, optimizer groups, and data. The change
does not privilege S over the other within-family arms.

## Required ablation

`gate_mode="channelwise"` remains implemented as the richer basis-dependent
GDN2-like ablation. It is ineligible for a Spin-equivariance or triality claim.
If it is trained, its result must be labelled `channelwise-non-equivariant` and
reported separately from the primary promotion table.

The external GDN2 control retains the maintained channelwise GDN2 law. It is a
memory-law comparator, not an equivariant Spin arm, and remains subject to the
original parameter/compute and native-backend eligibility rules.

## Claim boundary

This amendment establishes a mathematically coherent experiment design, not a
quality result. Exact memory-law conjugation is necessary but not sufficient:
the trained symmetry-task conjugation replay, delayed observability, generic
memory, natural text, and scaling gates remain binding.
