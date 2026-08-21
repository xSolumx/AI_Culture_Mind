# Spin(9) exact-candidate quadratic atlas

**Status:** exact ordered-field localization; compact complement open

**Code:** [`spin9_candidate_quadratic_atlas.py`](../../src/spin9_candidate_quadratic_atlas.py)

**Artifact:** [`spin9_candidate_quadratic_atlas_20260821.json`](../../artifacts/spin9_candidate_quadratic_atlas_20260821.json)

**SHA-256:** `26be9b1ccd8ad647286ff3baaba0ba1a282b1feddf77de1df55710ad5c2869d0`

## Exact scalar extension

Write the algebraic candidate as

\[
R_* = \frac{A+B\sqrt{241}}{D}.
\]

The maintained rational compact tensors for
`C1=21 Delta-20 N` and `C2=2 Delta-N` reconstruct

\[
\Delta=\frac{20C_2-C_1}{19},\qquad N=2\Delta-C_2.
\]

This gives every Bernstein control of
`(A+B sqrt(241)) Delta-D N` exactly. For each control `a+b sqrt(241)`, a
180-digit rational enclosure of the positive square root is selected according
to the sign of `b`. The resulting rational control is below the exact
quadratic-field control. Bernstein subdivision preserves that coefficientwise
order, so every strict leaf proves positivity at the exact irrational target.

## Depth-six localization

The two scalar-sign chambers contain 36 strict leaves and 29 unresolved
handoff boxes in total. These counts exactly match the earlier depth-six atlas
at the rational target `26201/25000`, even though the new leaves certify
`R_*` itself. Therefore the unresolved geometry is not an artifact of replacing
the candidate by a nearby rational number.

This is the useful negative result: blind deeper subdivision retains the
algebraic equality edge and cannot close the theorem economically. The next
certificate must excise the four explicit collars and introduce a cusp or
blow-up chart coupling distance from the equality quartic to the `V5` radius.

The artifact does not certify the compact complement, the second `V5`, or the
unrestricted quotient.
