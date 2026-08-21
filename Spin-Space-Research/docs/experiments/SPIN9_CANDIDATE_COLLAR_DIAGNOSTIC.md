# Spin(9) near-candidate collar diagnostic

**Status:** exact localization; not a positivity theorem

**Code:** [`spin9_candidate_collar_diagnostic.py`](../../src/spin9_candidate_collar_diagnostic.py)

**Artifact:** [`spin9_candidate_collar_20260821.json`](../../artifacts/spin9_candidate_collar_20260821.json)

Artifact SHA-256:
`bd13bd2445dc35585aa53a474282c49565108cd47dd1e1f574c7ffa2566e1316`.

## Question

The promoted coupled `V1 + V5` graph theorem bounds the determinant ratio by
`21/20 = 1.05`. The algebraic symmetric candidate has ratio

\[
R_* = 1.04803989226218765573809160796\ldots.
\]

Can the existing global cube nearly reach the candidate before new local
coordinates are introduced?

## Exact diagnostic

The target is replaced by

\[
\frac{26201}{25000}=1.04804,
\]

which is only about `1.0774e-7` above `R_*`. For each sign of the scalar
`V1` coordinate, the existing degree-84 compact Bernstein tensor is built in
exact integer arithmetic and split cyclically in `(t,a,z)` to depth six.
Strict leaves are certified; every non-strict leaf is retained verbatim as an
unresolved dyadic box.

The positive and negative scalar chambers produce respectively `18` strict
leaves plus `15` handoff boxes, and `18` strict leaves plus `14` handoff boxes.
Thus 29 boxes remain unresolved; neither chamber is promoted as a global
`26201/25000` certificate.

The calculation is deliberately shallow. Its purpose is to distinguish a
globally easy rational margin from geometry that needs candidate-centered
charts. An unresolved box is neither a failed inequality nor a counterexample.

## Consequence

Generic global subdivision is not the economical path to the exact candidate.
The equality set already consists of four pure-`V1` graph preimages over
`Q(sqrt(241))`, while the local quotient Hessian is strictly negative modulo
Spin(9). The next proof object should therefore blow up the mixed variables
around that algebraic fiber and use the exact Hessian as the leading positive
form. After those local collars are certified, the existing compact atlas can
cover their complement.

This does not yet control the second supported `V5` or the unrestricted
quotient. Those remain later gates even if mixed candidate maximality on the
first graph slice is closed.
