# Exact global five-probe triality certificate

**Date:** 2026-08-06  
**Status:** exact theorem for one displayed probe tuple  
**Preregistration:** none; this is a deterministic proof discovered while
auditing the earlier local-identifiability claim  
**Verifier:** `src/spin8_global_probe_certificate.py`  
**Artifact:** `artifacts/spin8_global_five_probe_certificate_20260806.json`

## Result in one sentence

One vector probe and four positive-spinor probes admit an explicit integral
configuration whose triality closure is a basis of all three eight-dimensional
representations, so its global Spin(8) stabilizer is trivial; deleting the
fourth spinor leaves an exact continuous stabilizer with Lie algebra
`su(2)`.

This closes the remote-discrete-ambiguity gap for the displayed five-probe
sensor. It does **not** yet classify every generic five-probe tuple or prove
that every possible four-probe allocation is insufficient.

**Subsequent result.** The later
[continuous probe orbit theorem](SPIN8_CONTINUOUS_PROBE_ORBIT_THEOREM.md) now
does prove universal four-probe insufficiency and an open dense globally free
stratum in every mixed five-probe allocation. The present document remains the
explicit global-closure certificate from which that stronger theorem grew.

## The invariant-closure lemma

Let `V`, `S+`, and `S-` be the three real eight-dimensional triality
representations, and let `rho_i` be the maintained octonionic triality tensor.
The three contractions are

\[
\begin{aligned}
\mu_-(v,p)&=\rho(v)p,\\
\mu_+(v,n)&=\rho(v)^Tn,\\
\mu_V(p,n)_i&=n^T\rho_i p.
\end{aligned}
\]

They are Spin(8)-equivariant. Therefore, if a shared action fixes `v`, `p`,
and `n`, it also fixes every defined product above. Repeating these products
gives the **triality closure** of the observed probes.

If that closure contains a basis of a representation, the action is the
identity on that representation. If it contains bases of all three, the
entire triality action family is the identity. Consequently, two shared
Spin(8) actions producing the same outputs on such a probe tuple must be the
same action globally, not merely in a neighborhood of the identity.

## Exact five-probe certificate

In the repository's fixed Fano convention choose

\[
(v;p_0,p_1,p_2,p_4)
=(e_0^V;e_0^+,e_1^+,e_2^+,e_4^+).
\]

All entries involved are integers in the maintained basis. Exact SymPy rank
arithmetic gives the following closure history, written as
`(rank V, rank S+, rank S-)`:

```text
(1,4,0)
(1,4,4)
(7,4,4)
(7,8,8)
(8,8,8)
```

The final basis matrices have exact determinants

```text
V:  -1
S+: -1
S-: -1
```

so none of the spanning claims depends on a numerical rank tolerance. The
corresponding exact half-integral generator constraint matrix has shape
`40 x 28`, rank `28`, and nullity `0`.

The closure argument is stronger than that differential check: the bases rule
out finite or remote stabilizer elements as well as infinitesimal ones.

## Exact four-probe counterfamily

Delete the final positive-spinor probe and retain

\[
(v;p_0,p_1,p_2)=(e_0^V;e_0^+,e_1^+,e_2^+).
\]

Its exact closure stops at `(4,4,4)`. The generator constraint matrix has
shape `32 x 28`, rank `25`, and nullity `3`. In bivector-plane notation, a
primitive basis of the annihilator is

\[
\begin{aligned}
X_1 &= E_{47}+E_{56},\\
X_2 &= E_{46}-E_{57},\\
X_3 &= E_{45}+E_{67}.
\end{aligned}
\]

Across all three triality representations the exact commutators are

\[
[X_1,X_2]=2X_3,\qquad
[X_1,X_3]=-2X_2,\qquad
[X_2,X_3]=2X_1.
\]

Each generator annihilates all four observed probes. Hence every
`exp(t X_i)` fixes those probes exactly for every real `t`. They are not inert:
on the withheld positive spinor `e_4^+` their velocities are respectively
`-e_7^+`, `-e_6^+`, and `-e_5^+`.

Thus the four-probe ambiguity is a genuine connected noncommutative
counterfamily, not optimizer failure or an SVD artifact.

## Exhaustive coordinate atlas: a Hamming-code surprise

The exact verifier also checks all `C(8,4)=70` coordinate choices for the four
positive-spinor probes while keeping the vector probe `e0^V` fixed.

- **56** choices generate the full closure `(8,8,8)`;
- **14** choices stop at `(4,4,4)`.

The exceptional supports are

```text
0123  0145  0167  0246  0257  0347  0356
1247  1256  1346  1357  2345  2367  4567
```

These 14 subsets have an exact classical structure. Add the empty support and
the full eight-coordinate support. The resulting 16 binary words:

- are closed under XOR;
- have binary dimension four;
- are self-orthogonal and doubly even;
- have weight enumerator `1 + 14 z^4 + z^8`;
- contain every coordinate triple in exactly one weight-four support.

Thus they are the extended binary Hamming `[8,4,4]` code, and the 14
exceptional supports form the Steiner quadruple system `S(3,4,8)`.

This is an exact exhaustive statement about the coordinate atlas, not a random
pattern match. It reveals a discrete combinatorial skeleton for the
four-dimensional triality closures. The connection is plausibly the coordinate
shadow of the classical octonion--Hamming--`E8` relationship, but the verifier
does not yet prove that the same code classifies the full continuous exceptional
orbit set.

## What has changed scientifically

The earlier result established:

- rank 28 for five mixed probes in generic numerical frames;
- rank 25 for four mixed probes;
- ten-seed local recovery and long-horizon completion;
- machine-precision four-probe witnesses.

The new result adds:

1. an exact globally identifying five-probe tuple;
2. a proof that no remote discrete ambiguity survives for that tuple;
3. exact integer spanning bases with determinant `-1`;
4. an exact three-generator `su(2)` four-probe counterfamily.
5. an exhaustive `56 + 14` coordinate atlas whose exceptional supports are
   exactly the extended Hamming `[8,4,4]` codewords of weight four.

## Claim boundary

Proved:

- the displayed `(1,4,0)` five-probe tuple globally identifies a shared
  Spin(8) triality action;
- its displayed four-probe subset has a continuous `su(2)` stabilizer;
- the 14 exceptional coordinate supports form `S(3,4,8)` and the weight-four
  shell of the extended Hamming `[8,4,4]` code;
- all calculations in the certificate use integral or rational arithmetic.

Not proved:

- that every generic five-probe tuple in every mixed allocation has trivial
  global stabilizer;
- that no specially chosen four-probe tuple in another allocation can have a
  smaller stabilizer;
- a complete orbit-type classification of the five-query sensor space;
- proof that every continuous four-dimensional exceptional closure is
  Spin(8)-equivalent to one of the 14 coordinate Hamming blocks;
- any language-model or throughput consequence.

The natural next theorem is the orbit-type classification: prove that the
explicit free tuple lies in the principal stratum and determine the complete
exceptional set, including the quaternionic four-dimensional closures visible
in the counterexample. The Hamming atlas supplies exact candidate normal forms
for that classification.

## Replay

```powershell
$env:PYTHONPATH='src'
python -m spin8_global_probe_certificate
python -m unittest discover -s tests -p "test_global_five_probe_certificate.py" -v
```

The verifier regenerates the closure, annihilator, commutators, and withheld
probe motions. It does not trust the artifact's stored `passed` field.
