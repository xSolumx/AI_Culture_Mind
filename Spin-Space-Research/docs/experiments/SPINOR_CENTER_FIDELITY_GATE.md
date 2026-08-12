# Spinor center-fidelity gate

Date: 2026-08-03.

## Lineage and novelty boundary

The kernel obstruction is not newly discovered here. The project review had
already stated that a Cl(3) sandwich cannot faithfully represent Q8 because
`R` and `-R` act identically. The new contribution in this cycle is narrower:
the caveat is now an executable certificate, integrated into the actual
write-free model parameterization, and paired with a constructive,
parameter-matched left-spinor alternative and preregistered falsifier.

The original 1,000-step selective Q8 ladder already contained a left-action
family under the name `quaternion_even`. Its L16 final accuracies were
`51.15%` for GA sandwich and `48.24%` for quaternion-even; at L32 they were
`36.79%` and `11.18%`. Thus the old experiment did **not** demonstrate a
practical spinor advantage. Decay, additive writes, two learned quaternion
subactions, and optimization confound the kernel question. This is why the new
gate compares `pure_quaternion_spinor` and `pure_ga_rotor` with writes and decay
removed, rather than relabeling the old quaternion row as a new result.

## The obstruction

The sandwich action used by a geometric-algebra rotor recurrence is

```text
h -> R h reverse(R).
```

Its kernel contains the center of the spin group: `R` and `-R` act identically.
On vectors this is exactly the double cover

```text
Spin(n) -> SO(n),    kernel {+1,-1}.
```

Conjugating a full Clifford multivector does not repair the loss: central
`-1` commutes with every grade and still cancels in the sandwich. Therefore a
pure write-free sandwich recurrence cannot distinguish two words whose rotor
products differ only by central sign, no matter how many conjugation channels
or how powerful the decoder is.

## Q8 is the minimal falsifier

The quaternion group is a subgroup of unit quaternions `Spin(3)`:

```text
Q8 = {+1,-1,+i,-i,+j,-j,+k,-k}.
```

Under conjugation, `q` and `-q` give the same 3D rotation. The image has only
four elements and is isomorphic to `Q8/{+1,-1} = V4`. In particular,
`i^2=-1` is indistinguishable from identity under sandwich composition.

Under left spinor multiplication,

```text
s -> q s,
```

the central `-1` acts as `-I`, so all eight elements remain distinct on a
nonzero quaternionic spinor orbit. The CPU certificate records:

| Action | Distinct matrices | Distinct orbit states | Effect of `i^2` vs identity |
|---|---:|---:|---:|
| Quaternion left/spinor | 8 | 8 | state distance 2 |
| Rotor sandwich/vector | 4 | 4 | state distance 0 |

This is an architectural expressivity theorem, not a training result.

## Next experiment

Use endpoint-only Q8 products to compare write-free actions, with the spinor
and GA rows parameter matched and all other counts reported explicitly:

1. Cl(3) rotor sandwich state—preregistered impossible ceiling of four
   distinguishable central-sign classes;
2. quaternion/Spin(3) left-spinor state—faithful 4-real-dimensional action;
3. four-reflection O(4) action shared across two state blocks—generic capable
   faithful baseline;
4. the old two-reflection O(8) action—equal raw Householder parameter count but
   incapable control, since a faithful Q8 generator has `rank(I-A)=4`;
5. exact regular permutation action—discrete ceiling.

Require held-out words that differ only by the central `-1`, matched odd/even
dense lengths, exact streaming, and the Q8-specific parity/mixing audit. The
alphabet `{+-i,+-j}` is bipartite, so an even-only long gate would cover only
half the group.
If the spinor arm succeeds where sandwich provably cannot, that is direct
evidence for the state representation—not merely for noncommutativity.

For Spin(8), the same distinction is fundamental. Vector and two chiral-spinor
representations are permuted by triality, but their center kernels differ. A
triality SSM should therefore couple vector and chiral-spinor states explicitly
rather than assuming rotor sandwiching on a multivector is equivalent to a
spinor action.

Artifact: `spinor_center_fidelity_audit.json`.
