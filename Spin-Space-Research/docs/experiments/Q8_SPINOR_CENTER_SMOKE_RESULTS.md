# Q8 spinor center-fidelity smoke: results

Date completed: 2026-08-03. This is seed 0 only; it establishes a capacity and
implementation result, not the preregistered 10-seed reliability claim.

## Behavioral result

All rows use the frozen parity-complete endpoint curriculum, four channels,
no writes or decay, and the balanced `(w, w*i*i)` central-pair falsifier.

| Family | Action params | L15/16 pair result | Dense L15--L256 floor | Both-correct floor |
|---|---:|---:|---:|---:|
| Quaternion left spinor, 2pi | 48 | 100% | **100%** | **100%** |
| Cl(3) sandwich, 2pi | 48 | 47.8--51.6% | 0% | 0% |
| Cl(3) sandwich, 2.2 | 48 | 24.2--25.7% | 0% | 0% |
| Four-reflection O(4), shared | 256 | 100% | 0% | 0% |
| Two-reflection O(8) | 256 | 50% | 45.9% | 0% |

The capable Householder row proves why training-length accuracy is not the
gate: it fits L15 and L16 perfectly, then collapses to 0% at L31/L32. The
spinor remains 100% at every tested matched odd/even checkpoint through L256.

## Long gate

The frozen spinor checkpoint also scores 100% pair-member and 100% both-member
accuracy at base lengths 4095, 4096, 16383, and 16384. Central-pair state
separation remains `1.9895`--`1.9899`, close to the maximum sign-flip distance
of two. Recurrent full/chunk/token state parity is exact; maximum logit
difference is `2.86e-6`.

## Mechanism audit

The first aggregate report showed spinor linear-homomorphism RMS `0.633`, which
looked inconsistent with exact long behavior. The per-channel decomposition
resolves it:

| Channel | Active homomorphism RMS | Alone canonical accuracy | Decoder contrast energy |
|---:|---:|---:|---:|
| 0 | 0.00402 | 100% | 26.9% |
| 1 | 1.26607 | 12.5% | 9.5% |
| 2 | 0.00467 | 100% | 30.9% |
| 3 | 0.00350 | 100% | 32.8% |

Three independently parameterized channels discover near-exact faithful Q8
spinor actions; one channel is nuisance. The decoder places 90.5% of its
contrast energy on the three faithful channels, and removing any single
channel leaves canonical accuracy at 100%. The aggregate RMS was therefore a
bad summary, not evidence that the long pass was a non-homomorphic shortcut.

Direct quaternion inspection agrees: on channels 0/2/3, generator squares,
inverse antipodality, and the `ij+ji=0` relation have residual norms of order
`1e-3`--`1e-2`; the bad channel alone carries the large aggregate defect.

## Claim tier

This passes the preregistered **capacity** tier and the seed-0 optimization
smoke. It does not yet pass the 10/10 reliability tier. The equal-chart GA
failure isolates center-fidelity capacity from chart width; the
parameter-richer Householder failure at extrapolation is an optimization and
inductive-bias result at one seed, not a proof that generic O(4) cannot solve
Q8.

Artifacts:

- `q8_spinor_center_smoke_seed0.json`
- `q8_spinor_center_smoke_long_seed0.json`
- `q8_spinor_center_smoke_checkpoints/`
- `Q8_SPINOR_CENTER_PREREGISTRATION.md`
