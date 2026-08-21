# Spin(8) ordered-probe stabilizer tower

**Status:** exact defining-vector theorem and cross-artifact stabilizer atlas

**Code:** [`spin8_probe_stabilizer_tower.py`](../../src/spin8_probe_stabilizer_tower.py)

**Artifact:** [`spin8_probe_stabilizer_tower_20260821.json`](../../artifacts/spin8_probe_stabilizer_tower_20260821.json)

## Exact vector-frame tower

Fixing the images of `m` ordered orthonormal vectors under `SO(8)` leaves the
orthogonal-complement action `SO(8-m)`. Hence

\[
V_{8,m}=SO(8)/SO(8-m),\qquad
\dim V_{8,m}=28-\binom{8-m}{2}=\frac{m(15-m)}2.
\]

The exact generator-action ranks replay as

\[
0,7,13,18,22,25,27,28,28.
\]

The proposed odd refinement is therefore

\[
V_{8,3}\longrightarrow V_{8,5}\longrightarrow V_{8,7},
\]

with cumulative ranks `18,25,28`, residual gauges
`SO(5),SO(3),SO(1)`, and coordinate split

\[
\boxed{28=18+7+3}.
\]

The reverse path is exact: forgetting probes 6 and 7 maps rank 28 to rank 25;
forgetting probes 4 and 5 maps rank 25 to rank 18. These are quotient maps,
not approximate compression. They suggest a coarse/refine/coarsen controller
whose 7- and 3-coordinate fibers need only be retained when the corresponding
probe resolution is available.

The complete odd residual dimensions after `m=1,3,5,7` probes are

\[
21,10,3,0,
\]

explaining the `7 observed / 21 gauge` first stage: one vector probe has a
21-dimensional `SO(7)` stabilizer.

## Why the `SU(n)` ladder is separate

Stabilizers depend on the sensor representation. The maintained exact Spin(8)
coordinate-geometry artifact contains a mixed-triality chain

\[
SU(3)\longrightarrow SU(2)\longrightarrow 1,
\]

with Lie dimensions `8,3,0`. The Spin(9) Clifford gate records spinor-orbit
ranks `15,28,36`, hence stabilizer dimensions `21,8,0`, interpreted as

\[
\mathrm{Spin}(7)\longrightarrow SU(3)\longrightarrow 1.
\]

Those are not consequences of the defining-vector `SO(8-m)` formula. The new
artifact hashes and checks both source certificates precisely to prevent the
ladders from being conflated.

## Compiler interpretation and boundary

The `18+7+3` split is a gauge-resolution chart, not an isotypic decomposition.
It can organize adaptive controller work, but it does not itself construct a
global continuous frame selector or imply a machine-learning advantage. Chart
transitions and lift consistency remain necessary at singular boundaries.
