# Householder Holonomy Transfer Pilot

The GA multi-scale holonomy objective was applied unchanged to Householder
actions. Householder uses its full 8D channel in the loss; no Cl(3)-specific
active-grade restriction is reused.

| seed | role | previous L16/L64/L128 | with holonomy L16/L64/L128 | dense minimum | minimum margin |
|---:|---|---|---|---:|---:|
| 0 | no-harm control | 100/100/100 | 100/100/100 | 100% | 1.084 |
| 3 | known drift failure | 100/83/59 | 100/100/100 | 98.63% at L256 | 0.098 |

The same weights, power, start step, ramp, and multipliers 2/3/4/5 used for GA
were retained. Both runs use deterministic execution and save checkpoints.

This targeted result supports path-coherent holonomy as a general intervention
for learned noncommutative orthogonal actions. It is not evidence that the GA
chart has a special interaction with holonomy, and two selected seeds are not a
Householder reliability estimate. A full replication is justified only if the
changed-generator tier leaves Householder in contention.
