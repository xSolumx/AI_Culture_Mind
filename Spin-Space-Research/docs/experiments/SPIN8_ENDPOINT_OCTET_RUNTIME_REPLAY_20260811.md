# Spin(8) Endpoint-Octet Twenty-Output Replay

**Exact mathematical replay and operational audit — 2026-08-11**

## Why these files were revisited

Twenty endpoint-octet outputs existed outside a maintained promotion path.
Nineteen were workstation watchdog records from 2026-08-10; the twentieth was
the complete 32-child atlas for first-level cubic box `00001`. The watchdog
records contain elapsed time, peak RSS, affinity, and return codes. They are
operational evidence, not positivity certificates, and now live under ignored
`runtime/endpoint-octet/2026-08-10/` rather than `artifacts/`.

The mathematical JSON named by each watchdog command remains a published,
hash-manifested artifact. The replay tool reconstructs every command into an
isolated runtime tree, preserving the published files as immutable evidence.

## Frozen inventory

| Class | Count | Outputs |
|---|---:|---|
| Finite-radius blow-up charts | 5 | pivots `p0` through `p4` |
| Selected-boundary identities | 3 | pivots `p0` through `p2` |
| Boundary atlases | 10 | `p0_double`; five `p0_ui` routes; `p1`, `p1_0010`, `p2`, and `p2_0010` |
| Symmetry audit | 1 | middle-chart permutation audit |
| Complete mathematical atlas | 1 | all 32 children of first-level box `00001` |
| **Total** | **20** | one unique command and canonical mathematical output per row |

The completed `00001` atlas is a mathematical certificate, not a runtime
record. Its canonical SHA-256 is
`7238a7ff759f1053ef8dc5ebf7f9153b8a903de88639cea34d62ba1353e61b7a`.

## Acceptance policy

The campaign records two distinct reproducibility gates.

1. **Strict byte match:** fresh and canonical JSON files have identical
   SHA-256 hashes.
2. **Exact mathematical match:** after narrowly frozen schema normalization,
   the complete JSON trees agree exactly.

Normalization may remove only:

- `audit_engine` and `batch_entry_limit`, which describe the exact transform
  implementation rather than its coefficients;
- `parent_path` and `post_zero_path` when their value is exactly the empty
  default `[]`;
- `selected_zero_face` when its value is exactly the null default; and
- two exact old/current string aliases for the UI-face label and its
  corresponding nonclaim sentence.

Every difference path is retained in the campaign receipt. Nonempty routes,
selector objects, coefficients, multidegrees, tensor shapes, sign counts,
minimum controls, identities, pass flags, and unregistered claim-boundary text
are never normalized. Regression tests include adversarial nonempty-route and
scope-change controls.

## Completed campaign result

The isolated 2026-08-11 campaign completed all twenty jobs under the declared
two-job, six-worker, 14.8-GiB aggregate contract:

- **20/20 exact mathematical payloads matched;**
- **20/20 fresh processes exited with return code 0;**
- **11/20 outputs matched byte-for-byte;** and
- **9/20 required only the frozen schema-compatibility rules above.**

Strict byte reproducibility therefore remains a negative result even though
the mathematical replay gate passes. Fresh elapsed times ranged from 173.52
seconds to 14,055.63 seconds. The largest observed process-tree RSS was 5.895
GiB, below the 7.4-GiB per-job ceiling. The ignored local campaign receipt is
`runtime/endpoint-octet/replay-2026-08-11/campaign.json`, SHA-256
`e1307681d0fa9ee136be360907e3d58ceeb444f35fe74887b2f1bc80e97a7380`.

| Job | Byte | Math | Fresh rc | Seconds | Peak GiB | Historical rc |
|---|:---:|:---:|---:|---:|---:|---:|
| `atlas_nested_00001_complete` | yes | yes | 0 | 14055.63 | 4.734 | -- |
| `blowup_p0` | yes | yes | 0 | 549.08 | 5.702 | 0 |
| `blowup_p1` | yes | yes | 0 | 564.26 | 5.695 | 0 |
| `blowup_p2` | yes | yes | 0 | 530.21 | 5.614 | 0 |
| `blowup_p3` | yes | yes | 0 | 353.01 | 5.566 | 0 |
| `blowup_p4` | no | yes | 0 | 281.60 | 5.564 | 0 |
| `boundary_atlas_p0_double` | no | yes | 0 | 189.41 | 5.565 | 0 |
| `boundary_atlas_p0_ui_0000_r0_001` | yes | yes | 0 | 173.52 | 5.564 | 0 |
| `boundary_atlas_p0_ui_0000_r0` | no | yes | 0 | 177.67 | 5.565 | 0 |
| `boundary_atlas_p0_ui_0001_r0_000` | yes | yes | 0 | 179.50 | 5.565 | 0 |
| `boundary_atlas_p0_ui_0001_r0` | no | yes | 0 | 179.36 | 5.565 | 0 |
| `boundary_atlas_p0_ui` | no | yes | 0 | 920.56 | 5.566 | 0 |
| `boundary_atlas_p1_0010` | no | yes | 0 | 1062.73 | 5.564 | 0 |
| `boundary_atlas_p1` | no | yes | 0 | 892.37 | 5.566 | 120 |
| `boundary_atlas_p2_0010` | no | yes | 0 | 1047.39 | 5.569 | 0 |
| `boundary_atlas_p2` | no | yes | 0 | 880.22 | 5.564 | 0 |
| `boundary_p0` | yes | yes | 0 | 201.69 | 5.565 | 0 |
| `boundary_p1` | yes | yes | 0 | 192.46 | 5.564 | 0 |
| `boundary_p2` | yes | yes | 0 | 192.51 | 5.564 | 0 |
| `middle_symmetry` | yes | yes | 0 | 232.13 | 5.895 | 0 |

The internally failed `middle_symmetry` experiment remains a byte-identical
negative finding: no chart-swap shortcut was recovered. “Math = yes” in the
table means that this complete negative payload reproduced, not that its
internal `passed` field changed. Conversely, the historical return code 120
for `boundary_atlas_p1` did change to a clean fresh exit without changing its
exact mathematical payload.

## Historical return-code anomaly

The only failed 2026-08-10 watchdog record was
`boundary_atlas_p1_runtime_20260810.json`: return code 120 after 928.66 seconds,
despite a complete canonical mathematical JSON. The fresh isolated replay
completed with return code 0 in 892.37 seconds at 5.566 GiB peak process-tree
RSS. Its exact mathematical payload matches after the same frozen optional
default and UI-label schema normalization used elsewhere. This supports the
operational diagnosis that the old 120 came from the expired output wrapper or
broken pipe, not from the exact atlas computation.

## Mathematical promotion unlocked by the audit

The replay investigation exposed and completed a larger missing gate. A fresh
coarse atlas certifies the endpoint-factor quotient on all 32 first-level
five-cube boxes. Boxes `00001` and `00010` delegate only to independent,
hash-bound exact certificates; the other 30 run complete batched Bernstein
audits. The exact artifact is
`spin8_dirac_endpoint_octet_cubic_coarse_atlas_20260811.json`, SHA-256
`1747ea735a9bd9cfe16b392728eeb7b2c89293c6dbbae7056beca746bb461f79`.

The final assembly then recomputes in characteristic zero

\[
C=C|_{y=0}(1-y)^{36}+C|_{y=1}y^{36}+y(1-y)Q,
\]

including the 1,546,277-term cubic, 2,824,946-term quotient, exact division,
and forced-radical cancellation. Both endpoint faces and the complete quotient
cover are nonnegative, so

\[
\boxed{C(u_d,u_e,u_g,u_i,y)\geq0\quad\text{on }[0,1]^5.}
\]

The final artifact is
`spin8_dirac_endpoint_octet_cubic_certificate_20260811.json`, SHA-256
`86e86cbbba47639b716140abf15234d3a60f155ac25674f9356bd035f34217c0`.

## Claim boundary

The cubic principal minor is proved on the complete adjacent endpoint
five-cube. The fourth-order determinant of the second Schur block remains the
sole Schur-minor obstruction on that face. Therefore neither the complete
adjacent endpoint octet, the unrestricted seven-variable Dirac--Gram
inequality, nor global five-query optimality is promoted here.

The determinant experiment is frozen separately in
[`SPIN8_DIRAC_OCTET_DETERMINANT_PREREGISTRATION.md`](SPIN8_DIRAC_OCTET_DETERMINANT_PREREGISTRATION.md).
