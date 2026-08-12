# Spin(9) Dirac--Clifford sensing and exact design

## Scientific object

This program studies the nine symmetric Clifford involutions on the real
16-dimensional spin representation of \(\operatorname{Spin}(9)\). It extends
the maintained Spin(8) gamma system, but it does not replace the triality
programs and makes no sequence-memory superiority claim.

For

\[
D(a)=\sum_{i=0}^{8}a_iP_i,
\qquad
D(a)^2=\lVert a\rVert^2I_{16}.
\]

The second identity gives a norm-preserving real composition. Its use here is
in shared-action sensing and exact design.

## Claim ledger

| Claim | Evidence | Status boundary |
|---|---|---|
| Nine symmetric Clifford involutions and Spin(8) restriction | [`spin9_dirac_clifford.py`](../../Spin-Space-Research/src/spin9_dirac_clifford.py) | Exact coefficient and bracket checks |
| Generic one/two/three-spinor stabilizers | [`SPIN9_THREE_SPINOR_IDENTIFIABILITY.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_THREE_SPINOR_IDENTIFIABILITY.md) | Human stabilizer argument plus independent exact rank witnesses |
| Symmetric three-spinor conditioning curve | [`SPIN9_THREE_SPINOR_CONDITIONING.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_THREE_SPINOR_CONDITIONING.md) | Exact one-parameter theorem; not a global rank-three optimum |
| Frame-operator reduction and relaxed design optimum | [`SPIN9_FRAME_OPERATOR_REDUCTION.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_FRAME_OPERATOR_REDUCTION.md) | Exact theorem; the relaxation is unattainable by an exact three-probe frame |
| Numerical unrestricted search | [`SPIN9_THREE_SPINOR_GLOBAL_SCREEN.md`](../../Spin-Space-Research/docs/experiments/SPIN9_THREE_SPINOR_GLOBAL_SCREEN.md) | Ten-seed falsification campaign; no proof of global optimality |
| Cayley-null stabilizer and spectral branching | [`SPIN9_SYMMETRY_BRANCHING.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_SYMMETRY_BRANCHING.md) | Exact at the symmetric orbit |
| Fixed-plane orthonormality | [`SPIN9_FIXED_PLANE_ORTHONORMALITY.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_FIXED_PLANE_ORTHONORMALITY.md) | Exact for every plane on the symmetric curve; other support planes remain open |
| Grassmann normal slice | [`SPIN9_GRASSMANN_SLICE_THEOREM.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_GRASSMANN_SLICE_THEOREM.md) | Exact \(V_1\oplus V_5\) Grassmann normal slice; the second \(V_5\) belongs to supported frame-spectrum changes in the full rank-three tangent |
| Concrete slice-to-isotypic bridge | [`SPIN9_SLICE_ISOTYPIC_BRIDGE_RESULTS.md`](../../Spin-Space-Research/docs/experiments/SPIN9_SLICE_ISOTYPIC_BRIDGE_RESULTS.md) | Exact \(\mathbb Q(\sqrt2)\) intertwiner from the geometric slice to rational \(V_1\oplus V_5\), plus supported-basis alignment and compiler-certified \(V_1\oplus2V_5\); no global determinant consequence |
| Native algebraic isotypic compiler | [`ALGEBRAIC_ISOTYPIC_DECOMPOSITION_RESULTS.md`](../../Spin-Space-Research/docs/experiments/ALGEBRAIC_ISOTYPIC_DECOMPOSITION_RESULTS.md) | Direct exact \(\mathbb Q(\sqrt2)\) decomposition of the native slice and full quotient, independent of the rationalizing bridge; no determinant-sign consequence |
| Spin(8) and Clifford signature restriction | [`CLIFFORD_SIGNATURE_EXTENSION.md`](../../Spin-Space-Research/docs/manuscripts/CLIFFORD_SIGNATURE_EXTENSION.md) | Exact `8+ + 8-` restriction, triality Hom controls, faithful \(\mathrm{Cl}(1,4)\) image, and \(\mathrm{Cl}(3,0)\) even-subalgebra inclusion; no model consequence |
| Strict local rank-three optimum | [`SPIN9_STRICT_LOCAL_D_OPTIMALITY.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_STRICT_LOCAL_D_OPTIMALITY.md) | Exact internally replayed local theorem modulo Spin(9); independent external review remains pending |
| Complete spin-two ray bounds | [`SPIN9_V5_RAY_CERTIFICATE.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_V5_RAY_CERTIFICATE.md) | Exact zero-cubic and axisymmetric all-parameter graph-ray bounds; retained as the independently factored boundary theorem and Cayley-null radial counterexample |
| Full spin-two Cartan bound | [`SPIN9_V5_CARTAN_CERTIFICATE.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_V5_CARTAN_CERTIFICATE.md) | Exact all-shape pure-\(V_5\) graph theorem below \(101/100\) of Cayley-null |
| Coupled \(V_1\oplus V_5\) finite-radius theorem | [`SPIN9_V1_V5_RECONSTRUCTION.md`](../../Spin-Space-Research/docs/manuscripts/SPIN9_V1_V5_RECONSTRUCTION.md) | Exact characteristic-zero identity and global \(21/20\) bound on the complete coupled graph slice; exact candidate optimality, the second \(V_5\), and the unrestricted quotient remain open |
| Clifford memory boundary | [`SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md`](../../Spin-Space-Research/docs/experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md) | Exact 9-to-16 bind/unbind and Hopf coarse index; width expansion, not a same-state capacity advantage |

## Nonclaims and open gates

- The unrestricted global exact three-spinor optimum remains open.
- The full pure-\(V_5\) graph family and the complete coupled
  \(V_1\oplus V_5\) finite-radius graph slice are now certified. The latter has
  a raw characteristic-zero identity and a global \(21/20\) bound. Exact
  optimality of the algebraic candidate, the second supported \(V_5\), and the
  nonpolar global quotient remain open.
- The numerical search cannot replace a global proof.
- No matched experiment shows a same-state Spin(9) memory or sequence-model
  advantage. The exact binding is currently a candidate coarse router, not a
  fine-memory replacement.
- A single \(D(a)\) is an odd \(\operatorname{Pin}(9)\) element, not a
  Spin(9) rotor. Only an even product such as \(D(b)D(a)\) lies in the spin
  action.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python -m spin9_dirac_clifford --output artifacts/spin9_dirac_clifford_gate_20260807.json
python -m spin9_frame_operator --output artifacts/spin9_frame_operator_20260807.json
python -m spin9_local_hessian --output artifacts/spin9_local_hessian_exact.json
python -m spin9_slice_isotypic_bridge --output artifacts/spin9_slice_isotypic_bridge_20260811.json
python -m algebraic_isotypic_decomposition --output artifacts/algebraic_isotypic_decomposition_20260811.json
python -m clifford_signature_extension --output artifacts/clifford_signature_extension_20260811.json
python -m spin9_v5_ray_certificate --output artifacts/spin9_v5_ray_certificate_20260811.json
python -m spin9_v5_cartan_reconstruction --output artifacts/spin9_v5_cartan_reconstruction_20260811.json
python -m spin9_v5_cartan_certificate --output artifacts/spin9_v5_cartan_certificate_20260811.json
python -m spin9_v1_v5_reconstruction --workers 4 --quiet --output artifacts/spin9_v1_v5_reconstruction_20260811.json
python -m spin9_v1_v5_screen --output artifacts/spin9_v1_v5_screen_20260811.json
python -m spin9_v1_v5_boundary_char0 --workers 4 --output artifacts/spin9_v1_v5_boundary_char0_20260811.json
python -m spin9_v1_v5_blowup --output artifacts/spin9_v1_v5_blowup_20260811.json
python -m spin9_v1_v5_char0 --workers 4 --output artifacts/spin9_v1_v5_char0_20260812.json
python -m spin9_v1_v5_theorem --output artifacts/spin9_v1_v5_theorem_20260812.json
python -m spin9_clifford_memory --output artifacts/spin9_clifford_memory_boundary_20260810.json
python -m pytest -q tests/test_spin9_dirac_clifford.py tests/test_spin9_frame_operator.py tests/test_spin9_grassmann_slice.py tests/test_spin9_slice_isotypic_bridge.py tests/test_algebraic_isotypic_decomposition.py tests/test_clifford_signature_extension.py tests/test_spin9_local_hessian.py tests/test_spin9_v5_ray_certificate.py tests/test_spin9_v5_cartan_certificate.py tests/test_spin9_v1_v5_reconstruction.py tests/test_spin9_v1_v5_boundary_char0.py tests/test_spin9_v1_v5_blowup.py tests/test_spin9_v1_v5_global.py tests/test_spin9_v1_v5_char0.py tests/test_spin9_v1_v5_theorem.py tests/test_spin9_clifford_memory.py
```
