# Manuscripts

These papers isolate theorem-sized contributions from the chronological
research archive. They are written to be read independently, while their exact
certificates remain executable from the repository.

| Manuscript | Status | Central result |
|---|---|---|
| [Balanced Cayley Information Spectra](CAYLEY_INFORMATION_SPECTRUM.md) | Exact theorem manuscript | Complete orthonormal balanced spectrum; simultaneous D-, A-, and inverse-Frobenius optimality; exact three-mode boundary-conditioning law |
| [Signed-Star Cubic Gram-Volume Inequality](SIGNED_STAR_DIRAC_GRAM.md) | Computer-assisted exact theorem manuscript | Cubic determinant inequality on the full signed-star subfamily; exact structural factors and complete normalized equality classification |
| [Global Fourier-Energy Inequality](UNRESTRICTED_FOURIER_ENERGY.md) | Computer-assisted exact theorem note | Complete seven-cube bound on the combined energy of all fifteen nontrivial orientation sectors; exact boundary-supported Bernstein decomposition |
| [Cayley-Endpoint Klein-Four Positivity](UNRESTRICTED_ENDPOINT_KLEIN_FACE.md) | Computer-assisted exact boundary theorem | Positive-semidefinite Klein-four group-circulant reduction on a complete four-variable endpoint face; nested principal-minor certificate |
| [Adjacent Endpoint Octet Reduction](UNRESTRICTED_ENDPOINT_OCTET_REDUCTION.md) | Exact reduction and partial boundary theorem | Eight-sector subgroup-chain reduction; complete first Schur block; scalar, quadratic, and cubic second-block theorems; determinant endpoints proved and interior open |
| [Spin(9) Strict Local D-optimality](SPIN9_STRICT_LOCAL_D_OPTIMALITY.md) | Exact local theorem | Full \(44\)-dimensional rank-three tangent decomposition and negative-definite coupled \(V_5\) Hessian at the algebraic three-spinor candidate |
| [Spin(9) Full Spin-Two Cartan Bound](SPIN9_V5_CARTAN_CERTIFICATE.md) | Computer-assisted exact theorem | Complete pure-\(V_5\) graph family below \(101/100\) of Cayley-null by deterministic modular lifting and a six-cell strict Bernstein atlas |
| [Spin(9) Coupled \(V_1\oplus V_5\) Reconstruction](SPIN9_V1_V5_RECONSTRUCTION.md) | Computer-assisted exact theorem | Full characteristic-zero identity and global \(21/20\) determinant bound on the complete finite-radius coupled slice; exact candidate optimality, the second \(V_5\), and the unrestricted quotient remain open |
| [Clifford Signature Extension](CLIFFORD_SIGNATURE_EXTENSION.md) | Exact algebra and branching theorem | Faithful \(\mathrm{Cl}(1,4)\) matrix image, quaternionic volume sectors, exact \(\mathrm{Cl}(3,0)\) even-subalgebra inclusion, and Spin(8) triality-module controls |

The hierarchical Spin(8)/Spin(9) memory campaign is intentionally not listed
as a theorem manuscript. Its algebraic compiler, learned-routing cohorts, and
CUDA measurements form a reproducible empirical/systems report in
[`docs/experiments`](../experiments/SPIN8_SPIN9_HIERARCHICAL_MEMORY_RESULTS.md).
A future architecture paper requires a model-level matched benchmark and may
cite the triality theorem only for cross-view equivariance, not storage
capacity.

Neither manuscript claims the unrestricted signed Dirac--Gram inequality or
global five-query D-optimality over all allocations and nonorthogonal frames.
Those questions remain open.

The Cayley manuscript uses a hybrid proof at one specific point: the global
four-plane orbit classification is classical, while the internal \(2+2\)
split isotropy and information algebra are recomputed exactly here. See the
[Cayley flag quotient audit](../CAYLEY_FLAG_QUOTIENT_AUDIT_2026-08-06.md).

For independent review of the Cayley result, use the compact
[referee package](../../referee/cayley-information-spectrum/README.md). It
separates the canonical-family algebra from the orbit-completeness bridge and
reconstructs the degree-28 spectral consequences without importing the project
or a computer-algebra system.
