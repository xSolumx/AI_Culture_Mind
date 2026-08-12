# Preregistration: Determinant Gate on the Adjacent Endpoint Octet

**Frozen before reconstructing the determinant polynomial — 2026-08-11**

**Execution update — 2026-08-12:** frozen stages 1--6 have now been executed.
The exact reconstruction and both \(y\)-endpoint faces are proved. The
degree-matched two-endpoint quotient was exactly constructed and rejected by a
rational negative witness, while the determinant itself has a proved
order-eight squared nonnegative tangent form at the remaining equality corner.
The finite-radius interior and global determinant acceptance gate remain open. See
[`SPIN8_DIRAC_OCTET_DETERMINANT_ENDPOINT_RESULTS.md`](SPIN8_DIRAC_OCTET_DETERMINANT_ENDPOINT_RESULTS.md).

## Scope

The adjacent endpoint reduction leaves the symmetric Klein-four circulant

\[
Z=
\begin{pmatrix}
Z_0&A&B&C\\
A&Z_0&C&B\\
B&C&Z_0&A\\
C&B&A&Z_0
\end{pmatrix},
\qquad
A=\sqrt{s_1}Z_1,\ B=\sqrt{s_2}Z_2,\ C=\sqrt{s_3}Z_3.
\]

The scalar, all three quadratic families, and the cubic principal minor are
already nonnegative on the complete five-cube. The sole remaining Schur-minor
obligation is

\[
\begin{aligned}
D:=\det Z={}&Z_0^4-2Z_0^2(A^2+B^2+C^2)+8Z_0ABC\\
&+A^4+B^4+C^4-2(A^2B^2+A^2C^2+B^2C^2).
\end{aligned}
\]

Because the previously verified forced-square identity gives

\[
s_1s_2s_3=\tau^2,
\qquad
\tau=u_du_eu_gu_i\,y^2
(1-u_d)(1-u_e)(1-u_g)(1-u_i),
\]

the exact radical-free target is

\[
\boxed{
\begin{aligned}
D={}&Z_0^4-2Z_0^2\sum_{j=1}^3s_jZ_j^2
{}+8\tau Z_0Z_1Z_2Z_3\\
&+\sum_{j=1}^3(s_jZ_j^2)^2
-2\sum_{1\le j<k\le3}s_js_kZ_j^2Z_k^2.
\end{aligned}}
\]

Before coefficient reconstruction, this abstract identity was checked by
expanding both the symbolic four-by-four determinant and the product of its
four Walsh eigenvalues. The eventual source harness must repeat that check;
this preliminary algebra check is not a positivity result.

The domain is
\((u_d,u_e,u_g,u_i,y)\in[0,1]^5\). Degree
\((24,24,24,24,48)\) is the structural prediction and must be checked after
exact reconstruction; it is not yet an artifact-backed fact.

## Frozen evidence hierarchy

1. Reconstruct all four radical-free coefficients from the published exact
   coefficient source and reverify \(s_1s_2s_3=\tau^2\).
2. Verify the displayed determinant formula independently against a symbolic
   four-by-four determinant before any positivity audit.
3. Record term count, multidegree, exact endpoint factors, native Bernstein
   sign counts, and complete negative-support histograms. Native negativity
   rejects only that certificate basis.
4. Use the exact \(y=1\) implication \(Z=X^2\), \(X\succeq0\), hence
   \(D\geq0\). Prove the \(y=0\) face independently; do not infer it from
   numerical samples or from the already-proved cubic.
5. Only after both endpoint faces pass, divide the exact selector remainder by
   its verified forced factor and audit the quotient.
6. If failures localize at an equality point, compute the first nonzero exact
   tangent form before choosing dyadic depth or max-coordinate blow-up charts.
7. Every atlas must be a finite complete binary cover. Delegation is allowed
   only to an independently hash-bound exact certificate whose domain contains
   the entire delegated cell.
8. A floating-point search is a falsifier only. Any apparent negative point
   must be reconstructed and checked in exact arithmetic.

## Acceptance and rejection

- **Determinant pass:** a domain-wide exact certificate for \(D\geq0\), with
  every endpoint, selector, tangent, delegation, and atlas obligation replayed.
- **Exact disproof:** a feasible five-cube point with \(D<0\) verified in exact
  arithmetic.
- **Inconclusive:** neither a complete certificate nor an exact counterexample.

## Resource and artifact contract

- Each exact process uses at most six FLINT threads and remains below 16 GiB
  RSS under `spin8_resource_limits.py`.
- Expensive transforms are checkpointed atomically and resumable.
- Runtime watchdog records live under ignored `runtime/`; only mathematical
  JSON outputs enter `artifacts/` and `ARTIFACTS.sha256`.
- Compact verifiers must state when they trust stored Bernstein summaries; the
  full source harness remains the replay path for every transform.

## Nonclaims

This preregistration proves nothing about \(D\). Even a determinant pass would
prove only the complete adjacent endpoint octet after assembly with the
already-proved proper minors. It would not prove the unrestricted
seven-variable Dirac--Gram inequality or global five-query D-optimality.
