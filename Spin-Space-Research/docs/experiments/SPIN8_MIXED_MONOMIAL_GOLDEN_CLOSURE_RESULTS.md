# Exact Mixed Monomial--Golden Spin(8) Closure Result

**Updated:** 2026-08-21T14:29:36+02:00
**Status:** exact infinitude discovery certificate. Its former “no density claim” boundary is superseded by the later exact [`SPIN8_MIXED_CLOSURE_SO8_THEOREM.md`](SPIN8_MIXED_CLOSURE_SO8_THEOREM.md).

## Result

In the maintained eight-dimensional basis, take the nine published generators of the associative signed-monomial octonion-operator normalizer and the golden-field vector generators `a,b` of the icosahedral `2.A5` construction. Their generated subgroup is **infinite**.

The exact witness is the positive length-three word

\[
b\,\bigl(\beta\,(\rho_{e_1})\bigr),
\]

where `b = golden_vector_b`, `beta = fano_automorphism_b`, and `rho_e1` is left multiplication by the first imaginary octonion basis unit. Its characteristic polynomial over \(\mathbb Q(\sqrt5)\) is

\[
(x-1)^2(x^2+1)\left[x^4+\frac{5-\sqrt5}{4}(x^3+x^2+x)+1\right].
\]

The quartic factor divides none of the cyclotomic polynomials whose roots can have degree at most eight over \(\mathbb Q(\sqrt5)\) (equivalently degree at most sixteen over \(\mathbb Q\)). It therefore contributes an eigenvalue that is not a root of unity. A finite-order matrix has only root-of-unity eigenvalues, so this word—and hence the subgroup containing it—has infinite order.

Because the all-three-triality-view mixed system contains these eleven generators, it is also infinite. The later theorem identifies the closure of this fixed system—and hence of that enlargement—as \(SO(8)\). It still does not identify an abstract presentation for the dense subgroup.

## Reproducible evidence

The replay script uses canonical pairs \(a+b\sqrt5\) with rational `a,b` for all bounded matrix arithmetic and comparisons. The stored artifact records:

- 758 distinct positive words through length three (cumulative counts `12, 119, 758`);
- exact mixed-pair orders, including two order-60 products;
- two candidate vector orbits that each exceed the exact 2,048-state cap;
- the exact characteristic-polynomial factor gate above.

An independent exact commutant calculation has dimension one over
\(\mathbb Q(\sqrt5)\), both for the seven left-operator generators and for
the full eleven-generator mixed system. Thus only scalars commute with the
stated mixed representation. This is an irreducibility diagnostic, not a
density or Lie-closure result.

The floating-point eigenvalue calculation is retained only to select a short candidate word. The infinitude statement comes from exact factorization, not from that screen or from an orbit cap.

```powershell
Set-Location C:\Users\HaydenLocal\Programming\AI_Culture_Mind\Spin-Space-Research
$env:PYTHONPATH = "src"
python src\spin8_mixed_monomial_golden_closure.py `
  --output artifacts\spin8_mixed_monomial_golden_closure_20260821.json
python -m pytest -q tests\test_spin8_mixed_monomial_golden_closure.py
```

## Claim boundary

Proved in this discovery-stage certificate: infinitude of the specified mixed vector-view subgroup and every arithmetic statement recorded in its artifact. The later theorem proves its topological closure. Neither report proves novelty, an abstract presentation, a finite quotient, an all-triality irreducible representation theorem, or any ML/scan-speed advantage.
