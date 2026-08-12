# Referee package: balanced Cayley information spectrum

This directory is a compact review object for one theorem-sized result. It can
be read and verified without following the repository's chronological gate
history.

## Claim in one sentence

For the displayed one-parameter balanced `Spin(8)` triality sensor family, the
complete exact information spectrum has determinant

\[
\det I_c=\frac{(1-c^2)^3(9-c^2)^2}{1024};
\]

the Cayley-null member uniquely optimizes three spectral criteria within that
family, and exactly three eigenvalues vanish at rate `(1-c^2)/8` at either
calibrated endpoint.

The extension from the displayed family to every orthonormal balanced design
uses a separate orbit-normal-form proposition. That bridge is identified
explicitly rather than hidden inside the symbolic certificate.

## Read in this order

1. [`THEOREM.md`](THEOREM.md) — the precise statement and domain.
2. [`PROOF.md`](PROOF.md) — the human-readable proof and every reduction.
3. [`TRUST_BOUNDARY.md`](TRUST_BOUNDARY.md) — what is proved by hand, supplied,
   or recomputed.
4. [`OPEN_PROBLEMS.md`](OPEN_PROBLEMS.md) — nonclaims and unresolved questions.
5. [`REPRODUCE.md`](REPRODUCE.md) — a short replay procedure.

## Independent minimal verifier

[`verify.py`](verify.py) uses only the Python standard library. It does not
import SymPy, FLINT, or any project module. Starting from the four block
characteristic polynomials printed in the proof, it independently recomputes:

- the complete degree-28 characteristic polynomial;
- determinant, trace, and second direct moment;
- both inverse spectral moments;
- exact derivative factorizations and the Bernstein sign certificate;
- endpoint block factors and the three equal first-order slopes.

Run:

```text
python verify.py
```

Expected output begins with:

```text
PASS: independent exact Cayley-spectrum certificate reproduced
```

The stored exact coefficient artifact is
[`artifacts/certificate.json`](artifacts/certificate.json). Package and
upstream hashes are declared separately; hashes establish byte identity, not
mathematical truth.

## Submission manuscript

The arXiv-oriented LaTeX source remains in
[`papers/cayley-information-spectrum`](../../papers/cayley-information-spectrum/README.md).
The paper is broader exposition; this package is the compact object a referee
can audit first.
