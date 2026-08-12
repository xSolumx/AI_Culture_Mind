# Endpoint supervision has a finite-group mixing barrier

Date: 2026-08-03.

## Observation

The fixed-L16 endpoint-only rotor runs do not have vanishing recurrent
Jacobians: rotor actions are norm-preserving, and their measured action-gradient
RMS is nonzero. What disappears is **coherent first-order direction**.

For the exact A5 training sampler, the endpoint contains nearly the maximum
possible class entropy (`5.8911` of `log2(60) = 5.9069` bits), yet any one
token position contains almost no information about that endpoint by L16:

| Word length | Mean single-position endpoint MI | Position range | Mean conditional TV |
|---:|---:|---:|---:|
| 1 | 2.0000 bits | 2.0000--2.0000 | 0.7500 |
| 2 | 1.4536 bits | 1.4456--1.4616 | 0.5632 |
| 4 | 0.5508 bits | 0.4714--0.6373 | 0.3278 |
| 8 | 0.0404 bits | 0.0127--0.0911 | 0.0898 |
| 16 | 0.001279 bits | 0.000113--0.005932 | 0.0139 |

The augmented `(group state, previous token)` Markov kernel has second-largest
eigenvalue modulus `0.85871269`, consistent with rapid decay of conditional
structure under repeated steps.

At the common identity initialization, a 32-batch empirical audit gives:

| Length | RMS batch action-gradient norm | Norm of mean gradient | Mean cosine to mean gradient |
|---:|---:|---:|---:|
| 1 | 0.2967 | 0.2949 | 0.9942 |
| 2 | 0.3527 | 0.3483 | 0.9877 |
| 4 | 0.2973 | 0.2798 | 0.9439 |
| 8 | 0.2326 | 0.1528 | 0.6596 |
| 16 | 0.3807 | 0.1070 | 0.2676 |

Thus L16 gradients are not small per batch. They disagree strongly across
batches, and their coherent component is small.

## General statement

Let a finite-state endpoint process be represented by an ergodic augmented
Markov kernel `K`, and let `X_j` be the token at position `j`, `Y_L` the final
group state. For each token value `a`, the deviation

```text
P(Y_L | X_j=a) - P(Y_L)
```

is propagated through the suffix by a power of `K` (and through the prefix by
the corresponding forward distribution). On the zero-mass subspace this
deviation contracts at the mixing rate of `K`; for a reversible kernel the
standard L2 bound is controlled by the second eigenvalue, while the present
non-reversible/augmented sampler can be bounded with singular values or an
appropriate operator norm.

Pinsker's inequality connects this contraction to information:

```text
TV(P(Y|X=a), P(Y)) <= sqrt(KL(P(Y|X=a)||P(Y)) / 2).
```

Averaging the conditional KL terms gives `I(X_j;Y)`. Therefore any bounded
first-order gradient statistic whose useful mean is a conditional endpoint
contrast is bounded by the same vanishing conditional variation. This covers
the token-action gradient at an identity-symmetric recurrence up to the model's
bounded local Jacobian and decoder residual.

## What is and is not proved

This explains a **first-order optimization barrier near symmetric
initialization**. It is not an impossibility theorem for endpoint-only
learning:

- higher-order parameter interactions can encode products even when every
  single-position marginal is nearly independent of the endpoint;
- a sufficiently large random fluctuation, different initialization, or
  optimizer could escape the symmetric basin;
- active endpoint membership queries already identify the algebra exactly;
  the barrier concerns the neural task optimizer, not compiler
  identifiability;
- the numerical rates above are specific to the A5 alphabet and held-out-pair
  sampler.

## Curriculum as homotopy

The length curriculum is best understood as an information homotopy. At L1
and L2 the token actions receive coherent first-order gradients; after those
actions form a partial representation, longer products become learnable even
though their identity-initialized first-order signal would have cancelled.

This interpretation makes two controls mandatory:

1. shuffle the exact same short/long batch multiset to test whether exposure or
   staged ordering matters;
2. extend fixed-L16 training to test whether persistence alone eventually
   escapes the mixed basin.

Artifacts: `endpoint_credit_assignment_audit.py` and
`endpoint_credit_assignment_audit.json`.
