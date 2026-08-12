# One-calibration Cayley retraction: worst-case-safe GPU contract

Date fixed: 2026-08-03, after the exact-half 2-SAT falsifier and before any
121-edge GPU cohort.

## Mathematical contract

The mask contains one orientation from each of 120 true reverse-edge pairs,
plus the hidden reverse direction for exactly one uniformly selected pair. The
compiler sees 121/240 directed edges (50.4167%). It is not told which tokens
are inverses.

For the four-token alphabet there are three perfect matchings. In the complete
A5 action, the true matching has 240 directed two-step identities and both
wrong matchings have zero. The added calibration pair therefore gives the true
matching support 2 while both wrong matchings score 0, even if a wrong matching
is otherwise globally feasible. Once that pair is selected, the remaining two
tokens are forced to pair. This closes the adversarial ambiguity exhibited in
`inverse_cover_adversarial_audit.json`.

This is worst-case-safe for every orientation of the other 119 reverse pairs
under the stated assumptions: four distinct tokens arranged in two inverse
pairs, deterministic permutation transitions, exact labels, and no wrong token
pair composing to identity. It is not claimed for noisy labels, self-inverse
tokens, more tokens without additional calibration, or arbitrary missingness.

## Frozen cohort

- Training seeds: 0 through 9.
- Mask seed: `910001 + training_seed`.
- Exactly one globally allocated calibration pair.
- Architecture: four-channel pure Cl(3) rotor recurrence.
- 2,000 deterministic CUDA steps, batch 256, sequence length 16.
- Ambient token gradients before discovery; one joint shared-conjugacy
  retraction afterward. No token is normalized or rounded independently.
- Dense prefix labels remain available to the task loss.
- Untouched changed-generator selection index 44.
- Dense evaluation L16, L32, ..., L256; L4096 original and untouched; L16384
  untouched.

## Gates

All ten seeds must:

1. observe exactly 121 edges and infer `(1, 0, 3, 2)`;
2. complete exactly 119 edges and replay all 240 transitions;
3. reconstruct order 60 and compile a 3D representation with invariance and
   homomorphism RMS below `1e-10`;
4. trigger joint representation-manifold retraction;
5. reach at least 90% at every registered dense and long length on every
   required alphabet.

The headline is the all-seed count, not a mean. Compiler exactness, float32
mechanism error, dense minimum, and long minimum remain separate metrics.

## Strong invariance prediction

The 121-edge and completed 122-edge masks reconstruct the same exact action
before any compiler or optimizer choice. Because masking uses a local NumPy
generator and deterministic training uses independent fixed streams, the final
`state_dict`, logged trajectory, and post-recovery reports should be exactly
equal seed-for-seed. Tensor inequality would falsify the claimed separation
between evidence recovery and downstream optimization even if accuracy passes.
