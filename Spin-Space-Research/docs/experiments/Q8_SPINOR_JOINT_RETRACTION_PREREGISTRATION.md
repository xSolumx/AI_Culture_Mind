# Q8 spinor joint-family retraction: prospective contract

Date fixed: 2026-08-03, after the seed-0 learned spinor smoke and before any
retracted-checkpoint evaluation.

## Motivation

Seed 0 has three channels near a faithful Q8 action and one nuisance channel.
The behavioral gate is exact through L16384, but the learned operators remain
approximate. Test the requested principle directly: train unconstrained tangent
updates, then retract the **entire token-action family** onto one shared
representation manifold. Never round or normalize tokens independently.

## Frozen retraction

For each channel, read the four learned unit quaternions `(q_i,q_-i,q_j,q_-j)`:

1. form the inverse-pair vector differences
   `a_raw = vec(q_i-q_-i)` and `b_raw = vec(q_j-q_-j)`;
2. normalize `a_raw`, then remove its component from `b_raw` and normalize the
   residual, producing one orthonormal generator frame `(a,b)`;
3. replace the complete family jointly by
   `(a,-a,b,-b)` with zero scalar parts;
4. map those four exact target quaternions back through the common 2pi tangent
   chart and save a new checkpoint.

This construction enforces `i^2=j^2=-1`, inverse antipodality, and `ij=-ji` by
one coupled frame. It uses the known Q8 token/inverse-pair contract, but no
endpoint labels, decoder outputs, per-token search, or post-retraction training.
The initial orbit state, decoder, and every non-action parameter remain frozen.

All four channels are retracted, including the nuisance channel. No channel is
selected or dropped after seeing its effect.

## Gates

On seed 0:

1. per-channel Q8 relator RMS and active homomorphism RMS must be <=`1e-5`;
2. central-pair member and both-correct accuracy must remain >=99% at every
   smoke dense checkpoint and the four long base lengths;
3. recurrent state parity must remain exact within `1e-5`;
4. projection distance is reported per channel, especially for the former
   nuisance channel.

If seed 0 passes, apply the identical no-threshold retraction to every checkpoint
in the already-running ten-seed spinor cohort. Reliability is the all-seed
count; no channel-specific rescue is allowed.
