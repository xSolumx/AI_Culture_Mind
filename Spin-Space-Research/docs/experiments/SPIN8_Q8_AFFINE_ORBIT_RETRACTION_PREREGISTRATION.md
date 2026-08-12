# Spin(8) Q8 affine-orbit retraction: seed-4 diagnostic protocol

Status: frozen after observing the original 8/9 cohort and before applying the
new intervention to seed 4.

Date: 2026-08-03

This is a post-cohort mechanistic diagnosis, not part of the original fresh
reliability result. Its seed-4 outcome is exploratory. Any reliability claim
for this refinement requires new untouched seeds.

## Diagnosed error in the first retraction

An exact Q8 orbit in one channel need not be centered at zero or occupy its
faithful component with unit amplitude. It can have the form

```text
h_g = m + alpha E e_g,
```

where `m` is fixed by every token action, `E` is an orthonormal `8 x 4`
faithful frame, and `alpha` controls orbit variation. The first retraction set
`m=0` and `alpha=1` in every channel. That can turn a mostly fixed nuisance
channel into a full-strength varying code and change frozen-decoder logits.

## Frozen refinement

For each channel, using all eight canonical states jointly:

1. retain the orbit mean projected onto the complement of the faithful frame;
2. recover the faithful frame by the same central-pair polar decomposition;
3. set `alpha` to the least-squares scaled-polar value, the mean of the four
   singular values of the central-pair difference matrix;
4. initialize the exact orbit at `m + alpha E e_identity`, normalized once as
   the existing model contract requires;
5. induce all token actions from the exact Q8 law on `E` and identity on its
   complement;
6. leave the decoder frozen.

No channel is selected or removed. No threshold is introduced. Token actions
are still constructed jointly, never normalized independently.

## Seed-4 diagnostic gate

The refined seed-4 model passes if it retains:

- action reconstruction maximum `<= 1e-5`;
- full homomorphism RMS `<= 1e-5`;
- at least `99%` member and joint accuracy at every dense and long length;
- streaming state residual `<= 1e-5` and logit residual `<= 1e-4`.

Regardless of pass/fail, report `||m||`, `alpha`, orbit projection RMS, and the
old versus refined decoder behavior. Do not add seed 4 back into the original
prospective 8/9 count; report this as a separate repaired-seed diagnostic.
