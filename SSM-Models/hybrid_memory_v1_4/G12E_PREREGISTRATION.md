# G12E measured-compute matching amendment

**Frozen:** 2026-08-25, after G12D and before any G12E calibration or training  
**Reason for amendment:** G12C's parameter-matched BPE arm used 26.6% less
median update time than the raw AdamW control, so the original evidence does not
show what BPE can do at the raw control's measured CUDA budget.

## Fixed target and calibration

- Hardware/runtime: current native PyTorch CUDA runtime on the same RTX 2070
  SUPER used by G12.
- Target: v1.4.5 raw-byte width 64, expansion 2, vocabulary 256, AdamW `1e-3`,
  weight decay `0.01`.
- Candidate family: v1.4.5 retention-safe architecture, vocabulary 512, the
  frozen train-only ByteLevel BPE, and `HarmonicMuonAdamW`.
- Enumerate model widths 24 through 96 in increments of 4 and FFN expansions 1
  through 6. Invalid RoPE/head shapes are recorded as invalid and not timed.
- Each valid row uses seed 1951, one fixed random batch of shape `(16,256)`, five
  warmup updates, and 15 synchronized measured forward/backward/optimizer
  updates. The loss is ordinary next-token cross entropy.
- Select the row with median update time closest to the freshly measured raw
  target. Ties choose fewer trainable parameters, then larger width, then
  smaller expansion. Report the residual mismatch; compute matching passes only
  when the absolute residual is at most 10%.
- Calibration sees no TinyStories text and no validation loss.

## Outcome training

- Train the selected candidate on exactly the frozen G12C BPE token stream,
  windows, optimizer, seed cohort `(1871,1873,1877)`, batch 16, length 256, and
  1,000 updates. It receives no hyperparameter or architecture change after
  calibration.
- Record BPRB at updates 0, 250, 500, 750, and 1,000; original bytes presented;
  synchronized update time; peak CUDA allocation; parameters; tokenizer hash;
  checkpoints; and all individual seeds.
- The comparison control is the immutable G12C raw AdamW cohort. Its artifact
  and checkpoint hashes remain unchanged.

## Decision

A measured-compute Pareto improvement requires:

1. calibration residual within 10%;
2. all candidate metrics finite;
3. candidate mean final BPRB below raw AdamW mean final BPRB;
4. candidate worst-seed BPRB below raw AdamW worst-seed BPRB; and
5. outcome median update-time mean no more than 10% above the raw control mean.

The parameter-matched, token-update-matched, original-byte exposure, and
measured-compute views remain separate. This experiment does not estimate a
scaling exponent and does not prove hardware-general efficiency.
