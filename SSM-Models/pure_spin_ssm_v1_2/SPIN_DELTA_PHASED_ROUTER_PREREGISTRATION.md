# Spin-Delta phase-separated router gate

**Protocol status:** specified before frozen outcome training.

## Question

The autonomous causal-router gate identified every event and slot perfectly,
but retrieval ranged from 79% to 99% across seeds. The oracle intervention was
uniformly near-perfect because its routing was exact from optimizer step one.
This gate tests the narrow remaining hypothesis: early hard-routing errors
poison recurrence learning through joint co-adaptation.

## Matched schedules

Both rows begin with bitwise-identical Spin-Delta cores and causal routers and
see the same phase-A and phase-B training batches.

**Joint schedule:** 100 phase-A plus 800 phase-B steps optimize retrieval and
auxiliary router losses through the complete model.

**Phase-separated schedule:** 100 phase-A steps optimize only the router on
the auxiliary grammar loss; the untouched core is never evaluated or updated.
The router is then frozen and 800 phase-B steps optimize only the pristine core
on retrieval loss. Evaluation uses token IDs only and the frozen learned
router. No oracle controls are supplied.

This is a schedule intervention. Joint training receives 100 more retrieval
updates, so a phase-separated win cannot be attributed to a larger recurrence
budget. Optimizer state is intentionally separate: the phase-separated core's
AdamW state begins only after routing is frozen.

## Frozen configuration

- same two-key, 32-value overwrite grammar;
- train at 8 writes; evaluate at 8, 16, and 32 writes;
- `d_model=64`, two Spin(8) layers, two heads and two slots;
- unchanged 11,494-parameter width-32 causal hard-ST router;
- phase A: 100 steps; phase B: 800 steps; batch 128;
- 32 evaluation batches per length;
- learning rate `3e-3`, weight decay `0.01`, clip norm `1.0`;
- auxiliary coefficient 1 in the joint row;
- seeds 467, 479, and 487;
- fixed order: joint schedule, phase-separated schedule;
- identical initial tensors and phase-specific batch streams within a seed.

Development may use seed 463 only. Frozen seeds must not be inspected until
the implementation, tests, summarizer, and protocol are committed.

## Frozen decisions

**Phase-separated capacity passes** only if every phased seed reaches at least
95% accuracy at 8 and 16 writes and at least 93% at 32 writes.

**Router readiness passes** only if the frozen phase-A router reaches at least
0.99 on write-event F1, query-event F1, write-slot accuracy, and query-slot
accuracy at every evaluation length in every seed.

**Co-adaptation bottleneck passes** only if phase separation improves 16-write
accuracy over joint training by at least five points in at least two seeds and
by at least five points on the three-seed mean.

The three decisions remain independent. Phase-separated capacity without a
differential win would establish a usable schedule but not prove that joint
co-adaptation caused the earlier failures.

## Non-claims

- Train-time synthetic grammar labels remain privileged supervision.
- The schedule is not parameter matched to the original continuous controller
  and is not a natural-language result.
- A successful curriculum is not a new recurrence, algebraic theorem, speed
  result, or comparison with another model family.
