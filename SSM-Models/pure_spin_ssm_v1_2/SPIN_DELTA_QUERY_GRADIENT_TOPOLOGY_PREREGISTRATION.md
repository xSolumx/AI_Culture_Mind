# Spin-Delta query gradient-topology audit

**Protocol status:** frozen before full-model GPU measurements.

## Question

The label-free cohort exposed a query-event bypass: hard event zero selects the
internal query, while the routed query-slot branch is multiplied by zero. This
audit asks which isolated repair restores first-order query-slot credit with
the smaller initial functional perturbation.

## Compared paths

All paths share bitwise-identical model state, input batch, categorical
straight-through slot decision, internal query, recurrence, and retrieval loss.

1. **Current hard fallback:** hard query event selects routed or internal query.
2. **Soft query event:** replace only the binary query event's hard forward
   value with its sigmoid probability; retain the fallback interpolation.
3. **Authoritative routed query:** force only query event to one, removing the
   fallback at every token; retain the hard categorical routed slot.

Write controls are unchanged in all three full-model paths. No optimizer update
is taken.

## Frozen probes

- an exact float64 local two-slot query equation at event logit -3;
- full Spin-Delta retrieval loss on fresh initialization seeds `691`, `701`,
  and `709`;
- batch 128, two writes, one fixed batch per seed;
- identical two-layer `d_model=64` raw-CUDA model clones;
- gradient norms for router output rows: query event and the two query slots;
- final-token event value and maximum output-logit change from the hard path.

## Frozen decisions

**Dead-path certificate passes** only if the local and every full-model hard
path have zero query-slot gradient (local exactly zero; full at most `1e-12`),
nonzero query-event gradient above `1e-8`, and hard final query event zero.

**Soft restoration passes** only if local and every full-model soft path have
query-slot gradient above `1e-8` and finite outputs/gradients.

**Authoritative restoration passes** under the same query-slot and finiteness
conditions.

If both repairs restore gradients, the audit selects soft continuation only if
its maximum absolute logit perturbation is strictly smaller than the
authoritative perturbation in every seed. It selects authoritative routing only
under the reverse strict ordering. Otherwise no repair is selected.

This is a topology/initialization selector, not a quality promotion. The
selected repair still requires a separately frozen multi-seed training gate.
