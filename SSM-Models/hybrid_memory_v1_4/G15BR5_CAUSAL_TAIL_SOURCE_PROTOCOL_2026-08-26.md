# G15B-R5 causal tail-source decomposition protocol

**Frozen:** 2026-08-26, after sealing G15B-R4 and before evaluating any R5
history-only, current-only, or bias-only intervention metric. The bias control,
background-free authorization rule, and full-transition claim boundary were
added by prospective adversarial review before implementation metrics.

**Entry evidence:**
[`G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md`](G15BR4_OWNERSHIP_BACKGROUND_RESULTS.md)

**Status:** prospective zero-update retained-checkpoint diagnostic

## Question

G15B-R4 shows that the retained identity checkpoints need the injection one
token after each written value. Value-only component reset fails, while
value-plus-`t+1` ownership passes with or without query-time background. That
tail is semantically ambiguous because it sometimes coincides with the marker
that begins the next event.

The one-block G15B shell has a width-four causal depthwise convolution. At a
write tail, its three history taps see the completed preceding
`[write/select marker, key, value]` transaction, while its final tap sees the
current token. R5 asks the smallest next question:

> Is the replaceable tail association already recoverable from the causal
> convolution history alone, without assigning the current token's own
> contribution to the preceding key?

A positive result would identify a locally decidable pending-write/commit
source. A current-only result would instead expose semantic leakage from the
new token. R5 does not train a controller or re-adjudicate R4.

## Bound evidence and runtime

Use the retained G15B identity checkpoints for seeds 2309, 2311, and 2333,
the exact G15B evaluation namespace, 4,096 query decisions per
task/length/seed cell, batch cap 16, and lengths 128, 512, and 1,024. Perform
zero optimizer updates.

Bind the sealed R4 artifact:

- file:
  [`artifacts/g15br4_ownership_background_sm75_2026-08-26.json`](artifacts/g15br4_ownership_background_sm75_2026-08-26.json);
- SHA-256:
  `921d45e3c492e172fae62064120e9e051dca2965bacc44891268b135d8cef26e`;
- required status: clean-start exact-SM75 `quality`, failed overall
  adjudication, passing `vt_lww_bgplus` and `vt_lww_bgminus`, no passing
  value-only arm, and the decision not to train while behavior remains
  tail-dependent.

The R5 quality result must start from a clean commit on CUDA compute capability
`(7,5)`. Parent, checkpoint, protocol, and source hashes are recorded.

## Frozen source decomposition

Let `x_s` be the frozen expanded block input to the local convolution at token
position `s`. Write the exact depthwise-convolution preactivation as

```text
u_s = b + h_s + c_s
```

where `b` is the learned convolution bias, `h_s` is the sum of the three
strict-history tap contributions, and `c_s` is the final/current-token tap
contribution. The ordinary mixed input is `m_s = silu(u_s)`.

Let `Phi(m)` denote only the frozen Spin-Dirac injection construction

```text
Phi(m) = key(m) outer (write(m) * value(m)).
```

At an in-range position `s=t+1` immediately after a valid write-value token
`t`, define:

```text
I_full = Phi(silu(b + h_s + c_s))
I_hist = Phi(silu(b + h_s))
I_curr = Phi(silu(b + c_s))
I_bias = Phi(silu(b)).
```

The ordinary frozen retention, transport, query, output gate, residual path,
and decoder remain unchanged. The alternative source changes injection
ownership only; it does not recompute the step transition from a masked token.
In particular, every arm retains the full-token left/right transition from
`silu(b+h_s+c_s)`. A history result is therefore injection-source sufficiency
conditional on the ordinary full-token transition, not a wholly history-only
tail update.

### Exact residual split

For a source `S` in `{H,C,B}`, assign:

- the full value-token injection at `t` to the written key component;
- `I_hist` (`H`), `I_curr` (`C`), or `I_bias` (`B`) at `t+1` to that key
  component;
- the exact residual `I_full - I_source` at `t+1` to background;
- every other full injection to background.

Thus the source component plus residual background equals the original
erase-free monolithic injection at every token. No-reset complete-state sums
must reconstruct the ordinary erase-free recurrence. At every valid write,
last-write-wins resets the selected key component before the current value
injection exactly as R3/R4; the residual background is never reset.

The signed residual is an attribution device, not a proposed standalone
bounded outer-product write. A training architecture is not authorized merely
because this algebra is valid.

`H`, `C`, and `B` are not additive or exclusive sources: all include the
convolution bias, and SiLU plus `Phi` are nonlinear. The bias arm is therefore
a required non-authorizing control. A history claim must beat the
background-matched bias arm rather than receiving credit for bias alone.

### Local timing

The tail mask must be decoded from tokens alone: at position `s`, the three
preceding tokens must form a valid local `[write/select marker, payload,
payload]` transaction. It must exactly equal the in-range
`write_positions + 1` audit mask. No collision history or query target is
available to the source split.

### Query-time background arms

As in R4:

- `BG+` reads the sum of key components and residual background everywhere;
- `BG-` excludes residual background only at locally observable query-key
  positions and otherwise reads the complete sum.

`BG-` does not select the queried key component.

Score:

1. `h_lww_bgplus`;
2. `h_lww_bgminus`;
3. `c_lww_bgplus`;
4. `c_lww_bgminus`;
5. `b_lww_bgplus`;
6. `b_lww_bgminus`.

Controls are `learned`, a shared `erase_free_no_reset_bgplus`,
`h_no_reset_bgminus`, `c_no_reset_bgminus`, and `b_no_reset_bgminus`.
Current-only and bias-only arms are diagnostic controls and can never authorize
training. Sealed R4 `V` and `VT` metrics remain bound references; they are not
retrospectively changed.

`BG+` is residual-coupled: it continues to read `I_full-I_source`, which can
contain current-token and nonlinear interaction information. Only a passing
`h_lww_bgminus` arm demonstrates history-derived injection sufficiency without
residual-background read support.

## Cohorts and query strata

Evaluate MQAR, ordinary overwrite, the deterministic R3/R4 overwrite guard,
selective copy, and needle at every registered length. For ordinary overwrite
and the guard, report:

1. `before_any_overwrite`;
2. `after_unrelated_overwrite_only`;
3. `after_same_key_overwrite`.

The ordinary generator may leave the unrelated-only stratum empty; report zero
support and make no ordinary-cohort claim for it. The constructed guard must
populate all three strata at every length or adjudication fails closed.

## Integrity

- reproduce sealed R4 batch fingerprints and its `learned` and
  `erase_free_no_reset_bgplus` query metrics within `1e-12`;
- verify the sealed R4 parent SHA, failed decision, and exact passing-arm lists;
- ordinary model forward and the `learned` arm are bit-identical;
- assert the exact one-block Spin-Dirac shell and depthwise convolution with
  kernel four, stride/dilation one, and `groups == expanded width`;
- reconstruct direct local-convolution preactivations from bias plus history
  and current contributions under both full-sequence and arbitrary-chunk
  execution; include per-tap impulse tests so reversed kernel indexing fails;
- prove `u_full = b + h + c` and the exact residual identity
  `I_source + (I_full - I_source) = I_full` under an independent FP64
  contract with residual at most `1e-10`;
- perturb the current expanded input while holding the preceding completed
  write fixed and require history preactivation and `I_hist` to remain
  invariant, while current preactivation and `I_curr` change nontrivially;
- perturb the preceding completed-write expanded inputs while holding the
  current expanded input fixed and require current preactivation and `I_curr`
  to remain invariant, while history preactivation and `I_hist` change
  nontrivially;
- verify future-token perturbations cannot alter earlier tail masks or source
  terms;
- independently compute `I_bias` and fail closed if source perturbation effects
  are numerically degenerate;
- ownership is exclusive except for the explicit additive source/residual
  split, whose sum is complete at every token;
- reset affects exactly one logical component per valid write and occurs before
  the current value injection;
- verify value-token injection remains full and unchanged, and full left/right
  transitions plus every non-injection control are bit-identical across all
  source arms;
- LWW and matching no-reset predictions are identical on no-overwrite tasks
  and at all prefixes before a first same-key overwrite;
- for each source, `BG+ read - BG- read` equals the residual-background read at
  locally observable query positions;
- FP64 monolithic, decomposed, recurrent, and parallel states/reads have
  maximum residual at most `1e-10`;
- FP32 scored paths remain finite, preserve required replay predictions, and
  leave all non-injection controls bitwise unchanged;
- local write, tail, and query decoders, task fingerprints, parent/source
  hashes, clean SM75 provenance, and zero updates must pass.

Continue to report final-token writes, state expansion, source and residual
norms and source/bias cosines by filler, write-marker, and item-marker tail
role, and wall time. R5 is not state-, parameter-, compute-, or time-matched.

## Frozen decision gates

An individual LWW arm passes only if every length satisfies all applicable
conditions below.

### Ordinary overwrite

- aggregate LWW accuracy improves at least `0.10` over `learned` and its
  source/background-matched no-reset control;
- `after_same_key_overwrite` improves at least `0.10` over both controls;
- within every seed, aggregate and post-same-key accuracy improve at least
  `0.05` over both controls;
- `before_any_overwrite` trails `learned` and its matching no-reset control by
  no more than `0.02`.

### Constructed guard

- aggregate and post-same-key accuracy are at least `0.995`;
- aggregate and post-same-key trail `learned` by no more than `0.005`;
- aggregate and post-same-key improve at least `0.10` over the matching
  no-reset control;
- before-any and after-unrelated-only accuracy are at least `0.98` and trail
  `learned` and the matching no-reset control by no more than `0.02`;
- all three guard strata have nonzero support.

### Safety and integrity

- MQAR and selective copy trail `learned` by no more than `0.02`;
- needle accuracy is at least `0.999`;
- every integrity gate passes.

In addition, a history arm must improve over its background-matched bias LWW
arm by at least `0.05` in three-seed mean ordinary aggregate and post-same-key
accuracy at every length, and by at least `0.02` within every seed on both
measures. This source-specific gate does not apply to current- or bias-only
controls.

Fresh pending-write/commit training is warranted only if
`h_lww_bgminus` passes every general and bias-separation gate:

- if `h_lww_bgminus` passes and `h_lww_bgplus` fails, screen a causal
  transaction path with a protected transaction read;
- if both history arms pass, history injection is sufficient without
  background, while the shared-residual arm is also viable; choose between
  them only in a separately frozen matched training protocol;
- if only `h_lww_bgplus` passes, record residual-coupled history evidence but
  authorize no training;
- if only current-only arms pass, reject this checkpoint-repair route because
  the apparent association depends on the new token;
- if the bias arm matches history or no background-free history arm passes,
  stop retained-checkpoint tail repair and do
  not train the present shell;
- if both history- and current-only background-free arms pass, report
  non-unique injection-source sufficiency; do not call history the unique
  source.

Any authorized training requires a separate frozen protocol, fresh seeds and
data, learned non-oracle addresses, explicit occupancy/transaction state, and
parameter/state/compute-matched controls. It is a training screen, not model
promotion. G15C and external-loss-only training remain blocked.

## Exact reproduction target

The implementation must expose smoke and evidentiary quality modes:

```bash
PYTHONPATH=SSM-Models /home/local/.venvs/sm75-native-2026/bin/python \
  -m hybrid_memory_v1_4.g15br5_causal_tail_source \
  --mode quality --device cuda \
  --checkpoint-directory /home/local/g15b_bd5045a_quality_attempt2_checkpoints \
  --output /home/local/g15br5_<commit>_quality.json
```

Quality must fail closed on a dirty tree, non-CUDA execution, non-SM75
hardware, missing parent/checkpoint hashes, absent source hashes, or incomplete
integrity/support.

## Nonclaims

R5 remains zero-update, retained-checkpoint, commissioned-task attribution with
oracle logical-key components. A background-free history pass would establish
causal injection-source sufficiency conditional on the ordinary current-token-
dependent transition; it would not establish a wholly history-only update or autonomous
transaction learning, generic association, ordinary-text recall, longer-
context scaling, efficiency, optimizer or tokenizer superiority, Spin
benefit, G15C, or model-family promotion. A failure does not falsify GDN2,
KDA, dual-address edits, or an explicit transaction memory trained from
scratch.
