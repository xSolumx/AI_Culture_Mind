# Optimizer, tokenizer, and present learning problem

**Completed:** 2026-08-25  
**Model architecture:** retention-safe Hybrid Memory v1.4.5  
**Training frontier:** G12A--G12E

## Direct answer

AdamW is a good general baseline, but it is not the best tested optimizer for
this model. The selected optimizer is `HarmonicMuonAdamW`:

- PyTorch Muon with `adjust_lr_fn="match_rms_adamw"` for eligible 2D hidden
  matrices;
- a custom AdamW variant with one scalar second moment per tensor for the write,
  decay, and residual controllers; and
- ordinary AdamW for tied embeddings, normalizers, biases, and convolutional
  tensors.

It is not 10,000 times better. On the valid paired development screen it lowers
final bits per raw byte (BPRB) by 0.1205 and 0.0948 on two seeds, while median
update time rises about 3.9%. On three fresh seeds, the raw-byte composite beats
raw-byte AdamW by 0.0667, 0.0671, and 0.0442 BPRB.

Raw UTF-8 bytes remain the clean falsifier tokenizer, but they are not the best
tested compute allocation. The selected fitted tokenizer is a lossless
training-only Hugging Face ByteLevel BPE with 512 tokens. It has no unknown or
special token, round-trips both splits exactly, and represents 2.302 training
bytes and 2.337 validation bytes per token. All cross-tokenizer losses are
reported as BPRB, never token loss alone.

## Why a blanket AdamW swap was rejected

The repository's strongest optimizer result is exact, not a trend:

- positive-half-Spin(8) and generic SO(8) action coordinates are related by an
  orthogonal 28x28 map;
- plain SGD preserves the mapped coefficients and logits to about `1e-16`;
- AdamW's coordinatewise second moments produce coefficient error `0.0375`,
  action error `0.0377`, and logit error `0.152` after 12 updates.

This proves a lack of coordinate covariance for that AdamW update. It does not
prove that SGD trains language models better. The prior audit therefore called
for scalar/tensorwise moments only on the affected tensors rather than a global
optimizer replacement. G12 follows that instruction.

For v1.4.5, the auditable composite partition is:

| Role | Optimizer | Tensors | Parameters | Second moment | Weight decay |
|---|---|---:|---:|---|---:|
| hidden matrices | PyTorch Muon | 14 | 102,400 | orthogonalized momentum update | 0.01 |
| memory controls | custom scalar AdamW | 6 | 522 | one scalar/tensor | 0 |
| depthwise convolution | AdamW | 1 | 256 | coordinatewise | 0.01 |
| embedding/norm/bias | AdamW | 8 | 16,784 | coordinatewise | 0 |

The custom scalar update has a float64 test showing exact covariance under a
random orthogonal coordinate transform through eight updates. The complete
partition is disjoint, covers every trainable parameter, checkpoints, and is
recorded in every G12 run.

## What the older models teach

- Pure Spin v1.2, Pure Rotor v2.1, Pure F4 v1.3, and the maintained v1.4
  natural-text screens overwhelmingly used AdamW. Their negative results do not
  isolate AdamW, but they make another unmeasured global AdamW run low value.
- Pure Spin's exact chart audit supplies the covariance falsifier that motivates
  scalar moments and Muon.
- Pure Rotor's five-seed transport ablation shows why parameter-, state-, and
  measured-CUDA-matched views must be separate. More exotic transport lost the
  measured-compute view even where a small prediction advantage existed.
- Pure F4's determinant shortcut changed float32 optimization enough to fail
  its prospective quality rule despite algebraic correctness. Evaluation order
  and optimizer trajectory are part of the empirical model.
- The older maintained language experiments mostly use raw UTF-8 bytes. Some
  historical GA-SSM scripts use GPT-2/tiktoken, but they do not supply a frozen,
  matched tokenizer result that can override G12.

## G12 natural-text result

All three G12C seeds use ordinary TinyStories next-token cross entropy only.

| Arm | Parameters | Mean final BPRB | Worst BPRB | Mean median update | Presented raw bytes |
|---|---:|---:|---:|---:|---:|
| raw bytes + AdamW | 119,962 | 1.8072 | 1.8204 | 108.96 ms | 4,096,000 |
| raw bytes + composite | 119,962 | 1.7478 | 1.7537 | 113.55 ms | 4,096,000 |
| parameter-matched BPE + composite | 124,534 | **1.5344** | **1.5446** | **79.93 ms** | 9,434,111 mean |
| CUDA-matched BPE + composite | 111,770 | 1.5498 | 1.5625 | 113.43 ms | 9,434,111 mean |

The parameter-matched BPE model is width 48 / FFN expansion 5 and is 3.81%
above the raw parameter target. The timing-only CUDA calibration selects width
64 / expansion 1: its calibration median is 110.72 ms versus 106.80 ms for the
fresh raw target (+3.67%), and its outcome update time is +4.10% versus the G12C
raw control. Both BPE views see about 2.30 times as many original bytes for the
same number of token targets. Therefore:

- the raw optimizer comparison isolates the optimizer;
- the parameter-matched BPE comparison changes original-byte exposure and
  compute;
- the CUDA-matched BPE comparison is a hardware-specific resource allocation;
- none is an architecture-only or universal tokenizer win.

Actual upstream comparisons are retained separately. G11 trained real
Transformers `Mamba2ForCausalLM` and `OlmoHybridForCausalLM` classes, not local
lookalikes; they reached 1.639 and 1.675 BPC on the one-seed raw-byte screen.
The actual pretrained `state-spaces/mamba2-130m` checkpoint was also downloaded
and executed in WSL. Those probes have different parameter, tokenizer, and
runtime boundaries and are not silently merged into G12.

## The present learning problem

The model now learns ordinary local text robustly. It does not robustly learn
long-range factual binding from this training distribution.

After ordinary length-256-token pretraining, all nine G12C models execute
finite held-out evaluations through 1,024 tokens. BPE improves from 1.5505 to
1.5075 mean BPRB as evaluated token context grows. Raw/composite stays roughly
flat (1.7542 to 1.7516), while raw/AdamW degrades (1.8175 to 1.8734).

The paired counterfactual recall probe nevertheless fails as a capability:
matching-prefix gains are tiny, change sign across seeds, and are not monotone
with distance. The BPE arm's all-positive seed means at 1,024 raw bytes average
only `0.0033` nats. That is not credible evidence of learned recall.

G13 then removed the training-horizon objection with an exact-target
`256 -> 512 -> 1,024 -> 2,048 -> 4,096` curriculum. It improves 4,096-token
ordinary BPRB in all three seeds, by 0.0126 on average, but misses the frozen
0.02 gate. At 8,192 raw bytes, all prompts are longer than the 2,048-token
attention window and counterfactual recall averages only `0.0000108` nats with
one negative seed. Longer training context helps compression; it does not solve
one-shot binding.

The failure is identification plus fast-weight overwrite, not global state
instability:

1. G12's 256-token horizon withheld long-range gradients; G13 supplies them,
   but the ordinary corpus still contains too little query-aligned pressure to
   identify protected one-shot binding;
2. TinyStories next-token entropy is dominated by local syntax and common word
   continuation, so optimization can improve BPRB without learning arbitrary
   fact storage;
3. retention near 0.999 protects directions not being rewritten, but G13's
   learned mean write strengths are 0.446--0.768 per token and contract the
   currently written direction to 0.553/0.232/0.329/0.382 per head;
4. ByteLevel BPE expands raw-text coverage, but compression alone does not
   create a binding/query learning signal.

## Best-supported next strategy

Keep the composite optimizer and lossless 512-token ByteLevel BPE. Preserve the
current Gated Delta layer as fast working memory, because disabling it worsens
G13 4,096-token loss by 1.3921 BPRB. Do not ask that same high-plasticity matrix
to double as a protected archive. The next causal intervention should:

1. add a separate slow memory timescale with sparse learned admission and
   protected consolidation;
2. derive delayed binding/span targets from natural text so the slow write and
   query policy are identifiable;
3. retain ordinary next-token loss and the fixed v1.4.5 arm as quality and
   compute controls;
4. reuse G13's exact-target curriculum and frozen counterfactual recall
   falsifier; and
5. name the new auxiliary honestly as commissioned self-supervised memory
   training, not ordinary pretraining.

This is the lesson shared by the older SSM programmes: capacity and nonzero
gradients are insufficient. Learning succeeds when the training horizon and
observable objective identify the intended recurrent mechanism. More Spin,
F4, rotor transport, retention, or optimizer sophistication cannot substitute
for missing long-range evidence.

## Evidence

- [`G12_PREREGISTRATION.md`](G12_PREREGISTRATION.md)
- [`G12E_PREREGISTRATION.md`](G12E_PREREGISTRATION.md)
- [`G13_PREREGISTRATION.md`](G13_PREREGISTRATION.md)
- [`artifacts/g12a_tokenizer_audit_2026-08-25.json`](artifacts/g12a_tokenizer_audit_2026-08-25.json)
- [`artifacts/g12b_optimizer_development_cuda_2026-08-25.json`](artifacts/g12b_optimizer_development_cuda_2026-08-25.json)
- [`artifacts/g12c_multiseed_natural_text_cuda_2026-08-25.json`](artifacts/g12c_multiseed_natural_text_cuda_2026-08-25.json)
- [`artifacts/g12d_post_pretraining_long_context_recall_cuda_2026-08-25.json`](artifacts/g12d_post_pretraining_long_context_recall_cuda_2026-08-25.json)
- [`artifacts/g12e_compute_matched_frontier_cuda_2026-08-25.json`](artifacts/g12e_compute_matched_frontier_cuda_2026-08-25.json)
- [`artifacts/g13_exact_target_long_context_curriculum_cuda_2026-08-25.json`](artifacts/g13_exact_target_long_context_curriculum_cuda_2026-08-25.json)
- [`artifacts/g13_posthoc_long_context_diagnostic_cuda_2026-08-25.json`](artifacts/g13_posthoc_long_context_diagnostic_cuda_2026-08-25.json)
- [`../experiments/SPIN8_SO8_OPTIMIZER_EQUIVARIANCE_RESULTS.md`](../experiments/SPIN8_SO8_OPTIMIZER_EQUIVARIANCE_RESULTS.md)
- [`../experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md`](../experiments/PURE_V2_1_TRANSPORT_ABLATION_RESULTS.md)

The Muon implementation is the actual `torch.optim.Muon` shipped by PyTorch
2.12. The BPE implementation is the actual Hugging Face `tokenizers` 0.22.2
ByteLevel/BPE stack.
