# Native SM75 runtime and baseline ledger

**Machine:** NVIDIA GeForce RTX 2070 SUPER, compute capability 7.5, 8 GB
**Policy:** source-current native execution only; no eager or architectural
fallback can qualify a fused baseline

The working copies live in the WSL Linux filesystem to avoid `/mnt/c` build
I/O. The repository stores probe code and small JSON evidence, not downloaded
weights, source clones, virtual environments, or compiled extensions.

## Source checkouts

| Implementation | Source revision | Role |
|---|---|---|
| official `state-spaces/mamba` | `e9594ce1c732d97440f0332fdc43170a2294dbfa` | Mamba-3 layer and Mamba checkpoint execution |
| `fla-org/flash-linear-attention` | `4b31cc8419a646a432951b44cf2e16ae5aca1949` | maintained GDN2 layer |
| `Dao-AILab/causal-conv1d` | `cd81f0413cad2fc1e6f17e785ac39f59aae690cd` | native Mamba checkpoint dependency |
| `ssiu/flash-attention-turing` | `52a67d7c8d85d2363d3aa2bf491898f1e826ea69` | Turing-compatible FlashAttention |
| `NVlabs/GatedDeltaNet-2` | `95709fc250357c2dd109361c353192f2aa5913f9` | upstream semantic reference; current FLA API drift prevented a clean standalone execution |

Each passing native artifact verifies PEP 610 `direct_url.json` against the
supplied source root and records the imported module or extension path. Merely
having a same-named wheel installed is insufficient.

## Environments

| Environment | Python / Torch / CUDA | Installed source packages |
|---|---|---|
| `/home/local/.venvs/sm75-native-2026` | 3.11 / 2.9.0+cu128 / 12.8 | FLA 0.5.2, Mamba 2.3.2.post1, causal-conv1d 1.7.0 |
| `/home/local/.venvs/sm75-mamba3-py310` | 3.10 / 2.9.0+cu128 / 12.8 | Mamba 2.3.2.post1, TileLang |
| `/home/local/.venvs/sm75-flash-turing` | 3.11 / 2.9.0+cu128 / 12.8 | source-built `flash-attn-turing` extension |

## Qualification result

| Backend | Result on SM75 | Eligibility boundary |
|---|---|---|
| Mamba-3 SISO | pass | native training forward/backward; complete finite input and all-parameter gradients |
| Mamba-3 MIMO | fail | TileLang selects an SM80 TF32 MMA path; no fallback and no SM75 baseline |
| FLA GDN2 fused recurrent | fail training qualification | finite output/input gradient, but recurrent-core parameter gradients are absent; inference mode is not a training baseline |
| FLA GDN2 chunk | fail bounded qualification | native forward compiled and ran; backward exceeded the bounded 1,200-second attempt; no result substitution |
| Turing FlashAttention | pass | native FP16 forward/backward at head dimensions 64/128, causal/noncausal, with max output error `4.8828125e-4` and max gradient error `9.765625e-4` against SDPA |
| official FlashAttention 2/3 mainline | ineligible | current official kernels require Ampere or newer; the upstream project points Turing users to the dedicated fork |

Passing here means the actual source-bound implementation executed on
`sm_75`. First-call times include compilation and are not throughput results.
Failing here means the backend is excluded; it says nothing about the paper's
algorithm on supported hardware.

## Actual pretrained checkpoints

The Hugging Face CLI downloaded real weights into the WSL Linux filesystem.
The probe binds Hub revision manifests, weight hashes, tokenizer revision
manifests, tokenizer-file hashes, source installation, device, and finite
natural-text loss. The weights are not committed.

| Checkpoint | Hub revision | Tokenizer provenance | SM75 result |
|---|---|---|---|
| `state-spaces/mamba3-siso-187m` | `6792c27c00f3bb41506db1066dcd1c51bb0f4b02` | checkpoint declares Llama 3.1; gated Meta repository was unavailable, so the public `NousResearch/Meta-Llama-3.1-8B` mirror at `1f47e50c...` is named explicitly | pass: 186,849,600 parameters, finite FP16 logits/loss |
| `state-spaces/mamba2-130m` | `3a5aea0c25d0fb43cc360e2c2aac82c26e3eed49` | `EleutherAI/gpt-neox-20b` at `c292233c...` | pass: 128,989,632 parameters, finite FP16 logits/loss |

The one fixed natural-text loss is only an execution/finiteness probe.
Different tokenizers and pretraining corpora make these losses ineligible for
a model-quality ranking.

## Bound G15B checkpoint diagnostics

The G15B-R1 zero-update checkpoint diagnostic also ran on this exact SM75
runtime from clean commit `dba3f9a`. It binds the completed G15B and G15B-R0
artifact hashes, replays all baseline accuracy/episode/BPQ cells and ordinary
model logits exactly, and records bitwise preservation of all non-erase
controls. The run completed in 1,726.8 seconds and rejected erase at every
valid write. See the
[`result`](G15BR1_EVENT_ERASE_RESULTS.md) and
[`artifact`](artifacts/g15br1_event_erase_sm75_2026-08-26.json), SHA-256
`c015b128846e4b5c63d927778815a87728a7d613369163b1027ed3dd9f0b2912`.

G15B-R2 completed on the same runtime from clean commit `5eae963`, preserving
the full learned write program while applying erase only at true collisions.
It passed all provenance and runtime-integrity checks but rejected symmetric
erase on post-same-key-overwrite recall. See the
[`result`](G15BR2_COLLISION_ERASE_RESULTS.md) and
[`artifact`](artifacts/g15br2_collision_erase_sm75_2026-08-26.json), SHA-256
`90652fe7034e5901b968eb5d139f02eb8bc714b0417c0889e16a2fdd6b7cf924`.

## Reproduction

Run [`native_sm75_probe.py`](native_sm75_probe.py) with an explicit backend,
source root, and artifact path. Run
[`pretrained_sm75_probe.py`](pretrained_sm75_probe.py) with explicit model and
tokenizer Hub revisions, local directories, and the official Mamba source
root. Both scripts require exactly compute capability `(7, 5)` and fail
closed. See the JSON files named `native_sm75_*` and `pretrained_sm75_*` in
[`artifacts/`](artifacts/).
