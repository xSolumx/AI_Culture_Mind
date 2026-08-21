# Natural-data comparison

**Status:** historical initial-backend comparison; superseded for throughput

The model/data conclusion remains valid, but the hardware conclusion below is
superseded by [`FRONTIER_TRAINING_RESULTS.md`](FRONTIER_TRAINING_RESULTS.md).
The raw coordinate backend plus exact nested-group ladder now measures 80,697
tokens/s across three seeds versus 87,991 for fused Mamba-2, rather than the
4.87x gap recorded here.

**Dataset:** raw UTF-8 bytes from `Salesforce/wikitext`, configuration
`wikitext-2-raw-v1`

**Artifact:**
[`artifacts/wikitext2_byte_3seed_summary.json`](artifacts/wikitext2_byte_3seed_summary.json)

**Canonical Git-blob SHA-256:**
`d2ff430dc55d3a65231f381de90b95278f4b345b58950384aaf5582db31e6008`

## Protocol

Both models trained for 300 optimizer steps on the same deterministic byte
batches at batch size 8 and sequence length 256. Both used AdamW, learning rate
`3e-3`, weight decay `0.01`, gradient clipping at `1.0`, identical validation
batches, and seeds 17, 29, and 43. Kernel compilation was warmed up outside the
training timer.

Pure Spin v1.2 has 641,996 trainable parameters; fused Mamba-2 has 623,740, a
2.84% gap relative to the larger model. The baseline is the official
`mamba_ssm.Mamba2` memory-efficient fused path, not a Transformers reference
fallback. The run used an RTX 2070 SUPER with Torch 2.10.0+cu130, Triton 3.6.0,
`mamba_ssm` 2.3.2.post1, and `causal-conv1d` 1.7.0.

## Results

| seed | Pure Spin bits/byte | Mamba-2 bits/byte | Pure Spin tokens/s | Mamba-2 tokens/s | Mamba/Spin speed |
|---:|---:|---:|---:|---:|---:|
| 17 | 2.718 | 2.500 | 15,029 | 74,279 | 4.94x |
| 29 | 2.739 | 2.470 | 14,892 | 63,908 | 4.29x |
| 43 | 2.738 | 2.500 | 15,039 | 81,031 | 5.39x |
| mean | **2.732** | **2.490** | **14,987** | **73,073** | **4.87x** |

Across these three seeds, Mamba-2's mean validation result was better by 0.242
bits/byte. Its mean measured training throughput was 4.87 times higher. The
direction of both differences was consistent in all three runs. These are
short-budget byte-language-model results, not standard word-level WikiText
perplexities and not a scaling-law claim.

## What was learned

Pure Spin v1.2 does learn natural text: mean validation loss fell from roughly
8 bits/byte at initialization to 2.732 bits/byte. The result nevertheless
falsifies the immediate claim that the current controller/action construction
is competitive with a mature fused Mamba-2 implementation at this scale and
training budget.

At the time of this artifact, the remaining hardware opportunity was sharply
localized. The raw CUDA
materialized recurrence can beat the maintained Triton scalar recurrence at the
recorded shape, but v1.2 training still constructs and differentiates 28 plane
factors inside every layer. The later implementation showed that a GEMM
controller plus raw coordinate-factor recurrence was faster than placing the
controller dot products inside the low-occupancy recurrent warp.

The cache audit identifies a separate advantage worth preserving. Official
Mamba-2 allocates 80,384 streaming-state scalars per sequence for this matched
model. Pure Spin's Spin recurrence uses 192; including the 1,536-scalar FIFO
needed by its depthwise convolutions gives a 1,728-scalar complete streaming
design, 46.5 times smaller. The wrapper does not yet implement that incremental
convolution API, so 1,728 is a design count rather than a completed runtime
claim. See [`artifacts/cache_audit.json`](artifacts/cache_audit.json).
Its SHA-256 is
`79a324399a70c4f8bc1b4c972b8d607e039256bd59ef0c84edee3f01364de4bf`.

Development smoke and ten-step pilot artifacts are retained only as provenance
for initialization, compilation-warmup, and dataset-hash checks. They are not
promoted benchmark results.
