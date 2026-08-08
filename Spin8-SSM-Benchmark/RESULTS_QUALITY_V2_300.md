# Local Spinor quality iteration

Run artifact: [`results/quality_v2_300.json`](results/quality_v2_300.json)

Artifact SHA-256: `e42f2fec152a29a68dcc88608920ea460cdd39f068c0b542a54b82c15b807ee8`

This run keeps the same model width, parameter count, optimizer, byte-level
WikiText-2 data, sequence length, 300 updates, and two seeds as the earlier
corrected local baseline. The only quality-path change is removing the
unjustified `1/sqrt(decoder_width)` divisor from the untied decoder logits;
Mamba-style untied language heads do not use that cosine-similarity scaling.

| Local model | Parameters | Validation bits/byte | Throughput tokens/s |
| --- | ---: | ---: | ---: |
| Earlier decoder-scaled Spinor | 674,322 | 3.060 ± 0.006 | 21,618 ± 317 |
| Spinor v2, decoder scale = 1 | 674,322 | **2.765 ± 0.006** | **23,170 ± 131** |

The v2 change improves validation by `0.295 bits/byte` (9.6%) and increases
measured throughput by about 7.2% on the RTX 2070 SUPER. This is a real matched
local ablation, not a parameter-count increase. It does not establish parity
with Mamba-2 or Mamba-3; those comparisons require rerunning the full three-way
benchmark with this v2 configuration.

## Rejected default ablation

An identity-initialized depthwise causal mixer was implemented as
`DepthwiseCausalMix`. It is equivariant because the same temporal filter is
applied to every Cl(3) coordinate, and it preserves chunk/stream parity. The
40-step screen was worse than the no-mixer control (3.918 versus 3.782
bits/byte), so it remains an explicit ablation rather than being silently
included in v2.

A zero-initialized coordinate-sensitive direct decoder ensemble was also
screened at 16, 20, and 32 bottleneck channels. None beat the v2 control in the
same 40-step screen, so it remains opt-in (`direct_decoder_channels`) and is not
part of the promoted quality result.

## Boundary

The local research audit says the geometric recurrence has not demonstrated a
language-quality advantage, and the newer Spin(8) results explicitly leave
global superiority over modern SSM baselines open. This iteration therefore
promotes only the decoder-scaling improvement. The stable rotor-affine scan,
equivariance, bounded-write contract, and streaming parity remain unchanged.

## Evidence validation

`validate_quality.py` replays the report-level acceptance contract: it requires
unique seeds and post-seed construction, checks both dataset digests, verifies
parameter/config closure, and recomputes perplexity, bits/byte, and throughput
from the stored losses and timing. The saved two-seed artifact passes this
validator; the model/scan test module passes 11 direct checks.
