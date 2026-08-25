# G15B interleaved commissioned-controller results

**Executed:** 2026-08-25--26 | **Start commit:** `bd5045a07123c73d3ada7a3934014aa699d22719` | **Start status:** clean | **Runtime:** WSL2, RTX 2070 SUPER, SM75, PyTorch 2.9.0+cu128

**Artifact:** [`g15b_interleaved_controller_sm75_2026-08-26.json`](artifacts/g15b_interleaved_controller_sm75_2026-08-26.json)

**Artifact SHA-256:** `f74d860e30ab40ec747521dfcecd74aac2bb75151206c25b7104d334727429eb`

## Bottom line

G15B fails its prospectively frozen commissioned-controller gate. G15C and the
external-loss-only lane remain blocked.

This is nevertheless a useful failure rather than a null result. The identity
arm learns nearly exact unique-key addressing, causally uses its recurrent
memory, reaches about 97% mean MQAR/selective accuracy, and solves exact needles
through distance 960. It does not learn reliable last-write-wins editing:
overwrite accuracy remains about 77--83%, while overwrite-erase recall is only
about 51%. Full Spin is worse than identity in every MQAR, overwrite, and
selective mean cell. The commuting arm is worse again. Only the needle cells
satisfy the frozen Spin noninferiority test.

The main failure is a temporal-observability error in the commissioned target.
The erase controller sees the current token representation and a width-four
causal convolution, but its label asks whether the current key was written at
any earlier time. That collision bit is not available to the controller. A
last-write-wins fast-weight law does not need this classifier: erasing the
address on every valid write is harmless for a first write and necessary for
an overwrite.

The transport result is also decisive. Token-wise rotations move stored
addresses through an accumulated frame, while later queries are produced in a
local token frame with no access to that accumulated action. Generic content
addressing therefore prefers identity transport. Spin transport should remain
a specialized supplied/coherent-frame mechanism unless query, key, and state
are transported in one consistent frame.

## Frozen execution and integrity

All execution gates pass:

- exact FP64 state-law replay and oracle direct reads pass every task and arm;
- recurrent, parallel, arbitrary-chunk, token-step, and complete-control paths
  agree within the frozen tolerance;
- all required activation gradients and optimizer partitions pass;
- all nine arm/seed cells complete 4,200 updates;
- every arm has 57,949 parameters and 1,792 FP32 streaming-state bytes per
  sequence;
- every seed/arm receives 375,360 scored training decisions, 518,720 valid
  writes, and 184,320 overwrites;
- paired arm schedule hashes agree within each seed;
- every train/evaluation fingerprint intersection is empty;
- the run starts from a clean commit on the bound SM75 runtime.

The complete cohort takes 14,280.8 seconds. Peak allocated CUDA memory is
4.121--4.129 GB across the nine cells. The earlier metric-blind attempt was
stopped before a checkpoint or output existed solely to replace an unsafe,
underutilized evaluation batch; it has no result.

## Retrieval result

Three-seed mean query accuracy:

| Task and length | Identity | Commuting `SO(2)^4` | Full Spin(8) |
|---|---:|---:|---:|
| MQAR 128 | **0.9724** | 0.8178 | 0.9464 |
| MQAR 512 | **0.9729** | 0.8169 | 0.9337 |
| MQAR 1,024 | **0.9718** | 0.8125 | 0.9076 |
| overwrite 128 | **0.7678** | 0.4169 | 0.6903 |
| overwrite 512 | **0.8328** | 0.4573 | 0.7201 |
| overwrite 1,024 | **0.8288** | 0.4631 | 0.7285 |
| selective 128 | **0.9719** | 0.7597 | 0.9235 |
| selective 512 | **0.9770** | 0.7686 | 0.9206 |
| selective 1,024 | **0.9752** | 0.7710 | 0.8775 |
| needle 128, distance 64 | 1.0000 | 1.0000 | 1.0000 |
| needle 512, distance 448 | 1.0000 | 1.0000 | 1.0000 |
| needle 1,024, distance 960 | **1.0000** | **1.0000** | 0.9998 |

Identity is the generic associative-memory reference. Spin trails identity by
3.4--8.8 points on MQAR, 6.1--17.2 points on overwrite, and 1.0--12.3 points on
selective copy at the individual-seed level. The full-Spin three-seed mean
fails noninferiority in all nine non-needle cells. Commuting transport drives
its coordinate RMS to about 0.0945 and repeatedly reaches the `0.25` coordinate
bound; full Spin uses smaller RMS coordinates, about 0.009--0.015, but still
degrades generic retrieval.

## What learned, and what did not

### Addressing learned

The address controller is not the bottleneck:

- address top-1 is at least 0.99976 for identity and 0.99927 for full Spin;
- the commuting minimum is still 0.9880;
- same-key overwrite address consistency is about 0.97--0.99;
- removing memory or writes drops identity accuracy by a mean 0.938 and full
  Spin by 0.877;
- substituting a wrong live query drops identity by a mean 0.920 and full Spin
  by 0.871.

Successful retrieval is therefore genuinely content-addressed and causally
memory-dependent. This is stronger than a final-score-only result and rules
out the old residual/decoder bypass as the primary explanation.

### Edit timing did not learn

Across overwrite cells, mean erase recall is approximately:

| Arm | Erase recall | Query accuracy |
|---|---:|---:|
| Identity | 0.506 | 0.810 |
| Commuting | 0.538 | 0.446 |
| Full Spin | 0.479 | 0.713 |

Write recall is 1.0, but write F1 is about 0.79 for identity/commuting and 0.94
for full Spin because gates spill onto non-write positions. The frozen 0.98
write-F1 gate therefore fails. More importantly, erase recall remains near the
uninformative collision prior despite direct erase supervision. No optimizer
can recover history that is absent from the controller input.

### The frozen learned-path oracle is gauge-mismatched

The direct-memory oracle preflight passes every task and arm exactly below
`1e-10` residual. The later `oracle` intervention replaces learned addresses
with one-hot basis addresses while leaving learned values, the Clifford second
read sector, and the output decoder fixed. Its low accuracy therefore mixes
controller correction with a change of learned address gauge. It is not a
clean capacity ceiling and should not remain a binding gate in this form.

This protocol defect does not rescue G15B: overwrite retrieval, erase recall,
write F1, and Spin noninferiority independently fail. The replacement should
preserve learned key prototypes while correcting event timing, or score the
direct state read separately as the existing preflight already does.

## Architectural diagnosis

The present recurrence forms query, key, value, erase, write, retention, and
transport coordinates from the block input before the recurrent transition.
With one block, that input is token embedding plus a width-four causal
convolution. Consequently:

1. a local marker can identify a valid write;
2. a key projection can identify the address;
3. the controller cannot know whether the address is already occupied;
4. a token-wise transport can rotate stored addresses, but the future query
   does not know the accumulated frame product.

The model learned exactly what is observable: address identity, write presence,
and memory-dependent reads. It failed what is not observable or not
frame-consistent: collision-conditioned erase and generic moving-frame recall.

## Bound next move: G15B-R

The least-intervention repair is architectural, not an optimizer sweep:

1. Use identity transport as the generic associative-memory default.
2. For last-write-wins tasks, apply addressed erase on every valid write. Tie
   erase to the write event or use a delta-correction coefficient; do not ask a
   token-local controller to classify whether the key was seen before.
3. If independent state-dependent erase is required, expose a causal pre-write
   read/occupancy signal through a two-stage controller or separate occupancy
   memory and prove its temporal observability before training.
4. Strengthen and audit the local write-role parser by marker/key/value/filler
   strata; do not infer it from one aggregate F1.
5. Replace the one-hot learned-path oracle with a gauge-preserving intervention
   based on learned key prototypes, while retaining exact direct-memory oracle
   replay as the capacity gate.
6. Test Spin transport only on an explicit moving-frame task where the query
   frame is observed or coherently transported. Generic association cannot
   promote it.

G15C and the external-loss-only lane remain blocked until a prospectively
frozen repair passes fresh seeds. G15A-S remains valid, separate evidence for
supplied-coordinate composition and center-sensitive chart transfer; G15B does
not overwrite or weaken that result.

## Claim boundary

This is a completed, exact-SM75, three-seed commissioned-controller failure and
diagnosis. It establishes that the current token-local collision target and
generic token-wise transport strategy are wrong for the intended memory task.
It does not show that content-addressed fast weights, supplied-frame Spin
transport, GDN2/KDA-like correction, periodic attention, or a future
state-aware controller cannot work. It is not ordinary next-token, natural
text, autonomous discovery, full triality, scaling, or fused-efficiency
evidence.
