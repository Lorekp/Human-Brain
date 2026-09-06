# Byte-event preprocessing qualification

Status: passed, 2026-09-06. Frozen source
`cf5c7a1d7e94550ace99dae75b5a8fe183d34735`.
The [fixed preflight](local_byte_event_preflight_v1.md) and independent audit
finish in **1.0885742 seconds**. Exact byte preservation and fewer projected
writer events earn integration of the fixed neuronal encoder and its QA.
No neural fit is admitted and no learning or simulation speedup is demonstrated.

## Preservation and counts

The utility groups generic ASCII word/whitespace runs, capped at 16 bytes, and
preserves every other byte as a singleton. It performs no normalization, role
parsing or vocabulary lookup. All 3,072 TRAIN document/query strings and 274
authored fixtures preserve exact payloads and contiguous offsets across the
declared chunkings. Pending buffers retire at finish; peak raw buffering is
16 bytes. Unicode and invalid UTF-8 remain exact bytes, with less compression
for non-ASCII text.

| One pass of TRAIN strings | Strings | Raw bytes | Raw events | Byte + EOS baseline | Event + EOS projection |
| --- | ---: | ---: | ---: | ---: | ---: |
| Documents | 2,048 | 49,152 | 20,480 | 51,200 | 22,528 |
| Questions | 1,024 | 25,600 | 12,288 | 26,624 | 13,312 |

This is a 56% reduction in projected document-controller events and a 50%
reduction for question controllers. The future fixed encoder still processes
every byte and EOS. Its additional work must enter fresh complete-workload cost;
event reduction alone says nothing about total runtime or useful binding.

All repeated host checks total **7,558 segmentation invocations**, 160,352 offered
bytes, 68,131 raw events, 80,680 feed calls and 7,558 finish calls. The 7,558
conceptual end events are accounting projections: no EOS neuron was executed.
The independent regex audit segments 3,346 inputs containing 75,601 bytes once,
and scans the same bytes once for reference buffering. It verifies all saved
chunking receipts without re-executing the production utility. These additional
host passes are separate from the runner counts.

Actual neuronal encoder/controller steps, teacher offers, output decisions and
actions are all **zero**. The TRAIN loader supplies only document/query strings
from the authenticated schedule; labels and final endpoint checks are not passed
to segmentation. The prior roster is hash-bound, with no prior model loaded.

## Verification and resources

29 distinct affected tests pass in the existing tokenizer test file. The utility
lane records 28 cases in `.pytest-tmp/byte-events-unit-v1.xml`; the runner's tiny
counter fixture adds one in `.pytest-tmp/byte-event-preflight-unit-v1.xml`.
Their disposable byte exposure and doctor receipts remain recorded separately.
Independent authored regex, count/schema and three-process ancestry checks also
pass before launch. No extra fixture or delay was added to lengthen the run.

| Process | Outer seconds | Observed tree peak MiB |
| --- | ---: | ---: |
| Preflight | 0.6120250 | 35.55078125 |
| Independent audit | 0.4765492 | 35.53125 |
| Combined computation | **1.0885742** | Separate peaks above |

Both actual Python processes were observed in their creation-keyed owned trees,
including the intervening virtual-environment redirectors. The audit's reported
PID 33940 and injected timing context match its watchdog. Its 59-second ceiling
is within the remaining original aggregate 60-second budget. Sampled process
peaks can miss brief children; no sequential peak sum or exact allocator claim
is made. Source hashes, prior-input hashes and all receipts pass their checks.

The [decision record](../../benchmarks/local_byte_event_preflight_v1.decision.json)
binds admission, sources, saved records, QA and watchdogs. Independent audit SHA256:
`286b0bffad78fdf7938418aa5f7d1fa08378450e34ff0e4c8556d32e3db76669`.

Proceed to the [fixed encoder interface](../../docs/research/local-token-event-interface.md),
with shared document/query encoding, explicit state retirement, exact byte-model
reference parity and fresh cost admission. Lossless host spans do not certify
lossless neuronal codes, unseen-byte composition, learned writing or coding.
