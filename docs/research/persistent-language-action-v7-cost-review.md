# V7 partial cost review

V7 stopped at its clean reserve boundary after **185 of 192 attempts**, at **1771.0604454 seconds** and **818.6484375 MiB** owned-process peak. Checkpoint create/begin/capture/commit intervals total **917.9653919 seconds**, or **51.8314%** of the run. Persistence is therefore the first measured performance priority. These intervals include integrity work and serialization; they are not an isolated disk-write benchmark.

The saved-prefix audit passes after one auditor repair. The experiment remains incomplete: a passing custody review does not complete the seven missing attempts or establish learning success. The released source revision is `a1d900a688d353fcc00e43b86ccb14b1547f3065`. The selected ceiling was 1800 runner seconds plus 120 saved-review seconds, with 2048 MiB peak. No learner rerun or checkpoint decoding was performed for this review.

## Durable work and retained state

| Recorded exposure | Online | Learning-disabled | Simple |
|---|---:|---:|---:|
| Attempts / planned 64 | 62 | 62 | 61 |
| Execution receipts / planned 44 | 43 | 43 | 42 |
| Whole documents appended | 12 | 12 | 12 |
| Source raw bytes + EOS | 738 + 12 | 738 + 12 | 738 + 12 |
| Neural source offers | 750 | 750 | 0 |
| Correction occasions | 4 | 4 | 4 |
| Neural correction symbols / action labels | 71 / 6 | 71 / 6 | 0 / 0 |
| Actual free neuronal decisions | 2378 | 533 | — |

All fresh-transfer, retention and fixed recovery attempts are recorded. The remaining seven are acquisition encounters at the end of the interleaved schedule; the next cursor is episode index 45, simple route, attempt 1. Missing attempts are not silently recorded as observed failures. The 128 execution receipts belong to the original run; the audit's `actual_executions: 0` describes the audit itself, not the experiment.

All routes reach 1614 factual positions from 864. The online source counters report 580 recruited branches and 27 exhausted offers; corrections report another 66 recruitments. Disabled recruitment counters describe computed proposals against its unchanged bank, not published growth. There is no evidence that global capacity alone guarantees room in each output column.

The saved HEAD is sequence 221: **222 boundaries**, comprising initialization, 36 appends and 185 resolved attempts. The audit finds no outstanding intent, journal tail, execution file or execution directory beyond HEAD. Its `pending_transaction` identifies the next schedule position, not an open dirty intent. Both latest route payloads are authenticated and each is 91,080,533 file bytes. The audit decodes none. Unlike V6's lost live continuation, V7 retains a clean numerical-state boundary; actual restoration remains a separate operation.

The source-declared two actors/tapes occupy 182,123,888 tensor bytes; initial derived views/self-wiring bring named retained storage to 190,922,008 bytes. These quantities exclude Python objects, scratch and allocator overhead. The bank binds 30 route-payload paths; the current review does not infer every historical file's exact size from the latest pair. The attempt, source and teacher journals occupy 186,086,649 bytes, plus 1,614,587 bytes of checkpoint telemetry: **187,701,236 bytes** across those four journals. No final compatibility `final-state.pt` or completed-run audit is claimed.

## Time without double counting

| Saved interval, seconds | Online | Learning-disabled | Simple |
|---|---:|---:|---:|
| Attempt elapsed, including begin | 532.7235595 | 257.9346343 | 66.8277448 |
| Included checkpoint begin | 65.6889389 | 65.8472595 | 65.0110861 |
| Attempt elapsed less included begin | **467.0346206** | **192.0873748** | **1.8166587** |
| Whole-source ingestion/learning | 63.4198857 | 62.1850497 | 0.0139726 |
| Corrective teaching | 24.7827465 | 24.6183521 | 0.0028137 |

Subtracting the exact saved overlap removes **196.5472845 seconds** from the 857.4859386 inclusive attempt seconds, leaving **660.9386541 seconds**. This still includes actual actions, receipt conversion, hashes and acknowledgement; it is not pure forward time. Source intervals total **125.6189080 seconds**, and correction intervals **49.4039123 seconds**. Disabled source/teacher proposals are real computation despite no publication.

| Checkpoint phase | Calls | Seconds |
|---|---:|---:|
| Create | 1 | 0.3131199 |
| Begin | 221 | 221.4462138 |
| Capture frontiers | 222 | 69.8838896 |
| Commit | 222 | 626.3221686 |

The disjoint intervals above sum to **1753.9268663 seconds**. The **17.1335791-second** remainder includes initialization, prior/reference restoration, source capture, journal append overhead not included elsewhere, shutdown and other unallocated work. The two allowed aggregate artifacts do not isolate those components further, so no precise split is invented. Checkpoint journal metrics and enclosing phase intervals must likewise not be added twice.

The first saved audit failed after **12.9411646 seconds**; the repaired audit passed in **42.3644378 seconds**, peaking at 85.79296875 MiB. Both invocations are counted. The [final outer receipt](../../benchmarks/persistent_language_action_v7.measurement.json) binds the banker's **2.7871129 seconds**, yielding **1829.1531607 seconds** across the run, both audits and banking. The unchanged bank records 1828.7336663 seconds through banker preparation and marks its final outer measurement pending; the companion closes that subtotal. This accounting is not a completed workload result.

## Work interpretation and one optimization

Saved free/teacher counters report **4,621,658 factual tournament comparisons** and **241,196,159 event-router pairs**. All 2911 actual free decisions and 142 correction-prefix decisions retain the old prior circuit; source observation adds 1500 branch evaluations without that prior. The event and factual circuits have different relay/gate costs. Their counter ratio is not a measured latency ratio, and changed generation lengths prevent an isolated V6–V7 speed claim. The corrected online route generates substantially more decisions than disabled.

Recommend one concrete persistence change: **authenticate each immutable route-payload version once during publication and hold a read handle that denies writes and deletion for its owned session lifetime**. Later safe boundaries can reuse the sealed identity/hash/length through that same handle instead of repeatedly streaming unchanged historical payload bytes. This targets repeated prior-payload integrity reads inside begin/commit, while keeping every safe boundary and every live-model validation.

The implementation must fail closed if exclusive immutability cannot be established; a path, timestamp or tensor version alone is insufficient. Seal the exact bytes before publication, retain the handle and file identity, reject replacement/rebinding, and release only after the session ends. Keep mutable actor/tape checks, source bindings, journal-prefix checks, intent ordering and atomic HEAD publication unchanged. Independent reopen/restore must still authenticate bytes afresh. This is an owned immutable-file optimization, not a cross-session unchecked hash cache. Ordinary mutation/restore checks belong to the eventual selected change; no new capability or speed gate is proposed here. The 918-second checkpoint total is an upper context for opportunity, not the predicted saving from this narrower change.

The verified host has a Ryzen 5 7600X (6 cores/12 threads), 33,378,181,120 bytes RAM and an RTX 4070 SUPER with 12,282 MiB VRAM. The measured run remained CPU-only, one Torch compute thread, under its 2048-MiB limit. A GPU port does not directly remove payload hashing, file-system durability or host custody traffic. More RAM does not by itself solve repeated full-state reads. The 100-billion-neuron aspiration remains separate: one byte per cell alone is 100 GB, before connectivity or learning state; current tensor and repeated-work counts are not evidence of that neuronal capacity.

## Evidence and scope

Only the existing bank and repaired audit JSON were read for this saved cost review. No raw journal replay, new measured extractor, model import, checkpoint decoder, feature call, task execution or source modification occurred. Positive contact arithmetic and saved tuple arbitration are checked by the audit; omitted contact activations, negative vectors and inherited neural forwards are not reconstructed.

| Input | SHA256 |
|---|---|
| [Result bank](../../benchmarks/persistent_language_action_v7.result.json) | `b11b0e0517ef7939cf624e68dfdce05cdd75bc75aa8a81e400ddfd7974fcc67e` |
| Repaired audit `.pytest-tmp/persistent-language-action-v7-review-v1-repaired.json` | `457b15182b19b6e60b2b9cf1051be58b1b4c4e76128bab696c0e89a6e92c1423` |
| Frozen manifest, bound by both | `c2e1cb6237fdff468c6d413ce62c9dfbf514a607fb1a8af45bdde285c3f4b898` |
| Saved HEAD, bound by audit | `3bffe29937c840121fdd0e920bbfc3e54067389e0649d43530ce3df392d94656` |
