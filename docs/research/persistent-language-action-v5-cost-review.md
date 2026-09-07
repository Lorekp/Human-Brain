# Persistent language/action v5: complete cost review

The first full run and independent saved audit passed. Their complete elapsed
time was **566.9125368 seconds**, with maximum sequential owned-tree peak
**723.078125 MiB**, within the selected 840 + 60 seconds / 2048 MiB envelope.
This is resource compliance for the actual broader session, not an isolated
routing speed measurement or a claim of useful capability.

Released source: `c2b3f92e8a702e3908deb6332c1a9c27bcc14e6f`.
Manifest: `de182acf644b18fed25948c5c7886006253bee8618a4e6ce647508f624b86fca`.
The complete counter ledger is retained in
`.pytest-tmp/persistent-language-action-v5-resource-review.json`, SHA256
`a1ab4791748022cf99aa61ed143848d9ba8f480d275f14ac6faf2625b580b7c4`.
The saved-only extractor passed once: 1.602529 seconds internal and 1.7552412
seconds process elapsed. It authenticated 316 Python source files and all
completion-bound artifacts, streamed journals and hashed checkpoint bytes;
it loaded no tensors and called no model, action, builder or auditor.

## Complete and recorded stage times

| Interval | Seconds | Owned-tree peak MiB |
|---|---:|---:|
| Runner watchdog | 558.5371355 | 723.078125 |
| Saved-audit watchdog | 8.3754013 | 102.6640625 |
| Complete sum / sequential maximum | 566.9125368 | 723.078125 |

Time headroom was 333.0874632 seconds; peak headroom was 1324.921875 MiB.
Sequential peaks are compared, not added. The watchdog samples owned-process
working-set peaks; short-lived children may fall between samples. This is a
monitored desktop run, not an isolated cache benchmark. Root made a read-only
progress count during the run; that count was not used as a throughput forecast.

| Route | Attempt intervals | Teacher intervals | Append intervals |
|---|---:|---:|---:|
| Online | 145.7014518 s | 66.5981129 s | 0.6865599 s |
| Learning disabled | 269.5055270 s | 57.9483150 s | 0.6622685 s |
| Simple reference | 2.2702880 s | 0.0054741 s | 0.0041936 s |
| Total | 417.4772668 s | 124.5519020 s | 1.3530220 s |

Attempt intervals account for 74.92% of the report timer and teacher intervals
22.35%. These are composite timers: attempts include state hashes, prediction,
receipt conversion, actual edit execution and untaught acknowledgement. Teacher
intervals include pending checks, sequential proposals/publication and state
hashes. The largest individual teacher interval was online `A00-E`, 14.3887377
seconds; the largest attempt was disabled `R15-E`, 11.9832162 seconds. There is
no separately measured validation, routing or hash duration within these rows.

Other explicitly timed intervals were initialization 1.8127423 seconds,
reference restoration 3.2458192, reading prior outcomes 2.2496853, source capture
0.1380892 and checkpoint snapshot/serialization 0.2073673. Subtracting these
and all row timers from the 557.2529606-second report timer leaves 6.2170665
seconds unallocated, including journal append/fsync and other host work.
Completion follows the report by 0.4893242 seconds; the external run interval
adds 0.7948507 seconds. These residuals are not invented per-stage measurements.

All 156 actual children ran, 52 per route. Their complete execution-envelope
intervals sum to 5.8349732 seconds, with 5.3732621 seconds inside child intervals;
both are already nested in attempt time. Exit codes were 38 zero and 118 one.
No child timed out. Capture limits were reached six times in the disabled route
and seven in the reference; online had none. One disabled proposal was rejected
for lacking actual payload EOS and still received unchanged-workspace verification.
The three-second allowance is a limit, not an observed duration.

The two earlier author-only 36-program verification invocations are separate
development exposure: the first retained sibling-import failures, followed by
corrected verification of the same manifest. They are not among the 156 actual
session children and added no neuronal or corrective-label exposure.

## Actual state, learning and simulation work

There were 240 completed attempts and zero unsupported inputs. Both neuronal
routes made 80 free calls and 52 action proposals, with 72 untaught acknowledgements
each. There were 24 correction records, eight per route. Each neuronal route
received 155 raw teacher bytes plus eight EOS events and twelve action labels.
The two tapes grew only through six public documents per route: 353 raw bytes
plus six EOS, from 505 to 864 occupied slots. Factual association writes remained
enabled in the disabled control; source-next-byte teacher offers were zero.

| Population / publication | Online | Learning disabled |
|---|---:|---:|
| Free output events | 593 | 2377 |
| EOS / LIMIT responses | 80 / 0 | 79 / 1 |
| Free owner candidates, summed | 224726 | 558595 |
| Teacher owner candidates, summed | 51508 | 38305 |
| Teacher output offers | 163 | 163 |
| Proposed novel contexts | 163 | 163 |
| Staged output updates / new owners | 163 / 163 | 0 / 0 |
| Staged direct routing outcomes | 163 | 0 |
| Final occupied owners | 398 | 235 |

No teacher context was exact or exhausted, and no routing outcome was skipped.
Both controls started from the same retained 235 owners. The disabled route
computed the same kind of local proposals but never advanced them. Its slow
numeric state remained unchanged; both routes retained their original B/C/U,
codec and fixed-reader tensors. The reference restored 18 documents and 42 edit
demonstrations, then ended with 24 documents and 46 edits; reading corrections
were not stored as answers.

Across free and teacher routes, counters report 3296 current-context builds,
873134 owner scores, 876430 sparse activity constructions including selected-row
reconstruction, 478553206 fixed coincidence products and 489111753 learned
routing products. Actual f64 routing competition evaluated 121055943 pairs.
Free Hamming recruitment work was zero; teacher-only Hamming remained separate.
The incremental prefix helper evaluated 229439488 new lag pairs. Action heads
evaluated 29680970496 fixed-feature products, with their bounded blocks retained.

The full JSON records every primary work counter by free/teacher/ack/append/
initialization bucket. Selected aggregate custody values, including initialization,
are:

| Declared counter | Values or bytes |
|---|---:|
| Assembly validation values | 38288507280 |
| Factorized validation values | 8075463072 |
| Router validation values | 5333915248 |
| Factorized channel norm products | 6811447680 |
| Assembly custody bytes hashed | 46660540604 |
| Other counted tensor bytes hashed | 62087439616 |
| Router custody bytes hashed | 36852880338 |
| Raw factorized-context bytes hashed | 565227336 |
| Canonical owner-view rebuilds | 4460 |
| Router + assembly stage-copy bytes | 690539648 |
| Router + assembly publication-copy bytes | 345269824 |

Counts mix different units and scopes; validation values are not bytes or time.
Hash counters do not include every source/file/host hash. Feature named-tensor
allocations total 159390406568 bytes cumulatively and router named allocations
11773852660 bytes; neither is simultaneous live storage. Aggregate allocation
totals must not add component-detail counters to their already inclusive totals.

## Stored tensors and evidence

Initialization decoded the one complete v4 checkpoint: 38 tensors / 82646896
bytes, including both stored routes. It constructed two independent successors
and derived 235 owner views for each. Each new model owns 48764528 tensor bytes
and each capacity-1024 tape 1650696 bytes. Final checkpoint tensor payload is
100830448 bytes; the actual serialized file is 100852331 bytes.

Initial views contain 128641 sparse entries and 1031948 tensor bytes per route.
Final online views contain 216371 entries / 1735744 bytes; disabled retains
128641 / 1031948. These exact final entry counts are derived from existing
teacher work: all sparse pooling minus current dense writes equals proposed
new-view entries plus final canonical-view entries. Enabled proposals are all
installed, so `final_entries = (that sum + previous_entries)/2`; disabled views
do not advance. Build/rebuild counts, integrality, owner counts and bounds all
passed. No prototype or checkpoint tensor was decoded to obtain these figures.

Final model/tape/view tensor storage totals 103598140 bytes. Views are derived
runtime tensors, not duplicate checkpoint arrays. The measured 723.08 MiB peak
also includes transient arithmetic, snapshots, Python receipts/history and
allocator retention. The per-call teacher stage is 43158728 bytes; the much
larger cumulative copy total is repeated traffic across sixteen calls.

| Journal | Rows | Bytes | Largest line |
|---|---:|---:|---:|
| Attempts | 240 | 74199764 | 3065766 |
| Teachers | 24 | 9510646 | 1168596 |
| Appends | 18 | 28847 | 1949 |

All lines fit the saved audit's 8 MiB guard. The independent saved audit checks
recorded outcomes, custody, bounded work, direct routing scalars and local
outgoing-current algebra. It does not reconstruct every omitted lexical gate,
theta delta or tensor value. The source and ordinary mechanics QA own those
implementation checks.

## One priority: checked ownership for the free routing response

Root selected the narrow owned-free routing cleanup after these measurements.
The current public context builder validates its full population, and public
router evaluation immediately repeats that validation. Each free byte also
rechecks and hashes unchanged theta. A private path for one synchronous owned
free response can reuse its validated context/parameter scope and router-domain
theta digest while keeping the public checked API, entry/end byte binding,
captured object identities and all pending/teacher checks.

On the same actual 2970-event / 160-call trajectory, removing one duplicate
context validation per event removes 3068912880 declared validation values and
49101940800 cumulative named temporary bytes. Computing theta's digest once per
call instead of per event removes 23228673920 counted hash bytes. These are
source-operation differences, **not saved seconds or peak-memory predictions**.
Keep every factual and owner candidate, all f64 score reductions/thresholds,
selected activity and local update unchanged. Do not combine this change with
ragged batching or different reduction order. Ordinary checked-path parity and
mutation/rebinding/pending cases provide the appropriate verification; no new
skill or isolated speed gate is selected.

V4 complete time was 560.1913689 seconds; v5 is 1.20% higher descriptively. The
workload, memory, initial owners, routing and response lengths changed, so that
ratio cannot isolate the earlier optimization. Likewise, the online/disabled
timing gap is confounded by 593 versus 2377 free events and different owner
occupancies. Cost and behavioral outcomes must remain separately interpreted.

## Evidence identities

- Run report: `05307278505cbb3337c44e6bbc8d5168f898446c9f5f69c1486d233dfca8ad7c`
- Completion: `6c6972bda723a82b5afe1265b03485562fd446961893026e9b1ffa7381f6a64f`
- Run monitor: `44dc5ea3e89c03669f7b94a6ae4007e0f5f0860d19ae87956bb9d3680a1d73ab`
- Saved audit: `2d9a3d34da9fb46b61856b2ea2c51b73771c2f226d0dc5f3a35b7f57b9219898`
- Audit monitor: `948d7ebc99e9c38e5f7110f4ffbcab7f0d89062848b22daeb42869a85c2076bb`
- Final checkpoint: `a141230e836d6da83e9914ae52f665d39d0d3e59089a60203b7a963ecd59ea51`
- Extraction source: `05da99491c9419e1797047d802dfb485d7cb31671617e63579db87df263e9a13`
