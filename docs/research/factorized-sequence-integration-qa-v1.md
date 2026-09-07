# Factorized routing integration QA

The selected [routing contract](factorized-sequence-routing-v1.md) is implemented
at `406c62c2074f4decfd5c0d21ac6ec22058e3415b`. Three reviewed ordinary test nodes
pass on their first invocation, 2026-09-07. This is numerical and lifecycle QA,
not acquisition, transfer, retention or recovery evidence from the broader stream.
No experimental checkpoint, task builder or benchmark model was loaded.

The reader retains its numeric priors, raw observation/action interface and
context assemblies. New shared float64 routing synapses receive direct local
outcome teaching from the selected owner's pre-clamp emitted byte. All owners
compete; independent exact-context preparation is used only for teacher-time
reuse/recruitment. Both staged systems and derived views validate before a joint
copy-only publication. The action wrapper uses the same reader and existing
execute-before-teach transaction. No backpropagation or reverse credit is added.

Candidate views are deterministic runtime caches of actual receptive patterns.
Checkpoints store their authoritative prototypes and router weights; strict
loading validates incoming state and constructs canonical views before copying.
Response custody binds values and borrowed buffer/tuple identities. No persistent
cache permits unchecked external mutations between responses.

## Selected checks and exposure

All nodes are in `tests/test_byte_sequence_decoder.py`:

- `test_factorized_router_local_outcome_contract`: one authored router, two
  candidate builds, one current build, two routing evaluations and three sparse
  activity evaluations. Three direct valid float64 competitions and two rejected
  competitions check near ties, lower-index ties, nonfinite values and subtraction
  overflow. Independent scalar score/update arithmetic checks the actual fixed
  populations, half-energy normalization and local potential clamp. Five outcome
  attempts include disabled, enabled, tampered, stale and empty-bank cases; three
  enter outcome teaching, one private update commits. One authored same-instance
  snapshot restoration, no reader or actual child.
- `test_factorized_reader_action_persistence_and_atomicity`: one combined actor
  and its one shared reader ingest 512 raw bytes plus EOS into 513 of 1024 slots.
  Two direct reader predictions and one action prediction produce five completed
  free events. One additional failed single-event response detects identical-value
  buffer replacement before pending publication. Four teacher calls offer eight
  total symbols including the failed final validation and explicit retry: two
  enabled successes, one disabled call, one rejected joint transaction. Held
  reader/action state and the tape remain unchanged where required. One strict
  same-instance restore rebuilds views; three invalid loads reject before any
  parameter publication. A real STOP action executes once, returns `0\n`, and
  precedes an authored SPLICE correction. The final view-validation rejection
  leaves both reader and action weights unchanged with their pending action
  intact. No task accuracy is required or claimed.
- `test_sequence_assembly_cells_local_staging_contract`: unchanged existing
  assembly fixture checks the new copy-only seam through its checked public
  publication path. It uses eight evaluations and seven entered/six completed
  assembly offers; no reader, checkpoint or child.

Pytest reports **3 passed in 10.71 seconds**. The outer monitored process takes
**11.84494 seconds**, with **635.27734375 MiB** maximum sampled owned-tree sum of
process peak working sets. One nonfatal pytest cache permission warning does
not affect the retained fixture outputs or JUnit. These timings describe the
authored checks, not a projection of the broader workload.

## Receipts

The doctor records the committed revision and an empty tracked diff. Untracked
v5 task source and the previously paused direct-request source are unexercised.

| Receipt | SHA-256 |
| --- | --- |
| `.pytest-tmp/dev/doctor-5evfag4o/receipt.json` | `4d6ea9e69d225bacafcc07c10f7c9221d8fb07265daaec5159eb6e39260fb291` |
| `.pytest-tmp/factorized-sequence-qa-v1.launch.json` | `3dc5ba3662ae05866ba575105bab302dcdbc555fbd368474fe9d16b9836435cb` |
| `.pytest-tmp/factorized-sequence-qa-v1.junit.xml` | `1f2f1e9f7660eca5d044444bd9fa951a0a85fc82fd9f3082f3bcaf42b251237f` |

The exact command is in the launch receipt. It uses this checkout's `.venv`,
the required CPU/thread environment and fresh fixture directory
`.pytest-tmp/factorized-sequence-qa-v1-tmp`, with a 60-second/2048-MiB watchdog.
Raw and passed fixture receipts retain actual local deltas and work counters.
Ruff passes for the five affected implementation files and the test file.

The [source/resource review](../../.pytest-tmp/factorized-sequence-source-resource-review-v1.md)
counts sparse routing, all-owner and all-fact competition, teacher-only Hamming,
staging, view construction and custody work. It found the borrowed-reference gap
fixed before this QA. Its cumulative allocation counts are not RSS. The complete
broader continuation, initializer, reference restoration and saved audit still
require their own reviewed release and actual measurements. No automatic
micro-gate or perfect-toy prerequisite follows this integration check.
