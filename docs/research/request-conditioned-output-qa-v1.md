# Request-conditioned output ordinary QA

Two ordinary mechanics tests passed on their first launch. This verifies fixed
cell arithmetic, direct local update arithmetic and action/state ownership. It
is not an acquired-skill threshold, scientific micro-gate or broader-session
result. No additional model run was performed by the independent reviewer.

## Execution and source scope

The checkout doctor passed at revision
`f672e49fdc77d97c67eb97e955a666d3db373302`. The appended test changes and new
implementation files were **uncommitted** at this ordinary QA invocation.
Doctor records tracked diff SHA
`3cd86181586aac49162b0ec03b621d2b723617613c557eb1e4ca31ce94d2df5c`
and the untracked file list; that diff does not authenticate untracked contents.
The tested code therefore must not be described as a clean committed snapshot
of that revision. Root will separately freeze the complete source before the
broader workload.

The selected nodes in `tests/test_byte_sequence_decoder.py` were:

- `test_request_conditioned_cells_and_weighted_local_update`: independent signed
  threshold currents on four authored request/prefix fixtures; actual two-address
  teacher currents captured without an extra forward; scalar responsibility and
  U/B/C update oracle; isolated-clamp algebra; fixed-wiring tamper rejected before
  publication.
- `test_request_conditioned_prefix_atomicity_and_actual_action`: authored
  nonadjacent address switch 0→2 with generated `a` then EOS; zero-byte limit;
  teacher prefix `b` distinct from the generated prefix; disabled correction;
  rejection on the second offered output proposal with no live publication;
  combined fixed-wiring rejection; one actual STOP execution before correction
  of that same pending action.

Pytest reports **2 passed in 6.53 seconds**; JUnit suite time is 6.529 seconds.
The monitored process took **8.0727801 seconds**, with an observed process-tree
peak of **317.44140625 MiB**. The 60-second ordinary QA watchdog passed. These
small authored tests do not measure full-memory teacher throughput.

The combined wrapper's all-new-parameter `_check` was present for this run.
Subsequent explicit direct-reader `ingest` and `snapshot` validation wrappers
were added and source-reviewed after it; this result does not claim those later
additions were independently executed by these tests. No extra test launch was
required or performed for that source-only addition.

## Actual exposure ledger

| Exposure | Count |
| --- | ---: |
| New readers / fixed-cell modules | 2 / 2 |
| Combined models / action circuits | 1 / 1 |
| Ingest calls / raw bytes / EOS | 2 / 3 / 2 |
| Free calls, including action payload / decisions | 3 / 5 |
| Teacher invocations / target presentations | 4 / 7 |
| Successful teacher calls / their offers | 3 / 5 |
| Presentations inside deliberately failed teacher | 2 |
| Extra standalone fixed-helper calls / candidates | 4 / 8 |
| Total helper calls / candidate populations | 16 / 33 |
| Actual execution children | 1 |
| Prior or file-checkpoint loads / public-stream runs | 0 / 0 |

The failed teacher first completed a private staged offer, then reached and
rejected the second offer's finite-validation hook. Neither offer published to
the live model. This is an authored fault-injection case inside a passing test,
not a failed QA launch or a retried learning experiment.

The real child PID39076 returned zero with stdout `610a` (`a\n`) after the
actual STOP proposal. Its child interval was 0.04005779999715742 seconds. The
fixture's subsequent SPLICE teacher was supplied only after its execution
receipt had been written. No correct-action selection or task fit is implied.

## Prospective broader resource accounting

These are source-derived counts for the near-final v2 runner, not measured
outcomes. One authenticated continuation load contains two previous model/tape
containers and 2,919,568 tensor bytes. Both new routes preserve the same previous
online model, experience46 and all453 occupied slots; neither receives a
task-selected memory. Each route appends two notes containing50 raw bytes and
two EOS symbols, reaching480 and then505 occupied slots. Allocated tape size
remains825352bytes regardless of occupancy.

The selected stream retains216 attempt slots and144 actual edit-verification
slots across three routes. Each neuronal route receives24 corrections with98
reader byte/EOS offers and36 action-head labels. Teachers evaluate all current
addresses even when learning is disabled. Summing the actual scheduled memory
sizes gives96,280 teacher candidate/position pairs across both routes,
7,225,236,320 residual forward synapse products, the same number of local output
products,788,725,760 fixed afferent applications and24,647,680 threshold cells.
This uses the concrete98-offer schedule, not24 maximal17-offer teachers.

Each new combined model contains634432old tensor bytes,604448new plastic tensor
bytes and73728fixed wiring bytes. Two models and two allocated tapes total
**4,275,920 tensor bytes** for the final checkpoint payload; serialized file
overhead must be measured afterward. The prior-load storage and final tensor
payload are not peak RSS estimates.

The runner retains all216 JSON attempt summaries for final metrics. It is not
a constant-row streaming implementation. A worst-case neural free summary has
17 sets of505 scores,505 prior values,257 logits and257 output probabilities;
up to136 supported neural calls can therefore retain about3.52million such
floating scalars, before dictionaries, list headers and other receipts. A
single maximal free neural trace additionally retains about19.6million bytes
of competition tensors; a generic17-offer all-address teacher retains about
17.7million bytes of logits at505slots. Previous feedback may overlap the next
teacher's current allocations. These bounds are material storage components,
not a complete allocator or RSS proof.

No concrete count or capacity blocker was found. The retained rows, repeated
prefix features, all-address output arithmetic, fixed wiring hashes, staging,
serialization, initialization and real children must all remain inside the
complete **840-second run plus60-second review /2048-MiB** envelope. The earlier
19.662127-second session and these ordinary tests cannot project the new full
runtime. Report the next measured stage times and memory after completion;
do not infer a throughput improvement or reject all-address teaching from
operation counts alone.

## Saved evidence

The reviewer read only saved JSON/XML and source text. Direct sandbox access
to the fixture directory was denied; one root-authorized elevated read of the
three exact receipts succeeded. No files were copied or regenerated.

| Artifact | SHA-256 |
| --- | --- |
| `.pytest-tmp/dev/doctor-g1jd4ytu/receipt.json` | `b37e88c4e54fb2322d106c581fbe9ffefd1deca21e6bace83fa86c46b444a044` |
| `.pytest-tmp/request-conditioned-output-qa-v1.launch.json` | `efabdb93bca0897132822b53b36307a042bc87db632c306bba9c823bd4efbe24` |
| `.pytest-tmp/request-conditioned-output-qa-v1.xml` | `153184a0ce07dcb7d0e11c841481b3a9a31cb3809d3dbba5712eaf1c9ec3d027` |
| `.pytest-tmp/request-conditioned-output-qa-v1.stdout` | `7213474da949354b7c8b8de7a84ab5b4cae01b1ce322b65dd07c70f4d9f69607` |
| `test_request_conditioned_cells0/conditioned-cells-local.json` | `8e2fffe1be073bfa925c176a16c75712b32fa377b5079afd3cbe306197a449a2` |
| `test_request_conditioned_prefi0/conditioned-prefix-action.json` | `219af4b0def61c1bc783c765e3755be60b6f53c02def7ce555c2224cfcc330e0` |
| `test_request_conditioned_prefi0/actual-action.json` | `411b23f913843dba296999b86e8e5dc11411843de27ad150dad2b8ec4e6947b1` |

The final three paths are beneath
`.pytest-tmp/request-conditioned-output-qa-v1-temp/`.
