# Exact factual competition: integration QA

The two selected ordinary checks pass on their first launch at source
`af975cd9c3ed6fda8a11b8c47c10df4ad9090625`. Pytest reports12.75 seconds; the
180-second/2048-MiB watchdog reports **13.9122443 seconds** and
**688.4765625 MiB** observed owned-process peak. This verifies mechanics and
integration, not a new skill result or measured full-workload speedup.

Doctor records master, an empty tracked diff and only the preexisting paused
direct-request source as untracked. Ruff passes with `tools/ruff.toml`. A
whole-file formatter check would reformat existing nearby tests; no unrelated
formatting was applied. Source review fixed actual feedback-field transport and
comparison of tensor-bearing teacher-step dictionaries before the first test
launch. There is no failed QA invocation. Pytest's nonfatal cache-directory
permission warning does not affect saved JUnit or fixture receipts.

## Circuit and learning behavior

`test_factual_tournament_exact_circuit` in the existing
`tests/test_byte_sequence_decoder.py` compares18 valid inputs with the old
all-pairs network and rejects eight invalid inputs. Widths include1,2,3,127,128,
129,511,512,513,1760,2017,2047 and2048, with extra equal-score ties, odd carries,
signed zeros and opposite finite float32 extremes. Every saved level is checked
independently from original scores/identities: comparator currents, threshold
activities, copied potentials and identities. Owned original-score storage,
signed-zero copying and full/odd-width allocation bounds pass.

There are18 old-network calls and26 new calls including eight rejected inputs,
with no learner, teacher, checkpoint or child execution. The new network performs
N-1 comparisons with explicit gated potential/identity relays; it does not claim
identical all-pairs intermediate currents or the predecessor's two-layer theorem.
The [source contract](factual-address-tournament-v1.md) defines actual and logical
work counters.

`test_tournament_recognition_executed_learning_parity` constructs two fresh
combined actors, preceding recognition and its tournament successor. Each ingests
`ab`, proposes an actual STOP, executes a real child over `print(0)`, and receives
one `1`/EOS correction with three action labels. Each then predicts on its actual
feedback and receives a disabled `2`/EOS offer. Each actor's same reader learns
and acts throughout.

Corresponding numerical model/tape tensors, proposals, complete generated steps,
address scores and enabled/disabled teacher-step algebra are exactly equal.
Disabled correction preserves state. Explicit metadata and factual circuit work
differ, as intended. One same-instance strict in-memory state load preserves the
new schema and exact tensors while rebuilding independent derived wiring. No
new checkpoint tensor is introduced. Old frozen source classes and float64 event
competition remain unchanged.

| Integration exposure | Count |
|---|---:|
| Fresh combined actors | 2 |
| Ingestions / raw bytes / EOS | 2 / 4 / 2 |
| Free calls / byte-or-EOS decisions | 4 / 6 |
| Teacher calls / symbols | 4 / 8 |
| Enabled / disabled symbols | 4 / 4 |
| Action-head labels | 6 |
| Actual children | 2 |
| Same-instance in-memory state loads | 1 |
| Experimental or file-checkpoint loads | 0 |

No V6 task, prior checkpoint, corpus or lesson is replayed. These authored
mechanics cases do not assert learned task accuracy and are not a micro-gate.

## Cost interpretation

The [partial V6 review](persistent-language-action-v6-cost-review.md) records
3,357,755,083 factual pairs over2642 durable decisions. Those same populations
would require4,151,630 tournament comparisons plus explicitly counted relays.
This is source-operation arithmetic, not an808-fold latency claim. Response
lengths, validation, hashes, actions and checkpoint work still matter.

A2048-address call retains59,367 circuit tensor bytes. Across every allowed
occupancy, cumulative named allocations are bounded by124,086 bytes (at2017)
and peak named live buffers by82,899 bytes (at2047). These exclude Python objects
and framework internals and are not RSS estimates. The next broader measurement
must include this integral code and persistence overhead; no isolated speed run
is selected.

## Evidence

| File | SHA256 |
|---|---|
| `.pytest-tmp/dev/doctor-0rmsth9x/receipt.json` | `c7b0c9a083ec9a0579e9c194dac53862b122800c4adee52de92dbe2521180dc4` |
| `.pytest-tmp/factual-tournament-qa-v1.launch.json` | `d623971bce03965a7d5548e49400c1fb5f5708b2ccfd353a90cca1e6fe22214d` |
| `.pytest-tmp/factual-tournament-qa-v1.junit.xml` | `d9a82a9db8f429bb689703e3cb0b59eb98e5759f799c399581c25f7360d17ab7` |
| `.pytest-tmp/factual-tournament-qa-v1.stdout` | `4e0c1b38b6e10c1d0cf9435a2171e9b8352315cd46b7981bff6e05e00e0878d5` |
| `test_factual_tournament_exact_0/factual-tournament-passed.json` | `88e5e9dc232d64b78b1c660663eab1890d5ae72501d75a716f8ab72fd3d0d44f` |
| `test_tournament_recognition_ex0/tournament-recognition-parity-passed.json` | `89fe4789ff1d0cd3c4ce737ba503154c5b2ff329dadf82d4012cc834a47af245` |
| `src/humanbrain/factual_address_tournament.py` | `946e5fa7b0c6b1fe51e1c3e5086de3a8621c63af72a8972303de3ef51d3f90cb` |
| `src/humanbrain/tournament_recognition_learner.py` | `3de505ff9c8912ddd78bee9856f61aa9a30c54c9aeb7aa1d9d7d397e7556b6b9` |
| `tests/test_byte_sequence_decoder.py` | `ff634673fe0c0dccbcd6ef69e7878b620a68e8ddd737d7a91cf540085c11a4e4` |

Fixture paths are relative to `.pytest-tmp/factual-tournament-qa-v1-tmp/`;
actual child workspaces/receipts remain beside them. Watchdog stderr is empty.
