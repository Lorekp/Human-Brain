# Persistent session checkpoint QA

Both selected ordinary checks pass. The ten parent-only interruption cases pass
on the first launch at `cf26310d4ee96647db6eac4207f0f0a0161f66c5`. The actor
continuation case passes after a test assertion repair at
`8338af3938fb45c528b66529e1bb24d8b27cf72a`. The checkpoint implementation itself
is unchanged between launches. No V6 task, experimental checkpoint or old lesson
was replayed. This verifies persistence mechanics, not new task capability.

## Failure and focused repair

The first launch reports one pass and one failure in18.52pytest seconds,
19.824204outer seconds and1231.9609375MiB observed owned-process peak. The actor
case completes three checkpoint commits, two actual STOP children, both offered
corrections and strict restoration. It compares the online next trajectories,
then fails because the disabled actor's empty derived-cache tuple is the same
Python empty tuple as the restored actor's. No cached tensors exist in that tuple.

The repair checks independence of actual cached tensor storage, retaining exact
state and trajectory comparisons. Only the failed actor node is rerun, in a
fresh directory with the same180-second/2048-MiB watchdog. It passes in16.85pytest
seconds and18.0113583outer seconds at1232.08203125MiB observed peak. The passing
interruption node is not repeated. Combined outer time is37.8355623seconds;
the first failure and all its exposure remain recorded.

Both doctor receipts bind master and an empty tracked diff. Concurrent untracked
dendritic source work and the paused direct-request module are listed; none is
imported by these selected nodes. Ruff passes with the repository configuration.
Nonfatal pytest cache permission warnings do not affect the retained fixtures.

## What was verified

Ten parent-only fixtures exercise a clean boundary, an outstanding intent,
partial journal, complete attempt awaiting correction, uncommitted payload,
uncommitted manifest, published HEAD, source drift, payload corruption and an
unexplained empty execution directory. Their opaque payload bytes are never
tensor-decoded. Interrupted work rejects automatic resume and preserves the
last authenticated boundary; these constructed fixtures make no neuronal-state
validity or Windows power-loss claim.

The successful actor case constructs four actors: two original routes and two
fresh restore targets. Each original ingests `a`, then proposes and actually
executes STOP over `print(0)`. Both receive `1`/EOS with three action labels;
only online publishes learning. Three commits write three tensor payloads and
reuse three unchanged payloads. One restore reads two restricted local fixture
payloads, restores exact combined and reader counters and rebuilds independent
derived tensor storage. Both next reading trajectories equal their originals.

Exact numerical tensors, complete tape, actual file/feedback carry, simple
reference documents/demonstrations/offer counters and checkpoint evidence
frontiers are retained. The fixture reference history is explicitly fabricated
transport data; no extra reference prediction or teaching is credited.

| Activity | Failed launch | Repaired actor launch | Total |
|---|---:|---:|---:|
| Fresh combined actors |4|4|8|
| Tiny ingests / raw bytes / EOS |2 /2 /2|2 /2 /2|4 /4 /4|
| Free calls completed |4|6|10|
| Teacher calls / byte-or-EOS offers |2 /4|2 /4|4 /8|
| Action labels |6|6|12|
| Actual children |2|2|4|
| Checkpoint commits |3|3|6|
| Restore calls / fixture tensor loads |1 /2|1 /2|2 /4|
| Experimental checkpoint loads |0|0|0|

Checkpoint integration into a complete broader runner, clean evidence-segment
restart accounting and actual workload cost remain future work. Missing V6 state
remains missing. The [implementation contract](persistent-session-checkpoint-v1.md)
specifies interrupted-intent handling, source bindings and process-termination
limits; no silent retry or reconstructed learning is authorized.

## Evidence

| File | SHA256 |
|---|---|
| `.pytest-tmp/dev/doctor-67rblmvd/receipt.json` | `0da24916ad7c96f7aa3cb04a04e16a284f5b0afc64593b836f291012fd0c6eb7` |
| `.pytest-tmp/dev/doctor-rk_rx7c4/receipt.json` | `3d5622b9a10c3d109aee395b3ad37f065a5842b5d56a170ca64e8b4c38814831` |
| `.pytest-tmp/persistent-checkpoint-qa-v1.launch.json` | `ac2688852f455437709bc86d69c5af599689e3dc218d869742fea9741a63e86a` |
| `.pytest-tmp/persistent-checkpoint-qa-v1.junit.xml` | `8cd06da72578869fde124794b6e6d13ac2acdd490639fee440dd4830cfd65780` |
| `.pytest-tmp/persistent-checkpoint-qa-v1.stdout` | `6767140258fc3b6c3268ae004e9b1584658bf6ad0efd57b50e871f8dd82a5e66` |
| `.pytest-tmp/persistent-checkpoint-qa-v1-repaired.launch.json` | `9aba5d731000c02e1f4a1dc0ee4dd9fdf0cd22c43d7085a85f403b7eefc15b24` |
| `.pytest-tmp/persistent-checkpoint-qa-v1-repaired.junit.xml` | `7ba82163a8441edc4214e128109208a8bb2eb157ef480bc8b46fa209d65ba0ad` |
| `.pytest-tmp/persistent-checkpoint-qa-v1-repaired.stdout` | `3b2b9983b7403db05e276614fa933459e42557d3dea35ef02fc55ed52cda7cfb` |
| First fixture `test_persistent_checkpoint_int0/persistent-checkpoint-frontiers-passed.json` | `ca656fbdaa1bdc5cabd954a91392201a15176305d3edc4251598f83a8d3417da` |
| Repaired fixture `test_persistent_checkpoint_act0/persistent-checkpoint-continuation-passed.json` | `817784286bc30845b864e82ba55a7c50a9f42be86be16be7d93ce4c05b1f1f0c` |
| `experiments/r4_integrated/persistent_session_checkpoint.py` | `af68d18500e91d001f8efc8f8476fb053933e870c147504fb1626c94287272dd` |
| Repaired `tests/test_byte_sequence_decoder.py` | `2017fa6311cd22f05a34c6a4806b0315477f89195924d0ace52c938fd2a27cc9` |

Fixture paths are below the corresponding `.pytest-tmp/persistent-checkpoint-qa-v1-tmp`
and `.pytest-tmp/persistent-checkpoint-qa-v1-repaired-tmp` roots. Actual child files,
all payload generations and boundary/intent records remain beside the receipts.
