# Contextual evidence selection in the executed Python loop

Public repeatable development, declared before inspecting candidate outputs.
The adjacent `contextual_evidence_cases.json` fixes eight teaching examples and
six transfer cases. They test coding requests backed by longer prose; they are
self-authored development data, not substitutes for authentic exam questions.
Static pre-run review corrected the reversed case's expected value from
`indigo` to the source's literal `Indigo`. The document and request are unchanged;
neither the encoder nor motor normalizes copied answer bytes. This correction
preceded every run on these cases.

Observed failures: exact two-slot reading does not acquire useful frames from
the original exam passage; type annotations break the lexical action walk.
Hold the Python layout fixed here to isolate a broader language representation.

One candidate is allowed: frozen all-MiniLM-L6-v2 token representations, with
no QA head or generated answers. A shared neuronal GO/NO_GO synapse bank learns
span/evidence selection. All arms retain identical source-derived token spans
and contextual features. This is disclosed hybrid episodic storage, including
raw text; source removal means removal from live input, not semantic compression.
Pretraining includes QA/code data and exam contamination is unknown.

Use the existing learned occurrence motor and the real Python executor. Two
executed demonstrations establish a shared motor prior. It selects one old
lexical occurrence and copies the selected evidence span there, including
multiword spans. Wrong values or malformed Python receive executed failure
feedback; the host does not filter candidates to expected answer shapes.
For each teaching case:
ingest its raw document, record a first response and actual execution, then allow
one exploration proposal from the external-only similarity rule if the neuron
race stopped. Update only the executed candidate from its binary outcome.
Afterward an executed teacher repair supplies one positive span demonstration;
ambiguous repeated teacher bytes must reject instead of selecting a favored span.
The teacher is never counted as an autonomous success. Each learning rate has
a paired frozen-synapse control. It receives the same ingestions and the exact
executed updater action/outcome ledger, rebound to its own current checkpoint.
Its separate first attempts measure its policy; they do not alter that ledger.

Compare learning rates 1 and 3, fixed before the run. Each performs at most one
native local update per feedback event. The primary control freezes synapses
while receiving identical ingestion, encoder features and motor prior. Also
report the external-only query/span similarity rule and a conventional selector
using the learned linear drive without spikes. These distinguish benefit from
stored data, local learning, and physical competition. Shared bias changes alone
must not be called learned role selection.

Ingest the six transfer documents only after teaching; remove live sources.
Allow one response per transfer request with no teaching, tuning or hidden retry.
Score executable output, unchanged decoys, earlier-task retention, the unsupported
case's no-action, and each language stratum separately. The original KS2 Q6a/b
questions remain a separate untuned diagnostic using official marking guidance.

Completion of this comparison requires results for every declared arm/case,
including failures and timeouts, with source/encoder identity, feedback count,
synaptic change, storage bytes and timing. A useful capability gain requires
fresh executable transfer over frozen synapses and retained earlier skills;
matching the conventional learned selector does not establish a special benefit
from spikes. Failure to improve must return us to representation/learning design.

Budget: one encoder, the two learning rates, eight teaching and six transfer
cases per arm, at most two engineering corrections, one engineering day total.
No changes to fixture answers or question forms after the first run. No protected
R5/R6 evaluation or external-model SWE comparison is authorized by this pilot.

## Reproduce

Use the checkout's environment and attested native preflight from
[development.md](../../docs/development.md). Install the optional feature runtime:

```powershell
./.venv/Scripts/python.exe -m pip install -e '.[contextual]'
```

The encoder uses the pinned revision and file hashes in
[`frozen_context_encoder.py`](../../src/humanbrain/frozen_context_encoder.py).
Download these public model files once; inference never downloads anything:

```powershell
@'
import json
from pathlib import Path
from urllib.request import urlretrieve
from humanbrain.frozen_context_encoder import FILE_HASHES, MODEL_ID, MODEL_REVISION
root = Path('.models/all-MiniLM-L6-v2')
for name in FILE_HASHES:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(f'https://huggingface.co/{MODEL_ID}/resolve/{MODEL_REVISION}/{name}', path)
(root / 'manifest.json').write_text(json.dumps({'model_id': MODEL_ID, 'revision': MODEL_REVISION}))
'@ | ./.venv/Scripts/python.exe -
./.venv/Scripts/python.exe tools/dev.py doctor
./.venv/Scripts/python.exe tools/dev.py native
./.venv/Scripts/python.exe -m experiments.r4_integrated.contextual_evidence_python_session --output benchmarks/contextual_evidence_python_development_v1.json
```

Choose a new output name for a repeat. The runner retains materialized Python
workspaces and flushed events under `.runtime/contextual-python-*`, writes the
complete JSON result even on a caught execution error, and saves trained
synapses in an adjacent `.npz`. The raw documents and contextual vectors remain
external storage and are reconstructed from the ordered fixtures and pinned
encoder; this is not a compressed semantic-memory checkpoint. Report whole-run
process memory separately from per-arm retained byte/array payloads.
