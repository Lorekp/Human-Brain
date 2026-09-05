# Candidate-relative binding and spike interpolation: result

Completed 2026-09-05 from preregistered source commit `46b6870` under the
[binding protocol](contextual_binding_protocol.md). The
[artifact](../../benchmarks/contextual_binding_python_development_v1.json) and
[synapses](../../benchmarks/contextual_binding_python_development_v1.npz) preserve
the single fixed comparison. These self-authored fixtures are public development;
no protected evaluation ran and no generalization gate passes.

## Executed behavior

| Policy | Fresh edits | Retained edits | Both-property retained entities | Unsupported STOPs |
| --- | ---: | ---: | ---: | ---: |
| GAIN16 discrete neurons | 0/8 | 4/16 | 0/8 | 1/2 |
| GAIN16 interpolated neurons | 0/8 | 9/16 | 3/8 | 0/2 |
| GAIN16 raw or mapped learned-score reference | 0/8 | 9/16 | 3/8 | 0/2 |
| BINDING discrete/interpolated neurons or raw/mapped references | 0/8 | 0/16 | 0/8 | 2/2 |
| BINDING raw reference with new features disabled | 0/8 | 1/16 | 0/8 | 0/2 |
| BINDING raw reference with positional kernels collapsed | 0/8 | 0/16 | 0/8 | 2/2 |
| Frozen neurons, either readout/representation | 0/8 | 0/16 | 0/8 | 2/2 |
| External encoder cosine, either representation | 0/8 | 0/16 | 0/8 | 0/2 |

Interpolation recovers five correct retained GAIN16 selections at unchanged
weights. Its winners agree with raw/mapped references on this cohort. This is
useful neuronal readout recovery, not better ranking or fresh-task learning.
The previous cohort's 11/16 host retention is a different measurement; the table
compares readouts on the new paired-entity cohort at identical checkpoints.

BINDING STOPs on all 26 fresh/retained requests, including every supported one.
Its unsupported STOPs therefore do not demonstrate selective abstention. Removing
the new block yields a constant answer; collapsing its positional kernels still
STOPs throughout. The representation has not demonstrated a capability gain.
All selected actions preserve decoys.

## What the feedback actually taught

Each arm executes 48 teacher PASS repairs, admits 32 actual FAIL/PASS pairs,
and offers 118 certified replays. Frozen mates receive the same 300 feedback
events as their updater and keep zero synapses. Evaluation changes no checkpoint.
All policies share the fixed two-demonstration neuronal motor: these controls
isolate evidence learning/selection, not removal of the entire neuronal system.

GAIN16 applies 70 native updates: 32 new and 38 replay. It makes five successful
neuronal first attempts and six successful fallbacks. Its admitted negatives
comprise thirteen neuronal selections, eighteen raw-score proposals and one
external proposal.

BINDING applies exactly one native update, on its first correction. The other
31 new offers and all 118 replays make no update. Both learned readouts STOP
during teaching, so all 48 exploration attempts come from external cosine and
all 32 admitted negatives use that policy. None succeeds. Different on-policy
negatives are a material part of this result; it is not a comparison on identical
training contrasts. Teacher/fallback execution never counts as neuronal success
or autonomous recovery.

## Cost, exam pressure and verification

The complete run takes 187.05 seconds including child execution waits. Parent
peak working set is 324.32 MiB; final private memory is 575.63 MiB and traced
Python peak is 115.06 MiB. Child executor memory is excluded. GAIN16/BINDING
teaching takes 47.91/76.71 seconds; fresh evaluation 7.79/12.01; retention
12.25/19.27; original exam 1.49/8.86. BINDING also evaluates two extra ablations,
so whole-phase times alone do not isolate feature-computation overhead.

Each learner ends with eleven documents, 4,719 raw source bytes, 2,298 candidates
and 23 windows. Counted array/byte payload is 7,207,344 bytes for GAIN16 and
7,256,453 for BINDING, including BINDING's 47,093 bytes of interaction maps and
casefolded words. Compact correction ledgers add 2,139,428/2,234,757 bytes;
numeric eligibility payloads of 786,944/829,952 bytes overlap those serialized
figures. Python object overhead and the shared 90,405,214-byte pretrained encoder
are separate. All evaluation requests have zero live documents, but raw sources
and derived vectors remain in external retained memory.

The original KS2 Q6a/b remains untuned. GAIN16 raw/mapped/interpolated answers
are `worms and beetle` and `beetle`, pending official-rubric human scoring.
Discrete GAIN16 gives one STOP and one pending answer. BINDING learned/neuron
policies and all frozen variants STOP at 0/2. External `to` answers remain
pending. No exam improvement is established.

All 45 focused tests pass with the real encoder/native implementation. Fixed
four-current interpolation checks are within 0.004512 ms of direct `dt/64`
integration; the `dt/16` and `dt/64` results differ by at most 0.001563 ms.
This is a numerical approximation between coarse GLIF endpoints, not exact
continuous dynamics. Exact/adjacent-float64 crossing ties still STOP. Source,
NPZ, archived synapse and actual checkout-native DLL hashes verify; existing
discrete semantics and completed artifacts remain unchanged.

## Diagnosis and next decision

The [read-only score/receipt diagnostic](../../benchmarks/contextual_binding_diagnostic_v1.json)
reconstructs saved training/evaluation checkpoints and all 416 ordinary fresh
and retention policy decisions before interpreting scores. It takes 13.14 seconds
and performs no learning, new actions or new requests.
BINDING's unthresholded leader is wrong on all sixteen retained and eight
supported fresh requests. Lowering the STOP threshold alone would therefore
recover no correct answer. All 32 recorded pair margins are nevertheless
satisfied: minimum 8.052, median 18.765, maximum 35.370. Its saved weights match
the single first-update formula within 1.11e-16. None of its current wrong
leaders was an executed negative for the same query.

The failure already exists before fresh ingestion. Correct retained occurrences
rank 35–497 afterward; supported fresh answers rank 149–445. Negative direct
semantic-match features dominate suppression of the correct occurrences. This
is a learned scoring failure as well as abstention, not hidden successful recall.

The fitted external-negative contrasts are insufficient training evidence for
global selection. This does not establish that the proximity representation
lacks the required distinctions; it has barely been trained against its own
errors. Do not extend replay or tune these features on this cohort.

Next execute the unique highest raw candidate as explicitly labeled training
exploration even when its score falls below STOP, using external cosine only
when candidates have no unique raw leader. A wrong executed leader necessarily
gives a violated teacher-versus-leader margin, allowing informative native
updates. Keep evaluation STOP and exact ties unchanged; exploration successes
remain separate from neuronal successes. If correct leaders later remain below
STOP, that would motivate a separately declared action-versus-STOP learning
contract. Return directly to one bounded integrated comparison, as specified
by the [active plan](../../docs/research/balanced-evidence-to-action-session-v1.md).
