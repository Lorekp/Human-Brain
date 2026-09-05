# Executed own-error exploration: result

Completed 2026-09-05 from source `6120335` under the
[fixed protocol](contextual_exploration_protocol.md). The single comparison is
preserved in the [artifact](../../benchmarks/contextual_exploration_python_development_v1.json)
and [synapses](../../benchmarks/contextual_exploration_python_development_v1.npz).
Teaching is openly reused; transfer uses new instances of the declared public
development strata. No protected evaluation or generalization gate passes.

## Executed outcomes

| Policy | Fresh edits | Retained edits | Unsupported STOPs |
| --- | ---: | ---: | ---: |
| GAIN16 discrete neurons | 0/8 | 4/16 | 1/2 |
| GAIN16 interpolated neurons / raw or mapped reference | 0/8 | 9/16 | 0/2 |
| BINDING raw-score reference | 4/8 | 2/16 | 0/2 |
| BINDING discrete / interpolated neurons / mapped reference | 0/8 | 0/16 | 2/2 |
| BINDING raw reference, new block disabled | 0/8 | 2/16 | 0/2 |
| BINDING raw reference, positional kernels collapsed | 4/8 | 5/16 | 0/2 |
| Frozen neurons, either representation/readout | 0/8 | 0/16 | 2/2 |
| External cosine, either representation | 0/8 | 0/16 | 0/2 |

BINDING's raw successes are paxi/transport, qiro/transport, ruva/transport and
seki/audit. No fresh entity has both properties correct. Collapsing position
exchanges ruva/transport for ruva/audit without changing the total, and improves
old selection. The new feature block contributes to this trained scorer, but
these ablations do not establish that its positional geometry improves behavior.
Only GAIN16's interpolated/raw/mapped policies retain both properties for three
old entities. All policies preserve decoys. Teachers and exploration fallbacks
never count as neuronal first-attempt success or autonomous recovery.

## Feedback coverage is repaired; current mapping loses the gains

Both arms admit 32 actual FAIL/teacher-PASS pairs and offer 118 certified older
replays. Their frozen mates receive the same 300 feedback events and apply zero
updates. GAIN16 applies 70 native updates, comprising 32 new and 38 replay;
BINDING applies 80, comprising 32 new and 48 replay. BINDING now admits one
external negative and 31 actual raw leaders, rather than only external proposals.
Every admitted raw-leader correction changes synapses: its teacher-minus-leader
gap is negative beforehand, ranging from -26.394 to -0.069 for BINDING, and
positive immediately afterward, ranging from 0.218 to 18.398. This is effective
feedback on actual mistakes, not merely nonzero weights.

BINDING's 48 first attempts all STOP; its 47 raw and one external fallbacks yield
six successes. Three raw proposals bypass evaluation STOP early in training.
GAIN16 has five first-attempt and six fallback successes; its 25 raw and one
external fallbacks never bypass the raw threshold. Its admitted sources remain
13 neuronal, 18 raw and one external. Exposure streams differ between updaters;
comparison with the previous exploration policy is historical, not concurrent.

At the final checkpoint all 26 BINDING raw leaders are unique, with drives
26.697–58.794. Every supported correct occurrence is also well above the fixed
STOP drive. However, `600 + 200*tanh(drive)` rounds these large positive drives
to exactly 800 pA. The interpolated race reports 51–492 indistinguishable first
crossings. The mapped-score reference already loses the distinctions, so finer
spike timing cannot restore them. This is numerical saturation, not insufficient
activation; lowering STOP would not address it.

The raw scorer still ranks four fresh and fourteen old answers incorrectly.
Their best correct ranks range from 7–42 and 2–177 respectively. Fixing the
current map cannot repair these rankings or establish selective abstention.
The raw scorer wrongly acts on both unsupported probes; neuronal STOPs on every
query are not learned uncertainty. All final diagnostics and evaluations leave
their checkpoints unchanged.

## Resources and verification

The run takes 232.46 seconds, including child execution waits. Parent peak
working set is 330.34 MiB; final private memory is 586.83 MiB and traced Python
peak is 115.35 MiB. Child memory is excluded. GAIN16/BINDING teaching takes
60.94/99.87 seconds, transfer 9.90/12.73, retention 15.78/19.86, exam 1.71/8.85,
and read-only rank diagnostics 0.54/1.35. BINDING has two extra evaluation
policies, so whole-phase times do not isolate feature overhead.

Each learner finishes with eleven documents, 4,708 raw bytes, 2,298 candidates
and 23 windows. Counted array/byte payload is 7,207,267 bytes for GAIN16 and
7,256,365 for BINDING, including 47,082 bytes of interaction maps/words.
Compact correction ledgers add 2,139,569/2,232,309 bytes; eligibility arrays
overlap those serialized figures. Python object overhead and the shared
90,405,214-byte pretrained encoder remain separate. Zero live documents in
requests does not remove retained external sources or vectors. All controls
share the fixed two-demonstration neuronal motor and isolate evidence selection,
not the entire neuronal system.

Original KS2 Q6 remains untuned. BINDING raw answers `tenrec that live` and
`tenrec`, GAIN16 raw/interpolated answers `worms and beetle` and `beetle`, and
external `to` answers remain pending official-rubric human scoring. BINDING
neurons and both frozen readouts STOP at 0/2. No exam improvement is established.

All 33 focused core tests pass, including real native correction and frozen
replay for both feature policies. Inert runner checks preserve historical default
traces and verify exploration routing, budgets and evaluator-only ranks. Ruff
and diff checks pass. All 33 executable/source inventory files match the frozen
commit; NPZ, archived synapse and loaded checkout-native DLL hashes verify.

## Next decision

Integrate the learned raw selections through one explicitly named, numerically
bounded current-mapping policy, first at these saved weights and on these openly
reused tasks. Preserve ranking against tonic STOP, true ties and frozen behavior;
keep the existing policy and spent result unchanged. This is a readout comparison,
not a new generalization claim. Do not change representation, native learning,
exploration or training budget in that comparison. Afterward return to the
remaining relation-selection and retention failures; no extra epochs or feature
tuning on this transfer cohort. The
[active plan](../../docs/research/balanced-evidence-to-action-session-v1.md)
owns the bounded follow-up and advancement targets.
