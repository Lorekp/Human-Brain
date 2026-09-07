# Persistent language/action v3: saved learning review

The completed continuation changes neuronal outputs but establishes no task
competence. Actual-path correction reaches the cells that acted; its shared
positive feature population also couples corrections for different bytes strongly.
The next design should address both that interference and the fact that only the
first error receives output teaching, even though the complete answer is offered.

This is a saved-only review of source `fc7f614cfbc7c01489245c6fec05520a9d1c586f`.
The full run and independent review passed on their first launches. The extraction
read the completed report, manifest, completion, review and each journal once;
it used no model, checkpoint deserialization, new prediction, teacher, execution,
alternate decoding or auditor replay. All 56 episodes are now exposed development
material. The [result report](../../experiments/r4_integrated/persistent_language_action_v3_result.md)
owns the complete run and custody record.

## Behavior and persistence

| Route | Acquisition first responses | Fresh transfer | Teacher-free recovery | Delayed retention |
|---|---:|---:|---:|---:|
| Online | 0/24 | 0/16 | 0/8 | 0/8 |
| Learning disabled | 0/24 | 0/16 | 0/8 | 0/8 |
| Simple retrieval/edit reference | 12/24 | 6/16 | 0/8 | 8/8 |

Each neuronal route continues an independent copy of the same v2 final numerical
state and whole 505-slot tape. This session adds no documents: the empty ingest
journal is intentional. Online publishes learning; the disabled route preserves
its numerical parameters. Recovery always performs three attempts without a
teacher; neither any later attempt nor the final attempt succeeds. The eight
neuronal retention items were already incorrect at their earlier measured
responses, so these are continued failures, not eight measured instances of
forgetting. The simple reference retains eight previously correct cases.

There are two unsupported reading and two unsupported edit requests per neuronal
route; their raw inputs are not truncated. The reference succeeds on 5/12 fresh
cases within the common capacity and 6/16 overall. Its acquisition splits evenly
between reading and editing (6/12 each); fresh reading is 2/8 and editing 4/8.
The [separate reference review](persistent-language-action-v3-evaluation-review.md)
owns attribution and instruction-family overlap. These small public streams do
not establish broad instruction generalization for any route.

## Actual output and action failures

Each neuronal route makes 68 supported free calls: 22 reading and 46 edit payloads,
including retries. Online uses 1,125 output decisions and disabled 1,124.
Both have 20 reading responses at the 16-byte limit and two terminating at EOS.
Online's EOS responses include one empty response; disabled's two are empty.
Every one of the 46 edit payloads in each route hits the limit without EOS.
No neuronal reading or payload equals its raw target; none contains the complete
target literal as a substring. This descriptive substring check awards no credit.

Among matched first responses, online differs from disabled on 7/22 reading
outputs and 12/30 supported edit payloads. Selected-address sequences differ on
3/22 and 6/30 respectively. Online's 46 edit generations make 99 address switches
across ten calls, versus 34 across five disabled calls. Reretrieval and plasticity
therefore have observable effects, but those effects do not produce correct
answers or termination. Constant outputs remain common: 18/22 online reading
responses and 27/46 edit payloads, counting the empty response as constant.

All 30 supported first edit proposals in each route choose SPLICE. Online matches
the canonical path in 24 cases, start in one and end in two; disabled matches
23, one and two. These diagnostics overlap and canonical equality is not the
success criterion. Missing payload EOS prevents application of every supported
proposal. Including unsupported requests, each route has 48 genuine unchanged
workspace verifications and 48 rejected applications. Their 24 nonzero return
codes across retries come from the unchanged authored failing programs; they
are not evidence of model-applied destructive edits. STOP/span errors also remain,
but an operation-only intervention would leave the shared answer failure intact.

## Within-concept learning and teacher use

The manifest pairs six concepts across reading and editing in two acquisition
blocks. Online and disabled score 0/12 in both blocks; the reference moves from
4/12 to 8/12. All twelve online paired emissions change, but ten retain a first
error at position zero. Two edit pairs regress from a correct opening quote to a
position-zero error. No pair gains a longer correct prefix. One reading response
changes from a limit response to immediate EOS without becoming correct.
The blocks use related instruction forms, not repeated identical inputs, so
these are paired transfer observations rather than isolated update-effect tests.

Per neuronal route, 24 corrections expose 148 target bytes plus 24 EOS labels:
172 all-address B/C offers. Editing adds 36 action-head labels. Online publishes
24 actual-path U clamps; disabled computes 24 proposals and publishes none.
Twenty-two clamps target the first output event and two the second. **No output
clamp targets EOS.** Thus complete answers and EOS are consumed by address teaching,
but the output mechanism learns only one event from each offered sequence.
Making its representation sparse alone would not remove this coverage limit.

Online's offer-weighted teacher NLL is 3.97875, versus 3.92790 disabled. These are
actual teacher-prefix likelihoods during staged address teaching, not free-answer
scores or post-update probes. They do not show improved free sequence learning.
All-address B/C teaching holds U fixed, and the actual-path clamp uses the saved
free feature and currents after the real response/action. No target address or
teacher prefix is substituted into that actual event.

## What the recorded clamp features establish

The 24 online correction features have energy 32. Their 276 unordered cosine
overlaps average 0.80728 (range 0.73688–0.96281). The 221 pairs with different
target bytes average 0.80743; the 55 same-target pairs average 0.80669. This is
strong coupling across these observed contexts, without useful separation by
the required byte. It is not a proof of feature identity or a general separability
theorem.

For the two corrections at position one, the earlier actual opening quote was
correct. The correction feature overlaps that earlier feature by 0.93890 and
0.93842. From the actual clamp
`Delta U = (T - z_error) x_error^T / ||x_error||^2`, its contribution to the earlier
correct-versus-recorded-competitor margin is -6.59597 and -6.62389 respectively.
The recorded earlier margins were 1.54272 and 0.94714. These are scalar synaptic
effects at held recorded activities, not a regenerated output or a later observed
margin. They establish a concrete interference direction; they do not attribute
later errors uniquely to it, because addresses, contexts and other updates also
change. The analysis never applies an alternative decoder to saved state.

## One next architectural priority

Review one context-specific plastic output assembly mechanism shared by reading
and action payloads, with locally consumed causal prefix events for the **whole
already-offered sequence**, including EOS. Its purpose is to retain distinct
corrections and cover later sequence positions together. Input-driven assembly
selection/allocation must precede the current target; each output cell may use
only its own activity and local target-current error. Preserve the actual-response
record separately from the explicitly teacher-driven later prefix states.

This is a bounded source/math recommendation, not an implementation or run
authorization. Sparse assemblies alone cannot teach omitted events; consuming
all events alone in the same strongly overlapping population can introduce more
interference. The design must state finite capacity, novelty/reuse policy,
free-versus-teacher prefix mismatch and generalization limits. Continue the saved
learner on broader instruction, composition and feedback pressures rather than
adding epochs or requiring perfection on this exposed cohort.

## Evidence bindings

The ignored extraction is `.pytest-tmp/active-path-v3-learning-extraction.json`,
SHA256 `40b752cc924528455f4a44c3da8cda84d25e49b96acd258f20674560f45a4fce`.
It contains compact actual attempts, teacher summaries, paired concepts, the
selected correction vectors and scalar overlap calculations. Its hashes bind:

| Saved artifact | SHA256 |
|---|---|
| Report | `6686e9d2482a5868ff989e628ced6580f9180fd7340ec2ac42cc92bc9cbd7a8b` |
| Manifest | `fc83513a6d6accee744c5d03188bc9cec75022db3cd00ff5c49f561745aa3056` |
| Completion | `a9531d61dfc9059eddd9859ad57031bad75c71f24e282b1f70e117aa9c767fcc` |
| Independent review | `5b44e096d1ae5a81c4dc93646007c24497e14aed50316f663c66a81f4d3af6e8` |
| Attempts (216 rows) | `14bab3ac2908b3977736cadfa5886db2786a73dc48c504630213fa60eabe72d9` |
| Teachers (72 rows) | `b785cb87167e33233a81bb7125e3d2f8cdaa0e60b1009fe9ea100b385099f08f` |
| Ingests (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |

The full run plus independent review take 60.5972151 seconds; the larger sampled
serial peak is 512.5234375 MiB. The [resource review](persistent-language-action-v3-cost-review.md)
separates exact work savings from non-causal timing comparisons. Budget compliance
does not change the negative capability result.
