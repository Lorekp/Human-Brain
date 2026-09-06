# Neuronal splice v1: what the completed learning run establishes

Saved-evidence review, 2026-09-06. No checkpoint was opened, model constructed,
neural forward performed or action repeated. The frozen
[protocol](../../experiments/r4_integrated/neuronal_splice_online_v1.md) and its
criterion remain unchanged.

The [banked decision](../../benchmarks/neuronal_splice_online_v1.decision.json)
establishes useful but incomplete neuronal edit selection: learned succeeds on
12/32 fresh tasks and 12/16 retention tasks, versus fresh scores 1, 2 and 4 for
action-frozen, slow-frozen and memory-disabled controls. Acquisition first-action
successes rise from 5/32 in round one to 19/32 in round eight, totaling 101/256.
The learned reader preserves the same eighteen correct old questions before and
after acquisition. Combined run and audit cost is 96.1418275 seconds. The scoped
criterion fails; none of these figures establishes general software competence.

The meaningful advance over v6 is that neurons now choose the file, both raw
boundaries, replacement bytes and STOP/SPLICE. The host applies the actual proposal
and verifies it once; it supplies no correct span or repaired payload. The same
persistent learner receives supervised correction after execution. Current source
uses direct output-neuron errors on each synapse's own forward feature activity,
with no upstream reverse-Jacobian transport. Saved custody supports the claimed
sequence of events; it does not independently reconstruct unsaved synapse updates.
Nor is supervised correction evidence of reward-driven autonomous recovery.

The saved outcome summary narrows the unresolved question. All 28 fresh SPLICE
tasks have the canonical file/start/end, but nineteen have wrong payloads. The
remaining fresh failure is a missed STOP. All eight V value-swap tasks emit empty
EOS; none succeeds. Four of sixteen fresh contrast pairs pass both members.
All four retention failures have wrong payloads, and one also has a wrong span.
There are no endpoint rejection or process failures. Thus a new span selector or
executor optimization is not the evidence-led next intervention. The apparent
location transfer is still limited to the authored short-file layouts, explicit
paths and one seed; it is not proof that arbitrary duplicate contexts are resolved.

## Completed attribution and the next source/math question

The completed saved attribution covers **48 learned-arm endpoints only**. It does
not attribute matched-control outputs. For an observed first EOS, the recorded
prior `p_s`, posterior `b_s` and mixture probability `m` imply
`c_s(EOS) = m * exp(log b_s - log p_s)` and contribution
`p_s * c_s(EOS) = m * b_s`. This uses saved filtering activity, not a new codec
evaluation. Raw target/document indexing is external posthoc analysis and must
never enter neuronal observations or teaching. Empty values have no payload-byte
start, and STOP has no canonical reader teacher.

Across the eight V cases, the intended start receives prior mass 0.03345–0.09369.
Its reconstructed conditional EOS probability is only 0.01835–0.01963, contributing
0.000615–0.001839 to total EOS probability 0.14864–0.16550. Actual EOS sensory
slots contribute 0.03400–0.04297; all other non-EOS slots contribute
0.11242–0.12115. The three contribution categories are disjoint. Diffuse off-target
activity therefore dominates observed EOS; the intended literal start itself is
not a strong EOS producer in these cases. Among nonempty SPLICE endpoints, mean
target-start prior is 0.14105 for twelve successes and 0.09520 for twenty-three
failures, twenty of which emit first-step EOS. This association does not identify
why the learned address distribution has that shape.

The completed saved rank pass finds the intended literal start uniquely ranked
first in all eight V cases, in 22/23 failed nonempty SPLICE cases, and in all
12/12 successful nonempty SPLICE cases. The sole failed rank-two case is T01:
its target-minus-top log margin is -0.04286336898803689. Thus correct ranking
with diffuse activity, rather than wrong top ranking, describes the dominant
observed failure pattern. It does not prove that selecting the top location would
generate the right complete payload, and it does not erase the remaining
misranking or STOP failure.

Recommend **one bounded source/math review of neuronal address competition and
concentration before payload generation**. Its question is whether an explicit
neuronal competition circuit can turn learned address preferences into useful
content retrieval while preserving direct local learning. Require the proposed
forward state, competition dynamics, stopping/continuation semantics, local
feedback ownership and actual arithmetic/storage costs to be concrete. The same
persistent circuit must act and receive correction; a changed forward policy
cannot silently retain an incompatible teacher computation or introduce reverse
error transport. Preserve the action heads and identical unparsed raw request.

That review must account for the tradeoff between suppressing off-target pooled
emission and losing valid duplicate evidence or prefix disambiguation. It should
produce one justified mechanism or identify an unresolved blocker, not a policy
or gain sweep. Neither WTA nor temperature adjustment is selected by these ranks.
Request-context interference, deletion supervision and codec adaptation remain
candidate explanations for the learned distribution, not established causes.

This reconstruction identifies the conditional probability of the token actually
observed. It cannot recover every unsaved codec output, establish a corrected
counterfactual generation, or prove that changing one component would solve the
task. Its purpose is to select a justified single follow-up while preserving the
working local action circuit. Do not change epochs, rate, seed, recall policy,
request preprocessing or the success gate from these endpoint scores. Any later
mechanism must continue to receive the identical unparsed raw request and act and
learn in the same instance. No implementation, diagnostic execution or fit is
authorized by this recommendation.

The v1 endpoints are now exposed development evidence. Any successor comparison
must preregister new withheld cases before their outcomes are observed, retain
matched controls and old-reading retention, and obtain fresh execution admission.
The frozen v1 criterion and failures cannot be revised into a passing benchmark.

## Evidence identities

The review read the saved report/audit and streamed the worker proposal journal;
the independent descriptive summary supplies complete head-error counts.

- Decision: `0306bacaf835b88dca7219a0aee3e567b0a9669e7ea7de4dff87f49aa3fdfb00`.
- Run report: `862918bf7116d88baf337965c3e8177fad02efb40021628b0248afcba6264a4f`.
- Independent audit: `96cbf7f155beb8efea0f3541cce535a7ab727661293e93e9525f0312af7afe6d`.
- Worker endpoint journal: `21d83aedf0a1f0f4bcdf1ebac408e69147646fdc0c5995b5f8ba9603d48a0bbd`.
- Outcome summary: `8be8f588acd0db1428a636e1581dd28c5ea3a4a1efc594489489f3bbdbbef6e1`.
- Learned reader attribution: `c2b55caea8da7ba7266345263de24dc8b650d119a88dd75bc0a33b5837809ecf`.
- Learned address ranks: `07541baa6c17eba19c3bad670dbc4c0b253880dfcc98d0350616b8ebdf1bda5e`.
- Reader source: `2a6bfe480757b248c585a00bdc9c79155af6abd3a162b3e93a098d406b73a3b8`.
- Action learner source: `bb06e074d44431cf2d09d62f7e98119a2a207ed53e7f6d5796bb6c7b274770ab`.
