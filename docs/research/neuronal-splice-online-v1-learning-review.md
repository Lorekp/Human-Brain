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

## One next scientific question

**Why does the preserved reader terminate immediately on new raw action requests:
does it address EOS-producing locations, or does its codec emit EOS even at the
requested literal content?** This separates loss of content routing under changed
request context from a stopping/readout problem. Old-question retention alone
does not distinguish them. The endpoint failure labels and empty strings do not
either. A changed request, deletion supervision and codec adaptation are plausible
contributors, not established causes.

Recommend one bounded saved-receipt attribution before choosing another learning
mechanism or acquisition protocol. For the learned fresh nonempty-payload failures
that emit EOS first, use the already saved first prior `p_s`, posterior `b_s` and
mixture EOS probability `m`. The recorded filtering equation implies
`c_s(EOS) = m * exp(log b_s - log p_s)` and actual slot contribution
`p_s * c_s(EOS) = m * b_s`. Compute in log space, check finite normalization and
rounding, and group actual contributions at target-value starts, other value
starts, punctuation/document EOS and remaining positions. Position labels belong
only to this external saved-data analysis of the authenticated raw documents;
they must not enter neuronal observations or teaching. Contrast the corresponding
saved slow-frozen and action-frozen cases, without claiming that those arms hold
every downstream activity fixed.

This reconstruction identifies the conditional probability of the token actually
observed. It cannot recover every unsaved codec output, establish a corrected
counterfactual generation, or prove that changing one component would solve the
task. Its purpose is to select a justified single follow-up while preserving the
working local action circuit. Do not change epochs, rate, seed, recall policy,
request preprocessing or the success gate from these endpoint scores. Any later
mechanism must continue to receive the identical unparsed raw request and act and
learn in the same instance. No implementation, diagnostic execution or fit is
authorized by this recommendation.

## Evidence identities

The review read the saved report/audit and streamed the worker proposal journal;
the independent descriptive summary supplies complete head-error counts.

- Decision: `0306bacaf835b88dca7219a0aee3e567b0a9669e7ea7de4dff87f49aa3fdfb00`.
- Run report: `862918bf7116d88baf337965c3e8177fad02efb40021628b0248afcba6264a4f`.
- Independent audit: `96cbf7f155beb8efea0f3541cce535a7ab727661293e93e9525f0312af7afe6d`.
- Worker endpoint journal: `21d83aedf0a1f0f4bcdf1ebac408e69147646fdc0c5995b5f8ba9603d48a0bbd`.
- Outcome summary: `8be8f588acd0db1428a636e1581dd28c5ea3a4a1efc594489489f3bbdbbef6e1`.
- Reader source: `2a6bfe480757b248c585a00bdc9c79155af6abd3a162b3e93a098d406b73a3b8`.
- Action learner source: `bb06e074d44431cf2d09d62f7e98119a2a207ed53e7f6d5796bb6c7b274770ab`.
