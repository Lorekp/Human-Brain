# Contextual learning after the pairwise failure

Date: 2026-09-05. Bounded research recommendation; no new model run.

## Evidence and its limits

The [executed comparison](../../benchmarks/contextual_pairwise_python_development_v1.json)
completed with unchanged source hashes. ABSOLUTE and PAIRWISE each achieved
0/8 supported fresh edits, 1/16 retained edits, and 0/2 unsupported STOPs.
Each emitted one constant answer; conventional learned-score selection and
spiking selection disagreed on none of the fresh or retention cases.
This rules out spike resolution as the cause of these recorded ranking failures.
It does not establish that MiniLM lacks usable relations or that neuronal
learning cannot acquire them.

In [`_features`](../../src/humanbrain/contextual_evidence_memory.py), the
query-only block cancels between candidates. The query-times-window block also
cancels between spans from the same window. The span-only block can support
global answer preferences; query-times-span can still express some conditional
ranking. This is a structural limitation, not a proof of insufficient capacity.

The independent [receipt diagnostic](../../benchmarks/contextual_pairwise_diagnostic_v1.json)
confirms that all sixteen executed pairs form one acyclic value chain. A
query-independent span-only linear fit satisfies all sixteen margins to numerical
tolerance (maximum residual 1.15e-14). Meanwhile, ten of eleven initially
margin-satisfied pairs were later forgotten; 25 native replay passes still leave
five margin violations and one wrong pair. Optimization interference exists,
but fitting this ledger cannot establish query dependence. Replaying its chain
cannot manufacture the missing contradictory supervision.

## Three concrete options

| Option | Published support and useful mechanism | Limitation here |
| --- | --- | --- |
| Execution-grounded revisitation with episodic replay | [Chaudhry et al., 2019](https://arxiv.org/abs/1902.10486) found that jointly training on current and small retained experience buffers was effective across four continual-learning benchmarks. Keep actual correction pairs and rehearse them through the existing plasticity path. | Replay preserves supplied distinctions; it cannot supply missing contrasts. The paper does not establish local neuronal learning or our language task. |
| Candidate-centered token interactions | [Khattab and Zaharia, 2020, sections 3.1–3.3](https://arxiv.org/html/2004.12832v2) retain contextual query/document token vectors and combine per-query-token matches instead of reducing both inputs immediately to one vector. Our existing frozen encoder already exposes token vectors and offsets. | Their trained passage ranker is not an answer-span learner. Whole-window MaxSim gives both answers in the same window identical context scores. A proposed span adaptation must preserve candidate-relative token structure and learn which interactions matter; it is not an off-the-shelf ColBERT result. |
| Episodic synaptic fast weights | [Ba et al., 2016, sections 3–4](https://papers.nips.cc/paper/2016/file/9f44e956e3a2b7b5598c625fcc802c36-Paper.pdf) use decaying outer-product memory and iterative retrieval, with learned slow weights, to retain recent activity patterns and solve associative retrieval tasks. | Fast storage does not identify the right semantic key or role. Their slow weights use gradient training; temporary memory is not demonstrated lifelong consolidation. Adding it now changes both representation and memory without isolating the present failure. |

## Recommended next change: close the teaching loop

Keep the frozen encoder, candidate family, pairwise eligibility, native plasticity,
motor prior, and cell race. Add **bounded revisitation and receipt-backed replay**,
before adopting another representation. This is a learning-loop intervention,
not a new semantic filter. Its purpose is to make conditional behavior necessary
and test whether retained corrections survive subsequent learning.

Preregister one fresh development cohort of eight teaching entities with two
properties each; retain all sixteen raw teaching requests. Present each pair,
then revisit the earlier request after teaching the other property. Interleave
old requests by a declared schedule, without inspecting held-out answers.
A request order A, B, A alone is insufficient: count opposite executed
preferences or cycles on identical retained candidate occurrences. If they do
not arise, report the query-dependence admission test as unmet.

Only an actually executed wrong candidate followed by an executed teacher PASS
creates a new pair. Successful autonomous attempts earn no fabricated negative.
Store up to 32 immutable pairs with raw query, candidate/source identities,
workspace hashes, execution receipts, and parent checkpoints. Rehearsal uses
the same certified pair at the current checkpoint, reconstructing its eligibility;
it must not submit a stale decision or label a new candidate from an answer key.
Use one fixed cyclic sweep of at most four older pairs after each new correction.
All candidate scoring stays unfiltered; no entity, document, or answer lookup
chooses the inference winner. The external ledger supplies experience only.

Compare revisitation alone against revisitation plus replay, each with a same-seed
learning-disabled arm replaying its exact feedback. Their on-policy negatives
may differ; report both ledgers rather than claiming identical supervision.
Keep external-only and conventional learned-score references. Add a diagnostic
query-independent readout or query-shuffled eligibility replay on the same
certified ledger to test whether experience actually requires query dependence.
The native learner must change candidate-ordering synapses; measure pair margins,
old-query accuracy after each new correction, and checkpoint changes. Identical
conventional outputs would show no additional benefit from the spike realization.

After teaching, remove live source inputs, ingest fresh documents once, and run
eight supported fresh requests plus two unsupported probes; then all sixteen
retention requests. Disclose retained raw documents/vectors as external memory.
Count first-attempt executed success separately from teacher repairs. A proposed
progress gate is at least 6/8 fresh and 12/16 retained edits, with both properties
correct for at least three fresh entities; report unsupported behavior separately.
Keep the original exam untuned. This remains a narrow software task, not general
exam comprehension or cell-only knowledge acquisition.

Budget: one fixed learning rate, at most three teaching passes (48 autonomous
attempts), 32 new certified contrasts, and at most 128 replay updates per arm;
two hours implementation/diagnosis and one declared cohort. Report encoder time,
retained bytes, native updates, simulated cell work, execution time and peak RSS.
Do not extend the budget after seeing transfer results.

If cyclic constraints exist but the current representation cannot fit them even
under a bounded conventional capacity diagnostic, choose candidate-centered
token interactions next. If it fits them but the online learner loses them,
optimize retention/credit assignment. If retention improves without fresh
transfer, the evidence favors memorization rather than reusable role learning.
These are separate hypotheses; none warrants neuron scaling or tie tuning.
