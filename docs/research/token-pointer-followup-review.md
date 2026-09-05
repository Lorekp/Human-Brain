# Token-pointer follow-up: separate answerability credit from span acquisition

Advisory review, 2026-09-05: no fit, encoder call, download or implementation. The failed verdict stands.

## What the completed comparison establishes

The [result](../../experiments/r4_integrated/token_pointer_result.md) and [report](../../benchmarks/token_pointer_acquisition_v1.json)
(`d4080dde…`) show supported TRAIN acquisition of cells **3→14→24/96** versus
BiLSTM **4→28→53/96** at epochs 1/4/8. Cells obtain **18/192** untaught answers
and **26/64** STOPs; their supported failures are **129 wrong emitted spans** and
**45 abstentions**. The trained frozen producer answers nothing and always stops.
All eight supported fresh integration cell attempts select the correct original
document, but seven extract the wrong span; oracle restriction also gives 1/8.
The 23 continuation offers improve taught answers, not fresh first-attempt count.
Thus source routing is not the immediate explanation. Neither a threshold change
nor calling every wrong answer a boundary error is supported by these receipts.

## Three concrete alternatives

1. **Change recurrent dynamics:** learn a per-cell input-dependent update gate,
   `g=sigmoid(Wg*x+Ug*h+bg); h'=(1-g)*h+g*tanh(W*x+U*h+b)`.
   Tallec and Ollivier derive this tied carry/update form from time transformations
   and test 64-unit variants on sequence tasks; this is not QA evidence.
   The current initialization has local Jacobian `.5I+.5*diag(tanh')*.9I`, whose
   diagonal entries are at most .95 before training. That motivates checking
   selective retention, but does not prove a trained-gradient or memory defect:
   contextual MiniLM inputs and bidirectional attention already supply shortcuts.
   Gating tests cell contribution but leaves STOP/span competition unresolved.
   [Paper, §§1–3](https://arxiv.org/pdf/1804.11188), [current recurrence](../../src/humanbrain/token_pointer_reader.py).

2. **Couple endpoint predictions:** a soft start-weighted state can feed an
   additional end-processing stage; the BiDAF authors' code does this with
   `a1i`, `g2` and `logits2`. It does not select a gold or hard start at inference.
   Alternatively, joint span scoring can discourage combining two answer
   boundaries. However, Fajcik et al. Table 3 finds most tested joint similarities
   no better for BiDAF; Table 2 reports BERT SQuAD2 compound EM 73.53 versus
   independent 73.89. These are credible candidates, not universal fixes or
   evidence that wrong-number/role selections are recoverable boundary errors.
   [Author code](https://github.com/allenai/bi-att-flow/blob/master/basic/model.py), [joint-objective paper, Tables 2–3](https://aclanthology.org/2021.mrqa-1.2.pdf).

3. **Separate learned answerability from conditional extraction — recommended.**
   Keep the entire cell/BiLSTM/bypass producer, generic inputs and span inventory;
   replace shared endpoint NULL competition with an explicit question-conditioned
   answerability loss and supported-only span loss. Read+Verify identifies this
   conflict and uses separate auxiliary losses. Its full system also uses human
   plausible spans for negatives and an entailment verifier: neither is included
   here. That paper ADDS auxiliary losses; replacing its joint loss is our adaptation.
   Earlier [hierarchical cells](../../experiments/r4_integrated/query_aligned_rate_cell_result.md) also failed; this tests today's partly learned producer. [Primary paper](https://ojs.aaai.org/index.php/AAAI/article/download/4619/4497).

## One bounded implementation and comparison

Use question-conditioned states `M`: `a=softmax(w_a^T M)` over valid words, `z=w_z^T sum_i(a_i*M_i)+b`.
For pooled inference/feedback, pool across ALL valid tokens of ALL supplied documents; the producer remains paragraph-local.
Initialize `w_a=0`, `b=0`, and `w_z` with the existing Xavier convention using a
separate fixed RNG stream; retain all old producer/endpoint initialization bytes.
Train the mean per-example loss `BCEWithLogits(z,answerable) + answerable*L_span`,
where `L_span` marginalizes the original accepted endpoint pairs with NO NULL in
either endpoint softmax. Unsupported items supply no fabricated span target.
On an all-STOP offer, endpoint parameters and their Adam moments must not move;
the shared producer still receives answerability gradients. This does not remove
all task interference. Decode STOP at `z<=0`; otherwise use the unchanged best
legal within-source endpoint sum and exact-tie STOP. No threshold or length sweep.

Run one fixed eight-epoch comparison on the SAME 8,192 TRAIN questions, batches,
Adam, clipping and dropout policy. Four new arms: cells, exact-initial frozen
producer with all new heads trained, trained bypass, and BiLSTM. Retain their
initial and post-training frozen copies. The completed four old-objective arms
are source-attested reference checkpoints, not retroactively refitted controls.
This isolates an objective/readout package; it does not isolate BCE from pooling.
Use the existing 128 TRAIN probes at epochs 1/4/8 and the SAME 256 exposed
development questions for paired diagnosis against old finals and new initial controls.
Do not assume another eligible article-disjoint roster exists; this is repeatable
development readiness, not fresh adjudication. New-cohort advancement stays later.

Preparation requires meaningful gradients, masked batching/STOP optimizer-state
QA, and the unchanged sixteen-question fit check; numerical correctness is not
broad readiness. Retain the previous acquisition margins and STOP floor on the
exposed roster, and require both full cell and conventional readers to fit at least
48/96 supported TRAIN probes. A STOP-only improvement or lower loss fails the
extraction hypothesis. If only BiLSTM becomes useful, this is evidence against
the current cell recipe; if neither does, do not automatically extend epochs.

The previous entire run took 3,637 s; its four update totals were 1,166/306/323/1,299 s.
Reuse the authenticated cache. Admit the new pooling head's measured cost before fitting;
it adds linear token work, not a model download or new raw-input cache. Propose
two hours preparation and the existing four-hour/4-GiB scored cap, one seed/launch.

Return all four policies and frozen mates to the SAME 28-item retained-source
Python session, 23 offers each, unchanged ambiguity exclusions. Keep broad labels
paragraph-local; only the reviewed pool authorizes cross-source feedback. Report first attempts,
taught recovery, later untaught actions, retention and teacher cost separately.
Readiness remains conditional: renewed span learning without later action benefit
does not meet the persistent-loop goal.

The missing seam is knowledge use after source removal; stored answer bytes are episodic payload, not proof of weight internalization.
A span pointer also cannot produce option selections, ordered tables or constructed
official-exam responses without an honestly declared structured producer. Keep that
pressure visible without rewriting official questions or crediting format omissions
as reading successes. A SQuAD-tuned RoBERTa teacher would add QA knowledge and needs
its own strong external-only control; it is not the selected cell-learning change.
