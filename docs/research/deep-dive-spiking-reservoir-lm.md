# Deep dive: `spiking-reservoir-lm` — negatives, readout benchmarks, adoptable methodology

**Date:** 2026-08-25
**Status:** primary-source deep dive (cloned repository + full `PAPER.md` read)
**Source:** https://github.com/alfanowski/spiking-reservoir-lm (Apache 2.0; Andrea Alfano, independent researcher; unreviewed)
**Companion landscape doc:** [similar-projects-landscape.md](similar-projects-landscape.md) (which names this the closest architectural implementation)

## Bottom line

`spiking-reservoir-lm` independently arrives, from the opposite direction, at the same
wall our R0 → R0-b → R0-c → race-pivot chain hit empirically: **a frozen/random
recurrent spiking substrate stores zero task knowledge; everything learnable lives in
a tiny trained readout, and that readout's parameter count bounds the system**. They
argue it from first principles (Allen-Zhu & Li ~2 bits/trained-parameter capacity law,
PAPER.md §7/§8.1) and never attempt reservoir learning; we falsified four
local-learning routes under preregistered gates. **No measurement of theirs
contradicts any of our falsifications** — they simply never enter the regime we
falsified, because backprop-through-the-readout is load-bearing for them and forbidden
for us. Their most useful transfers are cheap methodology (seeded multi-candidate
substrate selection, spike-rate health pre-flight, mechanical contract tripwires,
windowed-decoder probes) rather than any architectural component. Their headline
quality numbers are *not* comparable to ours (different corpus, metric, task, and
contract), and their own honesty section says why: **0 of ~320 generations fully
coherent**, best general-corpus cost 2.4560 bits/byte on Italian web text.

---

## 1. What they built (specification, verified against code)

Raw UTF-8 bytes (vocab 256, no tokenizer) → trainable `nn.Embedding(256, 64)` →
**frozen LIF reservoir, N=8,192** (`snntorch.Leaky`, β=0.9 fixed, θ=1.0 fixed,
subtract reset with `reset_delay=True`) → **windowed causal-attention readout**
(InProj+PosEmb → 2 layers, d_model=128, 4 heads, 256-byte window → 256-way logits).
Trainable parameters: **1,526,400** (embedding + readout only). The reservoir runs
under `no_grad`; surrogate gradient flows through it solely to reach the upstream
embedding; its weights are buffers, asserted so at every checkpoint (tripwire clone +
`torch.equal`) plus a `len(parameters())==0` assertion.

Recurrent construction (`models/spiking_reservoir.py`):

| Parameter | Value | Notes |
| --- | --- | --- |
| N | 8,192 | selected by wall-clock benchmark, not quality |
| W_in | i.i.d. Gaussian × 0.3 | scale 0.1 collapses spike rate to ~7×10⁻⁶ (near-silent bug they caught); healthy ≈2% population rate |
| W_res dense mode | spectral radius ρ=1.0 via eigendecomposition rescale | O(N³) construct, O(N²) store |
| W_res TT mode | rank R=8, d=4 cores, modes [8,8,8,16], core std s=(v/R_int)^(1/2d), v=ρ²/N | **dense matrix never materialized**; matvec by core sweep, bit-exact vs dense |
| Entanglement entropy Ē | measured 0.87–0.99 | outside Sato et al.'s [0.1, 0.5] "productive" band from rate-based reservoirs — band **does not transfer** to spiking substrate; used as logged diagnostic, not selection gate |
| Selection | 4 seeds × 300-step training probe, global seed pinned across candidates | spread 0.0099 bits/byte (weak signal, honestly reported) |
| Soft-spike feature | φ(mem)=Φ((mem−θ)/(0.1θ)) opt-in | graded membrane CDF replacing binary spike as readout feature only; recurrence untouched |

Efficiency results: TT compression 474.9× (N=2048) → 6241× (N=8192) and genuinely
*faster* forward (+~1.73×) and backward; incremental `step()` generation 86–106×
faster, bit-exact.

## 2. What they measured and falsified (complete negatives inventory)

| # | Negative / boundary result | Their diagnosis | Relation to our ledger |
| --- | --- | --- | --- |
| N1 | **Knowledge ceiling**: frozen reservoir contributes zero storable knowledge regardless of size; bounded by 1.5M trained params (§7.1, §8.1, Allen-Zhu & Li arXiv:2404.05405) | First-principles + every mechanism evaluated confirmed it | **Directly corroborates** our program-grade termination of Bible-QA for fixed-local-connectivity ([t-route-program-conclusion.md](t-route-program-conclusion.md)): local rules on a fixed random-ish substrate cannot store the task. Independent convergence, stronger theoretical grounding than ours. |
| N2 | CMA-ES evolutionary readout capped at ~1k params → **pivot to backprop readout**; 4.6757 → 2.9492 bits/byte step change | Evolutionary optimizers don't scale past ~1k params | Mirrors our ridge retirement (O(N²)/O(N³)) and three-factor substitution: readout *capacity*, not reservoir size, moved their number. Corroborates "tiny trained-param readouts hit hard walls." |
| N3 | `torch.compile`: 0.92–0.97× (no win) | snntorch graph break per timestep; workload BLAS-bound | Neutral to us; we don't use snntorch. |
| N4 | Associative/parallel-scan LIF reformulation: correct, bit-exact, **5–11× slower** | At ρ=1.0 all-to-all coupling with N·2.2% rates, probability some neuron fires per step ≈1 → every speculative segment degenerates to length 1 | **Corroborates** our event-proportional stepping choice and warns against any time-parallel-scan shortcut. Note: their fix-concession (future work: block-sparse modular connectivity) is the regime we already live in — sparse local recurrence makes segments longer. Keep this in mind if anyone proposes scan tricks here. |
| N5 | Self-consistency / majority-vote sampling: does not fix low-temperature repetition loops (slightly worse 2/3 samples) | Voting over draws of an already-peaked single-step distribution reinforces the mode; classical self-consistency votes across whole diverse completions | Cheap avoidance for any future decode-harness work on our free-running lane. |
| N6 | Stage D safety patch (reallocate weight toward new authority-framing examples): target prompt unmoved **and regression on previously-fixed content** (0/40 → 3–4/40) | Capacity competition among categories at ~1.5M trained params | Strong rhyme with our typed-composition result (TRAIN replay ≠ DEV composition; prompt-trace retention ≠ generalization) and with developmental-growth staged-learning failure: reallocating/growing small trained capacity does not buy generalization. |
| N7 | Stage D2 additive patch: target still tied (2/30 across C, D, D2); partial regression repair only; forgetting monitor worsened (2.5933 → 2.6866) | Adding capacity isn't free either | Same lesson as N6. Two independent, well-reasoned data interventions produced zero movement — they conclude *capacity limitation, not curation*. This is our race-pivot verdict pattern ("diverse-but-wrong": de-collapse achieved, correctness not). |
| N8 | Coherence ceiling: **0/~320 generations fully coherent** at any tested setting; best setting (temp 0.3, rep-penalty 1.3) yields ~50%→73.5% partial-coherence; one celebrated lucky sample failed systematic re-test and was retracted | Structural (per N1), not a bug | Rhymes with our free-running gate failure (17.51%, ties always-space unigram) and our benchmark-status discipline: teacher-forced/reference-matched success ≠ free-running capability. Their lucky-sample retraction is a worked example of the one-shot/systematic-sweep discipline we preregister. |
| N9 | H1 Reservoir-State Self-Retrieval: reservoir-state keys beat raw-last-32-bytes keys (5.585 vs 6.517 bpb; top-1 38.66% vs 21.00%) — **but no interpolation weight improved final bpb at 128k-slot scale** | Slot-bank density suspected; mechanism not refuted | Closest structural rhyme in the whole paper to our **U-STATE** verdict: "union state separates but does not decode" (G-U1 huge pass; G-U2 fail below-unigram). Both projects found *dynamical-state keys carry real signal above trivial controls* yet fail to move the end-task metric. Their result is slightly *less* dead than ours (their keys beat the trivial control decisively; our reservoir-state route fell back to input-assembly space), worth one line in any future memory-design review. |
| N10 | Entanglement-entropy "productive band" [0.1, 0.5] from rate-based TT reservoir literature: **does not transfer** to a spiking substrate (measured 0.87–0.99, healthy and functional) | Domain mismatch, confirmed by dedicated research | Methodological caution that generalizes to us: dynamical-regime heuristics measured on non-spiking or non-human substrates may not transfer to GLIF dynamics. Supports our measure-on-your-own-substrate practice (e.g., [ustate-capacity-study.md](ustate-capacity-study.md)). |
| N11 | Rejected-with-reason: equal-count MoE (stores slightly less/param than dense), BitNet-style ternary quantization below ~3B params, recursive/looped computation below ~100–360M params, TTT/fast-weights (<5% GPU utilization in small chunks), imprinting pretrained LLM weights into cores (incoherent weight portability), pretrain-then-freeze reservoir (pays the same wall later), HDC bundled hypervector (60–70 items @D=10k), predictive coding / active inference as BP replacements | Each checked against primary sources; one cited source found likely fabricated (§7.3) | No overlap with our falsifications (different mechanisms), but their §A verification discipline is the same standard as our preregistration ledger. Their DLIF/dendritic-bilinear note (verified real, ICLR 2026 poster) explicitly flags that dendritic nonlinear integration is **untested in the frozen-dendrite + trained-readout regime** — i.e., nobody has shown dendritic computation helps under anything resembling our contract. Relevant context for APICAL-AND's post-mortem: the field has no counterexample either way. |

**Verdict for question 1:** every checkable claim corroborates or is neutral to our
R0/R0-b/R0-c/race-pivot chain; none contradicts it. The two programs differ in where
they placed the learning: they put *all* learning into a backprop-trained readout and
kept the reservoir frozen; we required learning to live in cells/synapses, falsified
four versions, and terminated the family. Their N1 capacity argument is the
theoretical umbrella our empirical chain sits under, and their N9/H1 result is the
closest independent replication of our U-STATE "separates-but-doesn't-decode"
signature.

## 3. Their readout performance vs our numbers

Their ladder (bits/byte, −log₂ p(byte), fixed seeded held-out set, identical protocol
across rows — Italian FineWeb2-HQ unless noted):

| Configuration | Result | Notes |
| --- | --- | --- |
| CMA-ES linear readout (English TinyStories) | 4.6757 bpb | ≤ ~1k trainable params; superseded |
| Baseline Transformer control | 3.2856 bpb | undertrained (200/5000 steps); not treated as final |
| Backprop pivot, dense N=2048 (English) | 2.9492 bpb | first point beating their Transformer control |
| Main run 3,000 steps, TT@8192 | 2.9835 bpb | diagnosed ~7× undertrained |
| **Main run 24,000 steps (corrected)** | **2.5286 bpb** | cosine LR, wd 0.07, ~60-bytes/param budget arithmetic |
| **Stage A soft-spike retrain** | **2.4560 bpb** | best general-corpus number; stochastic resonance confirmed (inverted-U, σ=0.1θ; frozen probe 2.5633 vs 2.6095 off-distribution) |
| Stage B fine-tune (own narrow split) | 1.8396 bpb | narrower corpus; forgetting monitor 2.5316 |
| Reservoir-state retrieval keys (H1) | 38.66% top-1 next-byte | vs 21.00% raw-byte-window control; did not improve final bpb |
| Generation coherence (best decode setting) | ~50%→73.5% partial-or-better; **0 fully coherent** | 180-generation sweep + 57-generation follow-up; temp 0.3, rep-penalty 1.3, best-of-5 helps modestly |

Our comparable numbers (from README evidence snapshot + falsification artifacts):

| Lane | Result | Metric / task |
| --- | --- | --- |
| Dendritic byte prediction | 3,889/4,500 = **86.42%** | held-out bytes, frozen code-language v3; matched Witten-Bell reference 86.40% (i.e., parity, not superiority) |
| Free-running continuation | **17.51%** / 1,536 bytes | ties always-space unigram; gate FAILED |
| Bible-QA span-acc floor | unigram **4.54%** (29/639), trigram 16.28%, majority-prior 4.07% | frozen comparators for all QA verdicts |
| Best QA span-acc across falsified arms | **1.10–2.97%** (U-STATE D5 MLP 1.10–2.19%; APICAL-AND best 2.97%; all below unigram) | learned decoders sit inside shuffled-label bands in several arms |
| Recurrent scaling cohort | teacher-forced 43.10→49.61%; free-running 6.25→12.76% | 24→384 active cells |

Comparability caveats (important, stated plainly):

1. **Different metrics**: theirs is teacher-forced bits/byte on natural-language web
   text; our 86.42% is next-*token* accuracy on a synthetic structured code language,
   and our QA numbers are span-exact-match. Bits/byte and top-1 accuracy are related
   but not convertible without the full predictive distribution, which they do not
   publish for the main runs.
2. **Different corpora/tasks**: FineWeb2-HQ Italian web prose vs KJV-Bible QA triples
   and frozen code languages.
3. **Different contracts**: their number is produced *by* a backprop-trained
   attention head choosing the byte (softmax argmax); our contract requires the
   output choice to be made by simulated cell state (first-spike race). Their readout
   performance is therefore an **upper-bound probe of what information exists in a
   spiking reservoir's recent spike history**, not evidence about what our contract
   can achieve.
4. Task-context note: the brief's "1.19% QA span-acc" does not appear verbatim in the
   repo artifacts; the closest recorded values are U-STATE D5 best-MLP 1.10%/1.41%/
   1.88%/2.19% (below-unigram, reservoir-state route dead) and APICAL-AND's 2.97%
   program-wide best. If 1.19% refers to a specific arm, cite the exact
   `experiments/*/probes_results.json` row; the honest summary band is
   **all learned QA decoders < 4.54% unigram floor**.

What the comparison actually establishes: a generic-LIF reservoir whose *only*
advantage is richness of random recurrent dynamics, read out by 1.5M backprop-trained
parameters, lands at ~5.5× perplexity per byte on natural text and never produces one
fully coherent generation — while our matched-reference parity result (86.42% vs
Witten-Bell 86.40%) shows our dendritic lane can *match* classical compression on
structured input but our free-running and QA lanes sit at/below trivial baselines.
Both programs, independently, demonstrate that **spiking-substrate sequence models at
small trained-parameter scale do not yet produce generative language competence**, and
that the binding constraint is trained capacity + learning rule, not neuron count.

## 4. Integration verdict — adopt / adapt / avoid

Legend: **Adopt** = lift directly into our workflow within the cell-only contract;
**Adapt** = transferable with rework; **Avoid** = incompatible with contract or
already falsified.

| # | Finding / technique | Verdict | Action for us |
| --- | --- | --- | --- |
| A1 | Seeded multi-candidate substrate selection (fixed global seed for trainable parts; independent generator for substrate; short 300-step probe decides) | **Adopt** | Apply verbatim to GLIF5 network instantiation for the A013 ladder and any Bible-QA rerun: build k candidate networks differing only in construction seed, run the cheapest existing downstream probe, pick the winner, record the spread. Replaces arbitrary seed choice; costs minutes. |
| A2 | Spike-rate health pre-flight (steady-state membrane std ≈ scale/√(1−β²); abort if population rate < ~0.1% or saturated) | **Adopt** | Their near-silent-reservoir bug (rate 7×10⁻⁶ at W_in scale 0.1, caught only mid-project) is exactly the failure our ≥1% activity EP-runtime gate guards at runtime. Add the cheap analytic pre-flight before long cohorts; pairs with our fixture-first discipline (APICAL-AND disclosure #1: zero-spike fixture caught pre-spend). |
| A3 | Mechanical contract tripwires: `assert len(list(reservoir.parameters()))==0`; clone-and-`torch.equal` weight snapshot asserted at every checkpoint | **Adopt** | We enforce the cell-only contract socially/preregistration-wise; add mechanical tripwires to the experiment harnesses so a forbidden-component merge or dtype drift fails loudly at checkpoint time. |
| A4 | Windowed decoder probe (readout attends over a bounded window of recent states instead of a single timestep vector) as an *offline diagnostic* | **Adapt** | Our G-U2/G-A3-style probes are single-timestep logistic/MLP decoders. Their architecture demonstrates windowed attention extracts substantially more from spike history. Upgrade one probe tier: windowed (logistic/MLP first) decoders over logged trace banks, to separate "state lacks the information" from "single-shot decoder lacks capacity." Contract-safe: offline analysis, not semantic decoding. Directly sharpens the localization of our next negative (or positive). |
| A5 | Graded membrane "soft-spike" readout feature φ(mem)=Φ((mem−θ)/σ), σ=0.1θ | **Adapt (diagnostics only)** | Reading graded membrane potential is reading genuine cell state, so it is admissible in *diagnostic decoders* — but it is not obviously compatible with first-spike-race semantic decoding, and our R0-b already found voltage-quantile state features capacity-regressive for QA. Worth including as one arm in an A4-style windowed-probe sweep; do not plan an architecture around it. |
| A6 | Byte-reformulated compute-optimal budget (~60 bytes exposure per trained parameter; diagnose undertraining ×7 before changing anything else) | **Adopt** | Use for planning full-KJV epoch value and any future teaching-process budgets: state the exposure-per-fitted-parameter arithmetic *before* a run so "undertrained" vs "falsified" is separable. Their grokking citation (Murty et al. ACL 2023) also justifies longer-than-plateau schedules with wd tuning if we ever tune plasticity constants. |
| A7 | Specialization-with-forgetting-monitor pattern (general-corpus replay slice mixed in; held-out general monitor tracked at every checkpoint) | **Adopt** | Matches our A013 teaching-process design needs: any future curriculum stage must carry a general-domain monitor so catastrophic forgetting is visible mid-run. Cheap, preregistrable. |
| A8 | Repetition-penalty decoding and best-of-N self-scored re-ranking | **Adapt (analysis harness only)** | Useful diagnostics for the free-running lane's failure modes, but decode-time logit surgery is plumbing that must never touch the semantic race in a scored run. Record as analysis-only tools. |
| A9 | Preserve failed mechanisms as tested, documented code (`forward_parallel` retained with root-cause comment) | **Adopt** | We already do this at the ledger level; their in-code root-cause comment block is a good pattern for native-kernel experimental paths. |
| A10 | Tensor-train/MPO-native recurrent construction | **Avoid** | Architecturally alien: TT approximates an all-to-all dense Gaussian operator, which is precisely the connectivity our sparse-local, Allen-fitted, biologically-motivated contract forbids. Its benefit — compressing O(N²) stored scalars — solves a problem our sparsity already solves (O(edges)). Their own N4 result further indicts the all-to-all regime for time-parallel execution. |
| A11 | Backprop-trained embedding/readout in the inference path; softmax-argmax byte choice | **Avoid (in scored runs)** | Forbidden by the cell-only contract for semantic choices. Legitimate only as offline upper-bound probes (see A4). |
| A12 | All-to-all dense coupling at spectral radius 1.0 | **Avoid** | Non-biological and their own worst performer operationally (scan-hostile, BLAS-bound). Our sparse local recurrence is the right call; their future-work item #8 (block-sparse modularization) concedes this. |
| A13 | Self-consistency / majority-vote byte decoding | **Avoid** | Measured negative (N5); do not spend cohort time on it. |
| A14 | Their RAG / VSA-slots / Hopfield-scratchpad knowledge-injection agenda | **Avoid for now** | Out of cell-only scope unless realized as cell-addressable memory; their own H2/H3/H4 implementations are untested. Revisit only if a memory direction survives user review post-T-ROUTE termination. |

Code licensing: Apache 2.0 — A1/A2/A3-style helpers may be lifted with attribution if
ever needed; realistically we re-implement (their stack is PyTorch/snntorch, ours is
Rust/CUDA + Python orchestration, and the ideas above are ~tens of lines each).

## 5. Discrepancies and caveats

- Source is a personal repository with a long technical report; **not peer reviewed**
  (same caveat our landscape doc records).
- Checkpoints are gitignored/not distributed; all numbers are self-reported, though
  the compression benchmark is exactly reproducible from the committed script and
  their internal consistency (bit-exactness tests, 152-test suite) is strong.
- The brief's "1.19% QA span-acc" was not found verbatim in our artifacts; nearest
  recorded values are listed in §3 item 4. Everything else quoted from our side
  (86.42%, 4.54%, 17.51%, verdicts) is verbatim from README/INDEX/verdict docs.
- Their corpus is Italian; every quality number above inherits that choice.
- Clone used for this deep dive lives at `%LOCALAPPDATA%\Temp\srlm-clone-20260825`
  (outside the repo; delete freely). Nothing from it was executed beyond reading;
  no code was imported into Human Brain.

## 6. One-paragraph takeaway for the roadmap

Treat `spiking-reservoir-lm` as an independent confirmation that the
reservoir-computing family cannot be rescued by readout sophistication alone: with a
modern attention readout, 8k spiking neurons, and every efficiency trick in place,
they still get ~2.46 bits/byte, zero fully coherent generations, and a
capacity-bound safety guard that survived two remediation attempts — while explicitly
declining to train the reservoir, which is the one thing our falsification chain
showed does not help under local rules either. Take their cheap disciplines
(A1, A2, A3, A6, A7), steal their windowed-probe idea to sharpen our next
localization (A4), refuse their architecture (A10–A12), and cite their §7/§8.1
capacity argument alongside our own ledger whenever the "why not just train it /
make the reservoir bigger" question recurs.
