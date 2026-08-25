# Deep dive: SpikeGPT + NeuronSpark — what spiking networks demonstrably do at scale under gradients

**Date:** 2026-08-25
**Status:** primary-source deep dive (TMLR-published paper read in full; arXiv HTML paper read in full; model card + code tree inspected)
**Sources:**
- SpikeGPT: Zhu, Zhao, Li, Eshraghian, *SpikeGPT: Generative Pre-trained Language Model with Spiking Neural Networks*, TMLR 06/2024 ([arXiv:2302.13939v5](https://arxiv.org/abs/2302.13939), [code](https://github.com/ridgerchu/SpikeGPT), pretrained [216M checkpoint](https://huggingface.co/ridger/SpikeGPT-OpenWebText-216M)). Repo **archived** (last push 2025-07-21; 911 stars).
- NeuronSpark: Tang (Boston University), *NeuronSpark: A Spiking Neural Network Language Model with Selective State Space Dynamics*, [arXiv:2603.16148](https://arxiv.org/abs/2603.16148) (2026-03-17, unreviewed); weights [Brain2nd/NeuronSpark-0.9B](https://huggingface.co/Brain2nd/NeuronSpark-0.9B) + `-Chat` (Apache-2.0).
**Companion docs:** [similar-projects-landscape.md](similar-projects-landscape.md) (names both as closest *task* peers), [gemma4-comparison-design.md](gemma4-comparison-design.md) (M4–M6 ladder), [t-route-program-conclusion.md](t-route-program-conclusion.md) (path-1 fork), [deep-dive-spiking-reservoir-lm.md](deep-dive-spiking-reservoir-lm.md).

## Bottom line

Both papers establish the **ceiling of the gradient-trained spiking paradigm**, and that ceiling is
still visibly below dense models at matched size — while simultaneously proving the two design facts
our own falsification chain converged on from the other side:

1. **Going spiking costs ~7% relative quality at best case** (SpikeGPT's own matched-settings
   ablation: 1.283 vs 1.201 test BPC vs dense RWKV at 45M), and costs catastrophically (+0.202 BPC,
   ~17%) when binary activations are used *without* recurrent dynamics to compensate. The spiking
   fraction of every successful system **shrinks as scale grows**: SpikeGPT-216M dropped its binary
   input embedding; NeuronSpark transmits **floating-point leakage currents, not spikes, between all
   layers**, keeping 0/1 spikes only as each neuron's internal event process.
2. Every working spiking LM is a **continuous residual carrier + LayerNorm/RMSNorm + dense readout
   head, with spiking confined to well-isolated nonlinearities**. That is independent, stronger-resourced
   confirmation of our architectural instinct (continuous fitted-membrane state carrying computation;
   spikes as events, not the communication alphabet) — and an independent confirmation of
   [t-route-program-conclusion.md](t-route-program-conclusion.md)'s mechanism statement that
   token-*identity* bits cannot express position: Heaviside-binarized RWKV without recurrence collapses;
   identity-carrying static priors without dynamic state collapsed for us too.

For the [gemma4 ladder](gemma4-comparison-design.md): **nothing here supports M4–M6 becoming more
plausible for any spiking architecture, ours included** — neither paper contains a single
code-generation result, a benchmark comparison against a modern dense LM (NeuronSpark ran *zero*
benchmarks, by its own limitation list), or evidence that the spiking quality gap closes with scale
(SpikeGPT's MiniPile scaling probe shows ordinary diminishing returns, gap intact). These papers show
what spiking CAN do **when gradients are allowed**: end-to-end surrogate-gradient BPTT, 10× learning
rates on neuron parameters, gradient-norm surgery across layers. That is exactly path 1 of our fork.
If path 1 is ever chosen, NeuronSpark's training recipe is the template; our Allen-fitted physiology
becomes a far better *initialization* than their hand-designed one, and the latency-race decoder stays
ours alone. Under the current cell-only contract, the transferables are inference-side encoding and
initialization choices, listed in the adopt/adapt/avoid table below.

---

## 1. SpikeGPT (45M / 216M) — architecture, recipe, measured cost of spiking

### 1.1 Architecture (verified against `src/model.py`)

RWKV-lineage Transformer block, N layers of [Spiking-RWKV token mixer → LIF → residual;
Spiking-RFFN channel mixer → LIF → residual]:

| Component | Implementation |
| --- | --- |
| Token mixer | **Spiking RWKV**: token-shift → linear projections to R,K,V → LIF sparsification (`MultiStepLIFNode`, τ=2 i.e. β_decay=0.5, threshold=1, reset=0, ATan surrogate α=2.0) → spike-form K,V enter the **linear-time WKV recurrence** O(T); sigmoid receptance gate stays real-valued |
| Channel mixer | **Spiking RFFN**: token-shift → square-ReLU² sparse activation (optionally LIF — "graded spikes", performance-neutral per their Tab. 5) |
| Input coding | **Binary embedding**: embedding output passed through Heaviside (surrogate backward) at 46M; **dropped at 216M** ("remove the binary embedding but use the first layer neurons for encoding") |
| Residual stream | **Continuous** real-valued with LayerNorm ("Add & Norm"); spikes are added events, not the carrier |
| Output | Dense `LayerNorm → Linear(d×V)` generation head |

LIF hyperparameters (threshold 1, reset 0, decay 0.5) inherited from Spikformer/SEW-ResNet practice —
generic LIF constants, nothing human- or data-fitted.

### 1.2 Training recipe

| Variant | Data | Details |
| --- | --- | --- |
| 46M | Enwik8, char-level | L=12, d=512, dropout 0.03, Adam lr 6e-4, 12 h on 4×V100 |
| 216M | OpenWebText2, ~5B tokens (per HF card), GPT-NeoX 20B BPE | L=18, d=768, Adam lr 4e-4, warmup 500, 48 h on 4×V100; then WikiText fine-tune (lr ~3e-6 recommended) |
| Scaling probe | MiniPile 0.9B tokens | 216M → 3.20, 430M → 3.00, 1.2B → 2.88 train loss: scaling laws hold, diminishing returns |

### 1.3 Achieved quality vs dense baselines — the answer to "how much does spiking cost?"

Enwik8 test BPC at ~45M (their Tab. 2):

| Model | Test BPC |
| --- | --- |
| Transformer 43M | **1.137** |
| Reformer / Performer / Linear Transformer | 1.195 / 1.199 / 1.207 |
| **SpikeGPT 46M (T=1024)** | **1.283** |
| **SpikeGPT 46M (T=3072)** | **1.262** (train 0.903 → overfitting; longer context doesn't buy test quality) |
| Synthesizer | 1.298 · SHA-LSTM 1.330 · Stacked LSTM 1.670 |

The decisive number is their own matched-settings ablation (Tab. 5, all L=12/T=1024/d=512):

| Variant | Train / Test BPC | Reading |
| --- | --- | --- |
| Vanilla RWKV (dense) | 1.014 / **1.201** | dense reference |
| Heaviside RWKV (binarize, no LIF/recurrence change) | 1.318 / **1.403** | naive binarization is catastrophic (+0.202, ~17%) |
| SpikeGPT-B (LIF only inside FFN) | 1.109 / 1.306 | partial spiking recovers most of it |
| Full SpikeGPT | 1.113 / **1.283** | **spiking cost ≈ +0.082 BPC (~6.8% relative) when recurrent dynamics compensate** |

At 216M (Tab. 3): fine-tuned WikiText-103 test ppl **39.75** — the paper's own text concedes it
"has fallen behind the GPT-2 series models" on WT-103 (quoted GPT-2 references: 124M and 346M
classes) while on WikiText-2 it posts **18.01 test**, better than both quoted GPT-2 numbers. Their
proposed remedies for the large-corpus gap are **knowledge distillation** and refined training —
i.e., leaning further on dense teachers. NLU (with pretraining): SST-2 88.76 / SST-5 51.27 /
MR 85.63 / Subj 95.30 — approaches, does not reach, BERT.

Energy claim, handled carefully: 32.2× fewer operations overall (33.3× on the matmul-dominated
blocks) is a **theoretical op-count/energy table at measured firing rate R̂=0.15** assuming
asynchronous neuromorphic hardware (AC≈MAC/5 pricing). It is not a GPU wall-clock result; GPU SNN
timestep loops are typically *slower* than dense forward passes.

## 2. NeuronSpark (0.9B, random init) — PLIF × selective state space

Single-author BU preprint; largest from-random-init pure-SNN LM claimed. **Pure-SNN** means: no
attention anywhere; PLIF neurons replace activations *and* the sequence mixer. It is **not**
binary-communication: see §2.2.

### 2.1 Architecture

Config: D=896, L=20, N=8 (so D·N=7,168 hidden spiking neurons per block; 143,360 total hidden),
K≤16 PonderNet timesteps, D_ff=2688, 6,144-token custom BPE, **context 512**, Chinese.

Core recurrence (**SelectivePLIF**) with SNN–SSM duality:

```
V[t] = β(t)·V[t-1] + α(t)·I[t]        # β↔Mamba Ā, α↔B̄, V↔h
s[t] = Θ(V[t] − V_th(t))              # spike
V[t] −= V_th(t)·s[t]                  # SOFT reset (residual charge kept)
```

β(t), α(t), V_th(t) are produced per-step by learned modulation networks from the input (7 parallel
projection paths incl. gate + skip). Structured initialization is load-bearing: β biases logit-spaced
across N groups targeting β∈[0.80, 0.99] (multi-timescale), target firing rates 25%→8% across groups,
thresholds calibrated from stationary membrane variance, modulation weights started at 0.1× so
dynamics begin near their (static, stable) biases. FFN mirrors SwiGLU with the elementwise product of
two leakage signals replacing SiLU gating. **PonderNet** learns per-frame halt probabilities
(K=16 max) at every sublayer (2L=40 aggregation points), λ_ponder=0.01.

Stabilization stack (all four required — every ablation that removed one stagnated at loss >7.0 vs
3.5 for the full stack, and a "cortical lateral cross-token causal spike propagation" variant diverged):
residual centering (mean-subtract each sublayer output — DC drift across 20 layers);
lateral-inhibition normalization (divisive normalization, mathematically = RMSNorm); natural-gradient
compensation (gradient descent in β-space to fix saturation pathology + per-layer gradient norms
equalized to the geometric mean); Triton fused single-pass scan kernels, fp32 arithmetic/bf16 storage.

### 2.2 The central design fact: floating-point inter-layer signaling

> "A key modeling choice is to treat layer-to-layer signals as floating-point leakage-current signals,
> while retaining 0/1 spikes as the internal neuronal event process; this distinction avoids the
> expressivity bottleneck of purely binary inter-layer communication."

Default inter-layer symbol is `leak[t] = (1−β)·V_post[t]` — the charge about to dissipate, framed via
Hodgkin-Huxley membrane leak conductance. Rationale given: binary spikes limit gradient flow through
the surrogate's narrow support *and* cap expressivity. A skip projection guarantees gradient flow
"even when all hidden neurons are silent." This is the second independent convergence with our
carrier-plus-events reading of §1.

### 2.3 Scale viability verdict (what 0.9B-from-random-init actually demonstrates)

Training: Adam peak lr 2e-4 with **neuron parameters at 10× lr (2e-3)**, warmup 1000 + cosine,
effective batch 64, clip 1.0, bf16, gradient checkpointing; 85K steps ≈ 1.4B tokens (14% of the
Seq-Monkey 10B-token Chinese corpus) → **pretraining loss 3.6** (from 9.0); throughput ~960 tok/s on
8×RTX 4090. SFT: 6.5K steps ≈ 42K conversations (1.2% of BelleGroup 3.5M_CN) → loss 2.1.

What it proves: a pure-PLIF backbone can be optimized stably at ~1e9 parameters from scratch, and
produce fluent, POS-structured Chinese dialogue. What it does **not** prove: competitiveness. Its own
Limitations section: (1) 512-token context; (2) **no quantitative benchmarks (C-Eval, CMMLU) or
Transformer baselines**; (3) Chinese only; (4) repetition artifacts, **no reasoning capability**
(measured: arithmetic 0/8, commonsense 25% coincidental, logic 83% via keyword repetition, coherence
6/6 fluent); (5) interpretability analyses correlational only. Loss 3.6 on a 6k vocab after 1.4B
tokens is far outside dense-model territory for this size.

Human-plausibility claims, assessed: the paper offers **analogies, not fits** — leak current framed as
membrane conductance, divisive normalization cited to Carandini & Heeger, learned β splitting into
fast (67.3%) / slow (32.7%) populations likened to fast-spiking interneurons vs regular-spiking
pyramidal cells, depth-increasing adaptive timesteps (SNNBlock E[K] ≈ 4 → 12.7 across layers 2→19)
likened to cortical hierarchical integration windows. Emergent findings genuinely worth noting:
punctuation/function words receive systematically fewer timesteps than content words with no linguistic
supervision, and E[K] tracks syntactic role, **not surprisal** (r=−0.12 excluding BOS). But there is no
human electrophysiology, no cell-type identity, no fitted parameters — the plausibility register is
interpretive, several tiers below our pinned Allen GLIF provenance.

Provenance warnings: both code repositories cited in the paper footnote
(`github.com/Brain2nd/NeuronSpark`, `.../NeuronSpark-V1`) return 404 as of 2026-08-25 — only HF
weights exist, so architecture claims rest on the paper alone. Hardware attribution conflicts:
paper Table 2 says 8×RTX 4090; the HF model card says a single DGX Spark GB10 (128 GB, 4-GPU DDP).
Unreviewed, single author, March 2026 — weight every number accordingly.

## 3. Implications for the gemma4-comparison-design M4–M6 ladder

Recall the rungs: M4 = HumanEval+ ≥ 10%; M5 = MBPP+ ≥ 20% + simple-SWE unseen-family transfer;
M6 = SWE-bench Verified attempt + headline vs Gemma-4-12B-QAT.

1. **Zero code-generation evidence exists anywhere in the spiking-LM literature these two anchor.**
   SpikeGPT's hardest evaluation is WikiText perplexity and sentiment classification; NeuronSpark ran
   no benchmark at all. M4–M6 are not merely unscheduled for spiking systems — they are undemonstrated
   even in principle in this cluster.
2. **Beating gemma-4-12b-QAT is not plausible for ANY spiking architecture at demonstrated scales,
   with OR without dense-equivalent compute — yet.** The best-case published spiking tax is ~6.8%
   relative (45M, matched settings, everything tuned to compensate). That tax persists at 216M against
   GPT-2-class dense models, and NeuronSpark shows no closure at 874M because it cannot demonstrate
   closure (no baselines). Gemma-4-12B-QAT is a frontier-lineage 12B model; the observed gap
   (spiking-vs-dense-at-matched-size) compounds with the absolute scale gap (≤1.2B-ever-trained vs
   12B). The honest formulation for our docs: M6 presumes closing BOTH gaps; no public evidence
   supports either closure for spiking backbones, and our constraint set (no global gradients) is
   strictly harder than theirs.
3. **"Only with dense-equivalent compute" is also not sufficient** — fairness rule 3 already demands
   the axis be named. Two specific accounting traps these papers expose:
   - SpikeGPT's 32.2× figure is ops/energy on asynchronous neuromorphic hardware at R̂=0.15 firing
     rate — do **not** let it leak into our compute tables as a GPU-side efficiency credit. Our
     comparisons are wall-clock + parameters on this box.
   - Training-throughput asymmetry: NeuronSpark sustains ~960 tok/s on 8×4090 at 0.9B. Any
     "matched-compute" framing that counts parameters but not achieved token throughput flatters the
     spiking side enormously.
4. **What the ladder should take from this deep dive:** a calibrated prior for the cost of spiking
   itself. Use SpikeGPT's ablation as the citation-grade range: ~7% relative quality loss when the
   architecture is engineered to compensate, ~17% when binarization is naive. Our system's concession
   versus a hypothetical dense-equivalent should be expected to exceed this (local learning ⊃ harder
   than surrogate-gradient BPTT), which sharpens, rather than relaxes, the doc's own power analysis
   ("M4–M6 year-scale"). No edit to the milestone definitions is warranted; the M0–M3 rungs are
   untouched by anything here.
5. **Independent corroboration dividend:** both systems survived only by keeping the information
   carrier continuous and confining spikes to local nonlinearities — and Heaviside-RWKV quantifies
   what pure identity-bit channels cost (+0.202 BPC). This is external replication, from the
   best-funded possible angle (end-to-end gradients), of the mechanism our t-route chain localized:
   identity bits without dynamic state cannot carry positional/temporal structure. Cite it as such.

## 4. Integration verdict

These are gradient-trained systems — outside our cell-only contract in learning rule, connectivity
provenance, normalization plumbing, and dense readouts. Nothing here touches our *output* rule
(fitted-cell latency race remains unique; CTP-Hybrid's claim is still unverified low-confidence).
What transfers is narrower and inference-side:

**Transfers to our inference path even though training doesn't:**

- **Continuous-carrier/local-events principle (both).** We already satisfy it (GLIF membranes are
  continuous; spikes are events). The lesson is directional: do not drift toward spike-only channels
  between modules (e.g., inter-region or inter-space messaging must remain able to carry graded state).
- **Structured multi-timescale initialization (NeuronSpark).** β logit-spacing across [0.80, 0.99]
  with explicit per-group firing-rate targets (25%→8%) and stationary-variance-calibrated thresholds is
  a principled, published heterogeneity prior. ADAPT for unfitted dimensions, growth-time variation of
  our cells, and synthetic-cell construction — where we currently lack Allen fits.
- **Adaptive temporal integration (PonderNet).** Per-token simulated-biological-time budget before the
  race, instead of a fixed window. Their version needs a learned halt head; a local-learning-compatible
  substitute would derive halting from adaptive-threshold/homeostatic state. Watch-item, not scheduled.
- **Silent-path audit.** Their explicit skip path exists because blocks can go entirely silent. Audit
  our race/readout seams for analogous silent-collapse modes (a block whose population rate hits 0
  must degrade gracefully, not dead-path).
- **Kernel engineering parity (minor).** Fused single-pass scan + CPU Hillis-Steele prefix-scan
  fallback parallels our SeamA/SeamB + overlap/bitmap speed stack; nothing new to adopt, mild
  confirmation.

**If path 1 is ever chosen (relax local-learning-only), these become the template — specifically
NeuronSpark's recipe:** surrogate-gradient BPTT through PLIF dynamics; 10× LR on neuron parameters vs
backbone; natural-gradient compensation for threshold/decay biases (saturation pathology +
cross-layer equalization); residual centering; lateral-inhibition (=RMSNorm) normalization; structured
initialization; PonderNet adaptive K; fused scan kernels. SpikeGPT adds the token-mixer lesson: pair
binarized activations with a recurrent/linear attention state (WKV-style) that carries the temporal
structure the spikes can't.

**What we would KEEP from our fitted-cell physiology under path 1:**

- **Allen GLIF3/GLIF5 fitted parameters as initialization.** NeuronSpark's structured init is a
  hand-built statistical proxy for exactly the per-cell heterogeneity (membrane, reset, ASC,
  adaptive threshold) our pinned fits give exactly and human-data-grounded. Initialization is
  contract-compatible even under relaxed learning rules and is the strongest scientific differentiator
  we could bring to a gradient-trained spiking run.
- Soft reset (they independently arrived at subtract-reset-with-residual-charge; our GLIF reset
  semantics are the biologically fitted version of the same idea).
- E/I balance and cell-type heterogeneity ([human-ei-cell-types-and-balanced-dynamics.md]) — they have
  none; their emergent fast/slow β split suggests the optimizer *wants* what we would initialize.
- Dendritic branch structure as a connectivity prior (nothing in either paper touches dendrites;
  HTM/DendriCL remain the relevant neighbors there).
- **First-spike latency-race decoding** — neither group has anything like it; it survives path 1
  intact and remains the differentiator.

**What path 1 would change:** learning rule (three-factor/local → BPTT + surrogate), connectivity
(fixed/grown-local → trained), and a formal contract amendment. Precise wording matters: relaxing
*learning* need not relax *output semantics* — a gradient-trained fitted-cell system could still select
tokens by earliest output-cell spike. Even so, calibrate expectations with §3: path 1 buys
optimization feasibility (NeuronSpark proves stability at 1e9), not competitiveness (~7% best-case
spiking tax, plus a ~3-orders-of-magnitude training-throughput disadvantage vs dense at equal size).

## 5. Adopt / adapt / avoid

| Item | Source | Verdict | Detail |
| --- | --- | --- | --- |
| Continuous state carrier + spiking as localized events | both | **adopt** (already implicit; make explicit) | Design principle: never restrict inter-module messaging to binary spikes; both groups converged on this as necessary for function |
| Structured multi-timescale init: β logit-spaced [0.80,0.99], firing-rate targets 25→8%, variance-calibrated thresholds | NeuronSpark §3.4/A | **adapt** | Use for unfitted/synthetic cell parameters and growth-time diversity; cite as published prior |
| Adaptive per-token temporal integration before decoding | NeuronSpark §3.6 | **adapt** | Local variant: derive halt signal from adaptive-threshold/homeostat state, not a learned head |
| Silent-path skip audit | NeuronSpark §3.4 | **adopt** | Add tripwire: detect fully-silent blocks at readout seams; require graceful degradation |
| Matched-settings spiking-tax numbers (+0.082/+0.202 BPC) | SpikeGPT Tab. 5 | **adopt** (as calibration) | Citation-grade prior for "cost of going spiking"; use in any dense-vs-cell comparison prose |
| Structural (not surprisal-driven) compute allocation | NeuronSpark §5.5 | **adapt** (weak) | Supports role/window conditioning in teaching processes; correlational evidence only |
| Fused single-pass scan kernels; CPU prefix-scan fallback | NeuronSpark §4.3 | **note only** | Parallels our Seams + overlap/bitmap stack; nothing new |
| Pure binary inter-layer/unit communication | Heaviside-RWKV result; NeuronSpark §1 | **avoid** | Independently quantified as catastrophic without recurrent compensation; corroborates t-route identity-bit limitation |
| Neuromorphic energy/op-reduction figures on GPU runs | SpikeGPT Tab. 1 | **avoid** | Theoretical, R̂- and hardware-contingent; excluded from our fairness-rule-3 accounting |
| Benchmark-free fluency as success evidence | NeuronSpark §5.2 | **avoid** | Reinforces preregistered-gate discipline; their own Limitations admit it |
| Surrogate-gradient end-to-end training recipe (10× neuron LR, natural-gradient compensation, residual centering, RMSNorm-as-lateral-inhibition, PonderNet, fused scans) | NeuronSpark §4–E; SpikeGPT §3 | **avoid now; path-1 template** | Frozen until the user rules on fork path 1; then adopted wholesale over fitted-cell initialization |
| Distillation from dense teachers | SpikeGPT §4.2 discussion | **avoid** | Would violate no-ANN/pretrained-embedding constraints outright; note only as the field's own remedy for the spiking quality gap |

## 6. Verification log and residual caveats

- All numbers above read directly from: SpikeGPT TMLR camera-ready PDF (arXiv v5, fetched today) —
  Tables 1–5 and §4 text; SpikeGPT `src/model.py` @ master (archived repo); NeuronSpark arXiv HTML
  v1 (full text) and HF model card README (fetched today).
- SpikeGPT repo confirmed **archived** via GitHub API (`"archived": true`, last push 2025-07-21);
  landscape doc's 2026-06-02 archive date retained as-is (not contradicted; API doesn't expose the
  archive timestamp).
- NeuronSpark's two cited code URLs are dead (404) as of today; architecture description rests on the
  paper alone. Paper-vs-model-card hardware discrepancy (8×4090 vs DGX Spark GB10) recorded, unresolved.
- Column ordering in SpikeGPT's printed GPT-2 baseline rows (WT-103 vs WT-2 assignment) is ambiguous in
  linearized text; we rely on the paper's own verbal claim (behind GPT-2 on WT-103, ahead on WT-2)
  rather than re-asserting specific GPT-2 cell assignments.
- Web search/extract tooling was unavailable (gateway down); all sources fetched via direct HTTP and
  verified against primary documents. OpenReview API was challenge-walled; the TMLR-published arXiv v5
  was used instead, which supersedes it for content purposes.
