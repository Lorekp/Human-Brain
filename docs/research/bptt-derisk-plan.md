# Path-1 BPTT de-risk plan — prove what we can at $0 before any rental

**Date:** 2026-08-25
**Status:** planning input for the path-1 fork item declared in
[path-decision-synthesis.md](path-decision-synthesis.md) §Path 1 / §4 step 6.
Answers the user directive: prove at least partially before spending; find ways
to avoid spending entirely. No runs executed in this document — every number is
either cited, read directly off primary sources today, or labeled ESTIMATE.
**Inputs:** [DD-SGS] [deep-dive-spikegpt-neuronspark.md](deep-dive-spikegpt-neuronspark.md),
[gemma4-comparison-design.md](gemma4-comparison-design.md),
[path-decision-synthesis.md](path-decision-synthesis.md), plus fresh primary-source
checks listed in §6 (verification log).

## Bottom line

1. **The planned spend ($60–90 vast.ai) is still the last step, but it is now
   preceded by three fully-$0 stages that each can kill or reshape it.** Two of
   them run on the local box, one is pure downloading. None requires the
   contract amendment; only the final training shot does.
2. **Free evaluation answers more than expected:** SpikeGPT ships two released
   216M-class checkpoints (OpenWebText-216M, BookCorpus; both L18-D768,
   ctx 1024 — no 46M/enwik8 checkpoint exists), and a third party
   demonstrated full-parameter fine-tuning of the 216M base **yesterday**
   (2026-08-24). NeuronSpark's complete architecture + surrogate-gradient
   training code ships inside its HF repos — the dead-GitHub caveat in
   [DD-SGS §6] is now resolved by the model cards themselves.
3. **No third-party matched-settings replication of the spiking tax exists.**
   A 100-citer sample of the SpikeGPT citation graph plus arXiv full-text
   checks found follow-up architectures (SpikeLM, WTA Spiking Transformer,
   SpikeDecoder, BiSpikCLM), distillation drops consistent with a real tax
   (SpikeBERT 59.7% GLUE; WTA "~20% below competitive LMs"), and zero
   independent re-measurements of the +0.082 BPC band. **Our gate question is
   publicly unanswered → we must measure our own band, and we can start doing
   that locally for $0.**
4. **Local partial proof is feasible on the RTX 4070S 12GB — both arms of it**
   (public-code reduced-scale tax measurement AND a tiny fitted-cell BPTT
   stability run), but **not** the faithful 45M run: the published recipe is
   ~10.2B token-passes ≈ 48 V100-hours [DD-SGS §1.2] ≈ 150–250 h on a
   4070S-class card (ESTIMATE). What fits in <12h is a ~5–10M-parameter pair
   at reduced context — enough for stability, rate health, our own small-scale
   tax delta, and wall-clock calibration, not enough to certify the 45M band.
5. **Free cloud tiers cannot cover the preregistered run either** (~150–250 T4-
   hours needed vs ~30 GPU-h/week Kaggle-class quotas, quotas UNVERIFIED
   today), and SpikeGPT's trainer saves weights-only checkpoints (verified:
   `src/trainer.py` L208 `torch.save(raw_model.state_dict(), …)`) — so
   cross-session resume would need a small patch and still perturb Adam state.
   Free tiers CAN cover all evaluation work and, with weeks of patience, one
   reduced-scale arm.
6. **Minimal honest spend if every $0 stage passes: ~$20–30 (single-seed 45M
   shot on vast.ai), escalating to the full $60–90 only if that seed lands
   in-band.** If the local stability arm fails, total program cost stays $0 and
   path 1 dies pre-purchase — which is exactly what the de-risk is for.

---

## 1. What each user question resolves to

| # | Question | Answer (detail in section) |
| --- | --- | --- |
| 1a | Tiny-scale BPTT locally — what fits <12h, does N≈5M give signal? | Yes, feasible: ~5–10M params, T≤512, ~100–400M token-passes <12h (ESTIMATE). It tests **stability/trainability of Allen-init under surrogate BPTT**, not the 45M band magnitude. §3.2 |
| 1b | Does public SpikeGPT code run unmodified on 12GB at reduced scale? | Yes with knob changes only (no code edits): smaller `n_layer/n_embd/ctx_len/batch_size`, `T_MAX` recompile, `B_GROUP_FORWARD/BACKWARD` micro-batching — all documented inline in `train.py`/`model.py`. Lets us measure **our own** tax delta + tok/s locally. §3.1 |
| 2a | Released checkpoints to evaluate/fine-tune free? | SpikeGPT: two 216M-class checkpoints, freely downloadable (§2.1). NeuronSpark: 0.9B/Chat/V3-1.1B with bundled code (§2.2). Evaluation is free; fine-tuning the 216M was just demonstrated by a third party. Code-task value is near-nil (see §2.2 caveat). |
| 2b | Third-party replications of the spiking tax? | None matched-settings. Direction-consistent secondary signals only. §2.3 |
| 2c | Colab/Kaggle free tiers cover a 45M run? | No for the faithful run (token-budget arithmetic + weights-only checkpoints + fluctuating quotas); yes for evaluations and possibly one reduced-scale arm. §4 |

## 2. Free-evaluation paths ($0)

### 2.1 SpikeGPT checkpoints — downloadable today

| Asset | Facts (read off HF API/cards today) |
| --- | --- |
| `ridger/SpikeGPT-OpenWebText-216M` | `SpikeGPT-216M.pth`, 861,716,013 bytes, Apache-2.0. Card: L18-D768, ctx 1024, OpenWebText2, GPT-NeoX 20B BPE. Paper claims WT-103 test ppl 39.75 [DD-SGS §1.3]. |
| `ridger/SpikeGPT-BookCorpus` | `BookCorpus-SpikeGPT.pth`, 737,553,365 bytes, BSD-2-Clause. Card: L18-D768, ctx 1024. Not referenced in the paper's eval tables — an uncharacterized public artifact. |
| Third-party finetune | `TaewoooPark/SpikeGPT-216M-Mixture-Finetune` created 2026-08-24: full-parameter SFT (oasst1 + smoltalk) of the 216M base, with SHA-256-pinned lineage and a fixed-split eval (PPL 44.48 base → 24.11 tuned). Proves the checkpoint loads and trains end-to-end outside the original lab. |

What free evaluation buys us:

- **F-EVAL-1 (checkpoint integrity + harness):** load both checkpoints with the
  archived repo's inference path, measure perplexity on a held-out slice of the
  matching corpus family. Gate: 216M WT-family ppl lands in the paper's
  neighborhood (39.75 ± our slice-composition tolerance); BookCorpus model gets
  characterized for the first time. Cost: ~1.6GB download + hours of CPU/GPU.
- **F-EVAL-2 (quality anchor refresh):** an independently measured number for
  "engineered best case" replaces trusting Tab. 3 — the anchor our +0.08–0.10
  band is defined against.
- **No 46M/enwik8 checkpoint was ever released**, so the exact configuration of
  our planned 45M replication cannot be evaluated for free — only re-trained.

### 2.2 NeuronSpark — code + weights live on HF (caveat resolution + new flag)

- `Brain2nd/NeuronSpark-0.9B` (+ `-Chat`) bundles **complete runnable code**:
  `modeling_neuronspark.py`, `configuration_neuronspark.py`,
  `atomic_ops/{plif_node, selective_plif, parallel_scan, lateral_inhibition,
  rms_norm, snn_block, snn_decoder_layer, snn_ffn, fp16_codec}.py`, tokenizer,
  fp32 safetensors; Apache-2.0; `trust_remote_code` interface
  (`transformers_version 4.52.0`). The two dead GitHub URLs in [DD-SGS §6] no
  longer block inspection: **the architecture claims can now be checked against
  shipped code, and the stabilization recipe read as implementation, not prose**
  (surrogate = Sigmoid α=4, soft reset, β=sigmoid(w) PLIF, external
  β(t)/α(t)/V_th(t) modulation, fused parallel scan).
- A V3 family also exists: `Brain2nd/NeuronSpark-V3-1.1B-Pretrain/-SFT`
  (en+zh, custom architecture, May 2026) — the line is active.
- **New provenance flag:** embedded `config.json` metadata reads
  `_training_step: 85000, _tokens_seen: 203,910,528` (~204M tokens) while the
  paper claims 85K steps ≈ **1.4B** tokens. Implied ~2,400 tok/step vs the
  ~32,768 expected at batch 64 × ctx 512. Likely gradient-accumulation
  accounting or stale metadata — unresolved; weight the paper's throughput and
  data-scale claims accordingly until clarified.
- **Code-task measurement (our gemma4 M-gates) is near-meaningless here:** the
  tokenizer is a 6,144-vocab Chinese BPE, the paper itself measured arithmetic
  0/8 and logic-by-keyword-repetition [DD-SGS §2.3]. A free inference sanity
  run (short Chinese perplexity + generation) is worth an hour; benchmark
  batteries are not. The real prize is the **training-side code template** for
  our own GLIF-BPTT port (§3.2).

### 2.3 Third-party replications — the gate question is open

Sampled the 100 most recent citers of arXiv:2302.13939 (Semantic Scholar) plus
arXiv full-text checks of the closest titles:

| Work | What it actually provides |
| --- | --- |
| SpikeLM (arXiv:2406.03287) | Capability-level comparison table vs SpikeGPT ("basic language modeling" only); no re-measured BPC. |
| WTA Spiking Transformer (arXiv:2604.11321) | States softmax-based spiking transformers land ~20% below competitive LMs on GLUE-style tasks — direction-consistent, different architecture. |
| SpikeBERT (via SpikeLM text) | Distilled GLUE 59.7% — a large drop, again direction-consistent. |
| SpikeDecoder (arXiv:2606.12287) | New GPT-architecture SNN built on SpikingJelly MultiStepLIF (same neuron library as SpikeGPT); no SpikeGPT-band replication. |
| BiSpikCLM (arXiv:2605.13859), SpikingMiniLM, Sorbet (2409.15298), SpikeLLM, MatMul-free-LM cluster | Adjacent cost-of-sparsity evidence; none answers "what is the matched-settings tax at 45M". |

**Conclusion:** the +0.08–0.10 band rests on exactly one source (SpikeGPT Tab. 5)
and has never been independently reproduced. This upgrades the importance of
measuring our own band (even at reduced scale) rather than treating the
published one as ground truth — it is simultaneously (a) harness validation and
(b) an actual replication attempt at whatever scale we can afford.

## 3. Partial local proof on the RTX 4070S 12GB ($0)

### 3.0 Step-zero prerequisites (minutes, do first)

Hardware probing was deliberately not executed during this planning pass;
before relying on any throughput estimate below, run once locally:
`nvidia-smi` (confirm 12GB card + driver), install `torch`+`ninja` (the repo
compiles its WKV CUDA kernel at import), and time one training step at the
reduced config. Every hour-estimate in §3 is tagged ESTIMATE until that
calibration exists. VRAM levers are all first-class knobs in the shipped code:
lower `ctx_len` (recompile with smaller `T_MAX`; the source warns T_MAX "TAKES
LOTS OF VRAM"), lower `n_layer`/`n_embd` (shipped defaults are L24/D768 — set
them explicitly), shrink `batch_size` (default 12), and split batches via
`B_GROUP_FORWARD/B_GROUP_BACKWARD`.

### 3.1 Arm L1 — measure OUR OWN tax band at reduced scale (public code)

Run the archived SpikeGPT repo twice, identical settings, differing only in the
spiking substitution: dense-RWKV control vs full SpikeGPT (this mirrors their
Tab. 5 ablation design, which is precisely the comparison our gate is defined
on).

- Config class that fits comfortably in 12GB with headroom (ESTIMATE):
  L6/D384/T512/batch 8 ≈ ~5–8M params (RWKV-lineage blocks ≈ ~7·d² per layer;
  enwik8 char vocab 205 makes embeddings negligible) on a 30–80M-char enwik8
  slice ≈ 100–400M token-passes in <12h.
- Deliverables: `Δ_small` = test BPC(spiking) − test BPC(dense) measured by us;
  wall-clock tok/s on this card = the calibration factor that turns the rental
  estimate from folklore into arithmetic; smoke-tested data/checkpoint plumbing
  reusable for Arm L2.
- Interpretation guard: Δ_small ≠ the 45M band (tax is scale-dependent and only
  ever measured at 45M). Pass criterion is directional reproduction
  (small positive delta, dense control strictly better) + stable training; a
  sign flip or divergence is itself decision-relevant (harness bug or
  scale-fragile recipe — either blocks the rental).

### 3.2 Arm L2 — tiny fitted-cell BPTT stability probe (N≈5M, the real unknown)

The single question whose answer currently justifies or cancels the entire
$20–90 tail: **do Allen GLIF3/GLIF5 fitted parameters survive surrogate-gradient
BPTT?** NeuronSpark proved stability for hand-built init at 0.9B; nobody has
proved it for physiology-fitted init at any scale.

- Design: single-step PLIF-style loop with the neuron swapped for
  TwoCompartmentGLIF5/GLIF3 parameters as frozen initialization (10× neuron LR
  per the NS recipe); continuous carrier throughout (both external systems'
  non-negotiable); dense softmax head first — the latency-race decoder is a
  separate later probe, not part of the stability gate.
- Budget: same envelope as L1 (100–300M token-passes overnight-class).
- Hyperparameters mirror 45M-equivalent ratios (lr 6e-4 scaled by width,
  warmup + cosine, clip 1.0).
- Gates (frozen before running, per §2.1 disciplines of the synthesis doc):
  1. loss descends below the unigram floor for the corpus slice;
  2. population rate stays inside the pre-flight analytic band (<0.1% or
     saturation aborts);
  3. per-layer gradient norms stay within ~an order of magnitude of each other
     (NS natural-gradient equalization check) without exploding through the
     adaptive-threshold/ASC path;
  4. silent-path audit clean (no fully-dead blocks).
- Falsifier: divergence, saturation, or gradient pathology through the fitted
  dynamics even after applying the four-part stabilization stack → **fitted-cell
  initialization is not trainable under BPTT at any tested scale; the rental is
  cancelled and path 1 ends pre-spend.**

### 3.3 What local proof cannot establish

- The band **magnitude** at 45M (only its existence/direction at small scale).
- Seed distributions at production scale (their trimodal-transformer warning
  [DD-DCL] applies to us too).
- Race-decoder interaction with gradients at scale (deferred probe above).
- Anything about the 216M regime.

## 4. Free compute tiers vs the 45M run

Arithmetic first: the faithful 46M recipe is ~10.2B token-passes
(1000 mini-epochs × 10,000 seqs × ctx 1024, read off `train.py`) ≈ 48 V100-h
[DD-SGS §1.2] ≈ 50–60 single-4090-h ≈ **150–250 T4-class hours (ESTIMATE)**.

| Tier | Reality check | Verdict for 45M ×1 seed |
| --- | --- | --- |
| Colab free | FAQ verified qualitatively today: free GPU access exists, "not guaranteed and not unlimited", limits fluctuate, no published hard numbers. | No — quota + 12h-session churn over ~200 T4-h is fantasy arithmetic. |
| Colab Pro/Pro+ (~$10/~$50/mo, compute units) | Paid CU balance relaxes limits; A100 occasionally granted. | Marginal alternative to vast.ai at similar monthly cost with worse determinism; not cheaper once realistic. |
| Kaggle free | Commonly documented ~30 GPU-h/wk, T4×2/P100 — **could not be verified today (docs page JS-only); treat as UNVERIFIED**. Sessions cap well under the run length; weights-only checkpoints mean naive resume restarts Adam moments. | Only with a resume patch (save optimizer + scheduler state; trainer.py currently saves `state_dict()` alone — verified) and ~6–8 weeks of quota. Schedule risk high; $0 though. Legitimate fallback if the user rules out spending entirely. |
| Lightning AI / others | Free monthly credits exist but are single-digit GPU-hours class. | Evaluations only. |

**Bottom line:** free tiers cover §2 evaluations and (slowly) one §3 arm; they do
not cover the preregistered run without weeks of patched-resume scheduling that
would itself violate the cohort-spent-once discipline if a session dies late.

## 5. Decision tree

```
STAGE 0 ($0, ~1 day) — environment + integrity
  verify GPU/torch/ninja locally; download both SpikeGPT ckpts; F-EVAL-1 gates pass?
  ├─ FAIL (ckpt won't load / ppl wildly off) → stop; investigate before anything else
  └─ PASS → STAGE 1

STAGE 1 ($0, ~1 week) — Arm L1: our own reduced-scale tax band
  dense RWKV vs SpikeGPT at L6/D384/T512, enwik8 slice, <12h/arm
  ├─ no positive delta / diverges → HARNESS-or-recipe problem: fix locally;
  │    rental BLOCKED (we cannot trust any band we'd measure remotely)
  └─ small positive delta reproduced → STAGE 2 (parallel with continued L1 seeds)

STAGE 2 ($0, ~1 week) — Arm L2: fitted-cell BPTT stability probe (N≈5M)
  Allen-init GLIF under surrogate BPTT, 4 frozen gates (§3.2)
  ├─ ANY gate fails persistently → PATH 1 ENDS. Total spent: $0.
  │    (Falsifier fires: fitted physiology doesn't train under gradients)
  └─ all gates pass → STAGE 3 decision point (user ruling required anyway:
       contract amendment per synthesis §4 step 6)

STAGE 3 — the only paid fork
  a) $0 route: Kaggle single-seed 45M WITH resume patch (~6–8 wk calendar,
     UNVERIFIED quota, session-death risk) → only if spending is ruled out
  b) MINIMAL SPEND: vast.ai single 4090-class seed, ~$20–30
       ├─ lands inside +0.08–0.10 band vs dense control → option to buy seeds
       │  2–3 (~$40–60) for the preregistered k≥3 claim → total ≤ $90 ceiling
       └─ lands > +0.15 (Heaviside territory) → path-1 terminal falsifier fires
          [synthesis §Path 1]; spent: one seed only
  c) never spend the full $60–90 up front — the single seed IS the gate
```

Residual unknowns even after Stages 0–2 pass and a single seed lands in-band:
band magnitude confirmation at 45M across seeds; race-decoder variant at scale.
These are exactly what seeds 2–3 (and the deferred decoder probe) buy.

## 6. Verification log

- All web search/extract tooling down again this session (Nous gateway
  unavailable, as in [DD-SGS §6]); every fetch below was direct HTTP.
- Read today from `raw.githubusercontent.com/ridgerchu/SpikeGPT@master`:
  `train.py` (defaults: ctx 1024, batch 12, lr 6e-4→1e-5, 1000 mini-epochs ×
  10,000 seqs; L24/D768 shipped defaults), `src/model.py` (T_MAX compile-time,
  B_GROUP micro-batching, ATan surrogate, spikingjelly vendored),
  `src/trainer.py` (L208: weights-only `torch.save`), `requirements.txt`.
  GitHub API: repo `archived: true`. NOTE: one repo-tree listing call and one
  local hardware-probe call were denied by operator policy mid-session; no
  retry was attempted — hence step-zero §3.0 instead of measured numbers.
- HuggingFace API (today): file listings + sizes for
  `ridger/SpikeGPT-OpenWebText-216M` (862MB pth), `ridger/SpikeGPT-BookCorpus`
  (738MB pth), `Brain2nd/NeuronSpark-0.9B` (full code tree incl.
  `atomic_ops/*`; `config.json` `_tokens_seen: 203,910,528` /
  `_training_step: 85000`), `Brain2nd/NeuronSpark-V3-1.1B-*`;
  README cards fetched raw for both SpikeGPT repos (L18-D768, ctx 1024) and
  `TaewoooPark/SpikeGPT-216M-Mixture-Finetune` (created 2026-08-24; PPL table).
- Semantic Scholar citations API (arXiv:2302.13939, 100 most recent citers)
  sampled for replication candidates; abstracts/full-text greps pulled for
  SpikeLM 2406.03287, WTA 2604.11321, SpikeDecoder 2606.12287 via
  `arxiv.org/html`. S2 rate-limited (HTTP 429) on a second query round —
  coverage of the citation graph is one sample window, claimed as such.
- Colab FAQ fetched (qualitative quota statements quoted); Kaggle docs returned
  a JS shell — weekly-GPU-hour figures left explicitly UNVERIFIED.
- All dollar figures and hour figures marked ESTIMATE are planning inputs
  anchored on [DD-SGS §1.2] published run sizes; nothing here is a measurement.
