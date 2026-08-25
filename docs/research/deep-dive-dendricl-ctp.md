# Deep dive: DendriCL + CTP-Hybrid — does continuous dendritic LMS fix what APICAL-AND couldn't?

**Date:** 2026-08-25
**Status:** primary-source deep dive; both PDFs read in full (DendriCL 26 pp., CTP-Hybrid 4 pp.)
**Sources:** [arXiv:2607.02283](https://arxiv.org/abs/2607.02283) (Shen, Wu, Chen — HK PolyU, submitted 2026-07-02);
[DOI 10.5281/zenodo.20077294](https://doi.org/10.5281/zenodo.20077294) → v2 record [20120983](https://zenodo.org/records/20120983) (Shutong Hou, Shanxi Agricultural Univ., May 2026).
**Context:** `apical-and-verdict.md` (APICAL-AND NEGATIVE-FALSIFIED, best median CI 1.0290 vs gate ≥ 1.5),
`similar-projects-landscape.md` (both entries; CTP-Hybrid entry needs corrections, see §3.4).

## Bottom line

**Yes — but not by fixing the gate. By replacing it.** DendriCL never makes dendritic activity amplify
somatic output; it makes the dendritic *state itself* the locus of computation (a persistent,
multi-dimensional, error-driven accumulator), with the soma demoted to a lossy readout channel. Our
falsification showed the gate approach dies in a recruitment-limited regime (threshold multiplication
has nothing to multiply: recurrent cascade currents ~650 pA dominate afferent crossings). DendriCL's
own baselines provide *independent convergent evidence* for our negative result: the mechanistically
closest prior art to APICAL-AND — Iyer et al. Active Dendrites, an externally-driven context **gate**
— collapses to chance at d=20 (R²=0.061), while every architecture whose computation lives in
persistent state survives.

So: **APICAL-AND stays dead.** What DendriCL resurrects is the *other half* of Direction B that we
never tested — a continuous apical state carrying a **learned transformation** (an online estimator),
rather than a transient coincidence **gate**. That is a new, preregisterable variant (call it **B′ =
APICAL-LMS**) with a cheap falsifier that matches our one-shot discipline. CTP-Hybrid, meanwhile,
verifies as real and code-released, but its v2.0 **does not contain output-side first-spike
competitive decoding** — the landscape doc's characterization is wrong for the current version, and
our membrane-latency output race remains unprecedented in public work.

---

## 1. DendriCL: the exact mechanism

### 1.1 Architecture and equations (verbatim semantics)

Single layer of `d_model = d_apical = 384` pyramidal-like units, ~0.75M parameters. Token t encodes
`[x_t; y_t; flag_t]`. Per unit:

```
u_B(t)     = W_B · x_t                        # basal feedforward projection
ŷ_t        = u_A(t)ᵀ (W_A · x_t)              # scalar prediction READ OUT OF THE APICAL STATE
e_t        = (1 − flag_t)(y_t − ŷ_t)          # error, gated OFF at query position
u_A(t+1)   = α·u_A(t) + γ·e_t·(W_A · x_t)     # apical online-LMS  ← the entire mechanism
v_soma(t)  = g_B·u_B(t) + g_A·W_A,out·u_A(t)  # somatic integration (linear, both compartments)
s(t)       = 𝟙[v_soma > θ], soft reset        # LIF spike
```

Critical structural facts:

- **u_A ∈ ℝ³⁸⁴ is initialized to zero and is NOT reset by somatic spikes.** It evolves continuously
  across the full context. The paper italicizes this twice; it is the load-bearing design choice.
- With W_A = I, the apical update reduces *exactly* to classical leaky Widrow-Hoff:
  ŵ_{t+1} = α·ŵ_t + γ(y_t − ŵ_tᵀx_t)x_t. Proposition 1: under x_t iid Gaussian, y_t = wᵀx_t + ε,
  for 0 < γ < 2/d, E‖u_A(k) − w‖² = O(d/k). This is the 1960 LMS convergence theorem; the
  contribution is embedding the recursion in a compartmental SNN.
- **How "Widrow-Hoff in subthreshold dynamics" should be read honestly:** u_A is an abstract
  voltage-vector updated once per context position by an explicit recursion. There are **no
  conductance-based dynamics, no plateaus, no biophysical derivation** behind the update — the
  biological grounding is *anatomical* (three-compartment layer-5 layout, Ca²⁺ plateau timescales
  100+ ms cited as motivation), not dynamical. "Subthreshold" means only: the state never spikes and
  is never reset. For our purposes this is the key transfer risk (§4.2): nothing yet shows a fitted
  GLIF5 apical membrane can host this recursion.
- Training: ALL parameters (α, γ, W_A, W_B, W_A,out, g_A, g_B, θ, FF weights) learned end-to-end by
  **BPTT** (AdamW, lr 1e-3, arctan surrogate, batch 64, 10k/50k steps, loss at query position only).
  **No parameter updates at inference** — adaptation during inference is pure frozen-weight dynamics.

### 1.2 What BPTT does and does not do (their own clarification, App. J)

"The architecture forces the apical recurrence to have LMS structure; BPTT's only degrees of freedom
are the step-size-like hyperparameters (α, γ) and the input projection W_A." Learned α → 0.994–0.997
(integration horizon 1/(1−α) = 111 steps at d=5 → 400 at d=30, ≈6–7× the context length k=2d);
learned γ ≈ 0.2× the theoretical optimum γ⋆ = 1/(d+2), always inside the stability bound γ < 2/(d+2)
in all 18 configurations. They explicitly do NOT claim BPTT discovers LMS from a larger class — the
structure is built in; BPTT tunes constants.

**Implication for us:** the inference-time update Δu_A = αu_A + γe_t x_t is a *local three-factor
rule* (presynaptic x_t, error e_t, postsynaptic state u_A). Only the meta-tuning of (α, γ, W_A) is
non-local/global-gradient. Hand-set constants inside the stability region are defensible without
violating our local-learning contract — the same way our pinned Allen fits coexist with designed
plasticity rules.

### 1.3 Results that matter

| Result | Number | Note |
| --- | --- | --- |
| Garg-2022 ICL, d=10 / d=20 | R² 0.807 / 0.820 | Best spike-based model; Transformer 0.996/0.989 |
| Super-d seed-stability (50k) | 0.809 (d25), 0.807 (d30), 0.787 (d40), 0.649 (d50), σ ≤ 0.036 | Only architecture > 0.5 at d≥30; Transformer trimodal/grokks at d30, chance-floor at d≥40 |
| Continuous ablation (no LIF) | 0.833 (+3 pp over DendriCL) | Compartmental structure drives capability; spike binarization costs only ~3 pp |
| Pure LIF baseline | 0.801 @d10 → 0.086 @d20 (chance) | Scalar membrane, reset each spike ⇒ no persistent multidimensional state ⇒ no substrate |
| **Active Dendrites (Iyer 2022)** | **0.668 @d10 → 0.061 @d20 (chance)** | External context **gate** — mechanistically closest to our APICAL-AND. Collapses. |
| Apical linear probe | R²(u_A → reference online-LMS trajectory) = 0.873 @d10, **0.931 @d20**, 0.97 @d30; holds ≥ 0.87 at every context position | The algorithm is structurally IN the dynamics, not discovered |
| Probe → true parameter | R² 0.820–0.837 ≈ task-level eval R² | ICL capability quantitatively explained by apical encoding quality |
| Width cliff | d_apical ≤ 384 trains; ≥ 512 diverges at d=20 (cliff ≈ d × 30) | Over-parameterized apical compartment pushes γ past LMS stability |
| Energy | 414 spikes/token; 15.3 nJ vs Spikformer 79.5 nJ (Loihi model) | ~6× per correct prediction; ~4× fewer spikes than Pure LIF |
| Off-LMS tasks | Binary clf 0.805; 2-layer ReLU regression 0.568 (Spikformer 0.582) | Advantage shrinks where no clean online-LMS interpretation exists |

Failure-mode taxonomy (App. K) is a methodological bonus: grokking (Transformer), clean ceiling
(Spikformer), high-variance partial rescue (Pure LIF σ=0.135), uniform terminal failure, and
"graceful absence of failure" (DendriCL, smooth monotonic curves from step 0, σ ≤ 0.036).

### 1.4 Stated limitations (their §7, plus code status)

Synthetic Garg function classes only — no language/vision; theorem covers the linear apical
compartment only (LIF reset nonlinearity unhandled formally; probe carries it empirically);
d ≤ 50 tested; energy claims projected, not measured; biological Tier-3 hypothesis (cortex implements
ICL via apical-LMS) explicitly falsifiable-not-established. Code: released as the "**snn-icl-bench**"
repository (~1,800 lines: models.py, tasks.py, train.py + raw result JSONs, per Appendix M). ⚠️ I
could not independently verify a live repo URL from this environment (external search unavailable);
treat code-existence as claimed-by-paper until checked.

## 2. Why it works where our APICAL-AND didn't

Direct mapping of our falsified mechanism onto theirs:

| Dimension | APICAL-AND (ours, falsified) | DendriCL | Verdict |
| --- | --- | --- | --- |
| Role of apical compartment | Gate: modulate somatic excitability (multiplicative plateau-gated threshold lowering) | Estimator: u_A IS the computation (running regression solution) | Role reversal — the decisive difference |
| Information carrier | Somatic spike count / burst recruitment | Persistent 384-dim analog subthreshold state | Continuous analog state, yes — but specifically *persistent and accumulative*, not merely analog |
| Memory across context | None (transient coincidence at delivery instants) | α→0.997 ⇒ integrates ~400 steps ≈ whole context | Accumulation integrates k supervised examples; coincidence gating cannot integrate |
| Error signal | None used | e_t = y_t − ŷ_t, local three-factor update | Supervision exists in our replay too (next token) but never fed a dendritic accumulator |
| Dependence on somatic gain | Total (needed CI ≥ 1.5× burst amplification) | None (spikes optional: removing them costs 3 pp) | In recruitment-limited regimes, gates multiply noise; estimators don't need gain |
| Readout tolerance | Required precise threshold arithmetic at soma | LIF reset tolerated; probe/readout downstream | Our G-A2/G-A3 showed states near-collinear AND unreadable — theirs decodable by a linear probe |

Answering the framing question head-on: **is it "continuous analog state vs our spike-threshold
plateau"? Partly, but that undersells it.** Three stacked differences: (1) the apical compartment is
the *estimator*, not a *modulator*; (2) the update is *error-driven accumulation* over the context,
not instantaneous coincidence detection; (3) the readout tolerates brutal somatic nonlinearity
because decoding happens downstream of the state. Our failure was localized at G-A1 (recruitment-
limited regime defeats threshold multiplication); DendriCL sidesteps that regime entirely by never
asking the soma to compute.

And the convergent-evidence point bears repeating: **Active Dendrites (context gate) collapsing to
chance while DendriCL (context estimator) succeeds on the identical benchmark is the cleanest
independent replication of the direction of our falsification.** Gate-style dendritic use fails;
state-style use scales. Our negative result is now corroborated, not contradicted, by this preprint.

## 3. CTP-Hybrid: verification results

### 3.1 Existence: CONFIRMED, upgraded from low confidence

Real artifact: Zenodo concept DOI 10.5281/zenodo.20077294 (what our landscape doc cites) resolves to
version-2 record 20120983, "CTP-Hybrid: From Hybrid Architecture to Native Spiking Foundation — A
Two-Phase Report on Consumer-GPU Spiking LLMs," Shutong Hou, Shanxi Agricultural University, Project
NeuroVein/Aurora Series, Preprint v2.0, May 2026. **Code is now archived**: `neurovein_v2_code.zip`
(9.53 MB) alongside `main.pdf` in the record files — the landscape doc's "promised official code not
located" was true at research time and is now stale.

### 3.2 What it actually reports (Phase 1 → Phase 2)

- **Phase 1 (hybrid):** Qwen2.5-7B-Instruct (4-bit NF4 + QLoRA r=8), FFN of decoder layers 2/6/10
  replaced by TD-SCA spiking microcolumns (32-neuron LIF columns, learnable delays). WikiText-2 PPL
  16.58 vs 10.31 vanilla backbone; CLUTRR 53.5% (spiking-augmented, STE, unfine-tuned) vs 69.5%
  vanilla. Cloud run converged to a degenerate region (PPL 624.73) under a 2,000-sample constraint.
  Tri-module joint training (IST+TCP+DSM) failed 7× with NaN — honestly tabulated.
- **Phase 2 (native spiking):** three custom `autograd.Function`s — IntervalSpikeEncoder (pure STE,
  sigmoid derivative deliberately dropped), TemporalCausalMask (zero-gradient; mask treated as
  constant), LIFSpikeSurrogate (triangular window) — unlock zero-NaN tri-module joint training at 7B
  hybrid scale, plus a fully Transformer-free 20M-parameter baseline (vocab 32k, D=1024, T=64, 3
  layers): **WikiText-2 eval PPL 13.57**, trained from scratch on ≤ 5,000 sentences with a character-
  hash tokenizer. No attention, no LayerNorm, no floating-point FFN, no softmax decoding.
- Self-declared ceilings: pure-spiking path validated only at 20M params; ≤ 5k sentences; only
  WikiText-2 PPL evaluated for Phase 2; Co-STDP path designed but never run.

### 3.3 The critical correction: there is NO output-side first-spike race

First-spike timing appears exclusively in the **input encoder**: IST maps each token ID to a pulse
train via a *learnable first-spike time* τ_i plus inter-spike-interval increment Δ_i ("semantic
information is carried by inter-spike interval ratios"). TCP derives causal masks from earliest-spike
times. But the **output path is SpikeReadoutV3: output spikes → vocabulary LOGITS** via learnable
time embeddings, spike-rate normalization, and linear projection — argmax-over-logits decoding, not a
spike-latency race among output cells. "No softmax" refers to omitting the softmax layer, not to
competitive temporal decoding.

Consequences for us: (a) the landscape doc row "explicitly proposes first-spike competitive decoding
for output tokens … closest in output-decoding language" is **wrong for v2.0** and must be corrected;
(b) our earliest-spike vocabulary race remains **without precedent** among these systems — the
feature-assessment table's "Earliest-spike vocabulary decoding: a very recent claim exists" should be
reversed; (c) nothing in CTP-Hybrid threatens or validates the race mechanism.

### 3.4 Evidence-quality assessment (unchanged conclusion, updated facts)

Single author, unreviewed Zenodo preprint, nonstandard filtered data splits, character-hash tokenizer
fragmentation acknowledged by the author, perplexities not comparable across papers, generation
quality unmeasured. Real tables, specific losses, and now shipped code make it *checkable*, which
raises it from "low confidence / unverifiable" to "low-medium: plausible engineering report pending
independent reproduction." Still not a foundation to build on — and architecturally orthogonal to our
contract (surrogate-gradient backprop throughout, learned embeddings at input, logit readout at
output).

## 4. Integration verdict

### 4.1 Adopt / adapt / avoid

| Disposition | Item | Rationale |
| --- | --- | --- |
| **ADOPT** | Computation-in-dendritic-state principle: persistent multidimensional subthreshold state carries the learned transformation; soma = readout only | Directly answers our G-A1 failure mode: stops depending on somatic gain in a recruitment-limited regime |
| **ADOPT** | Linear-probe verification methodology: freeze checkpoint, ridge-probe internal state against a reference algorithm trajectory; gate on probe R² BEFORE scaling | Zero-training diagnostic applicable to our already-saved `logged_states_apical_cohort.npz`; would have caught G-A2 collapse earlier and cheaper |
| **ADOPT** | Seed-stability reporting: report per-seed distributions and failure modes, not just mean±std; treat grokking-style bimodality as an architectural warning sign | Their trimodal Transformer at d=30 (σ=0.479) vs our one-shot N=1024 discipline — complementary protections |
| **ADAPT** | **B′ = APICAL-LMS** (new Direction-B variant, replaces dead APICAL-AND): give `TwoCompartmentGLIF5` a persistent multi-dim apical vector updated by local rule Δu_A = α·u_A + γ·e_t·x_t during observe() (e_t = next-token prediction error, available in replay), error gated OFF at decision points, decode from u_A at decision points | Resurrects the untested half of Direction B (estimation, not gating); update rule is local three-factor; α, γ settable by hand inside LMS stability bounds (γ < 2/(d_eff+2)) without violating the local-learning contract |
| **ADAPT** | Retro-probe of existing cohort states: rerun G-A2/G-A3-style evaluation using LINEAR probes over continuous subthreshold voltages (not just trace banks/spike counts) against decision-point variables | If even continuous-state probes sit inside the shuffled-label band, the problem is upstream state construction, independent of any mechanism — kills B′ at the corpus level before any new build |
| **ADAPT** | CTP-Hybrid's STE-surrogate toolkit (custom autograd.Function per pulse op; drop sigmoid-derivative multiplication; zero-grad masks) | Only relevant if a gradient-trained component ever enters (today a contract violation); recorded as the known cure for pulse-op NaN collapse at joint-training scale |
| **AVOID** | Any return to plateau-gated coincidence amplification | Our CI 1.03× falsification now independently corroborated by Active-Dendrites' collapse on the same qualitative pattern; DendriCL contains no gating-on-threshold component, so its success lends the dead mechanism zero support |
| **AVOID** | Importing DendriCL's BPTT meta-training wholesale | Breaks the local-learning contract unnecessarily — the architecture forces LMS structure and only tunes (α, γ, W_A); hand-set constants inside the stable region suffice for a falsifier |
| **AVOID** | Treating DendriCL as a language-model or human-cell result | Synthetic linear-regression ICL, d ≤ 50, clean scalar labels, stationary-task prior, abstract (non-biophysical) apical update; advantage visibly shrinks off-LMS tasks (ReLU regression 0.568 vs Spikformer 0.582); OOD degrades sharply beyond w_std = 2.0; Tier-3 biology unestablished |
| **AVOID** | Counting CTP-Hybrid as precedent, competition, or evidence for our output race | v2.0 decodes via logits; unreviewed; tiny filtered training sets; hash tokenizer; 20M-param ceiling self-admitted |
| **AVOID** | Apical dimensionality without a stability budget | Width cliff: d_apical ≥ 512 diverges at d=20 (γ exits stable region). Any B′ must declare d_apical and γ jointly inside γ < 2/(d_eff+2) at freeze time |

### 4.2 Does DendriCL resurrect part of the escalation ladder?

**One specific rung, transformed: yes.** Not the falsified rung (plateau-gate AND — dead, corroborated
dead), but the ladder's underlying premise that dendritic state can host computation. B′ is a genuine,
distinct, preregisterable experiment with a sharper mechanistic target than APICAL-AND ever had:

1. **Stage 0 (cheap, days):** Garg-2022-style regression task driven through the EXISTING
   `TwoCompartmentGLIF5` substrate with a hand-implemented local LMS apical update (flag-off parity
   unchanged; u_A rides alongside). Falsifier: linear-probe R² of apical trajectory vs reference
   online-LMS ≥ pre-set threshold (their benchmark shape: ~0.85–0.93). If a fitted GLIF5 apical
   membrane cannot host the recursion, kill B′ here — this is the single biggest unknown, because
   DendriCL's apical compartment is an abstract linear recurrence, NOT fitted physiology.
2. **Stage 1:** decision-point decoding from u_A on corpus replay (error = next-token prediction
   residual during observe(); gated off at decision points), against the SAME G-A3 controls
   (majority prior, shuffled-label band) that killed APICAL-AND.
3. **Stage 2:** only if Stage 1 clears controls: integrate u_A as an additional input to the
   output-cell race (never as a threshold gate).

Stop-on-failure at every stage; cohort spent once; all gates frozen before implementation — same
one-shot discipline as G-A1.

### 4.3 Corrections owed to `similar-projects-landscape.md`

1. CTP-Hybrid row: code NOW located (`neurovein_v2_code.zip`, 9.5 MB, record 20120983, v2.0 May 2026).
2. CTP-Hybrid row: "first-spike competitive decoding for output tokens" is inaccurate for v2.0 —
   first-spike timing is INPUT-side (IST encoder) only; output is logit-based. Feature table row
   "Earliest-spike vocabulary decoding: a very recent claim exists (CTP-Hybrid)" should flip to "no
   public precedent found."
3. DendriCL section: add code-release claim ("snn-icl-bench", ~1,800 lines + raw JSONs; URL
   unverified from this environment) and the Active-Dendrites-collapse datapoint.

(A dated addendum pointing here has been appended to the landscape doc.)

## Files / provenance

- Primary texts cached: `%LOCALAPPDATA%/Temp/dendricl.pdf|.txt`, `ctp_hybrid.pdf|.txt`,
  `dendricl_abs.html`, `ctp_hybrid.html`, `ctp_api.json` (session-temp only, not committed).
- This document: `docs/research/deep-dive-dendricl-ctp.md`.
- Addendum: appended to `docs/research/similar-projects-landscape.md`.
