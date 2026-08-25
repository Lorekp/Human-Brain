# Deep dive: NEST e-prop, HuMAM, and Digital Twin Brain — validated three-factor learning at scale

**Date:** 2026-08-25
**Status:** primary-source deep-dive (fetched full texts; equations transcribed from official docs)
**Scope:** NEST e-prop model docs + Bellec et al. 2020 + Korcsak-Gorzo et al. 2025; HuMAM (Pronold et al. 2024); Digital Twin Brain (Lu et al. 2024) plus its 2026 multitask-behavior follow-up. Companion to [similar-projects-landscape.md](similar-projects-landscape.md).

## Bottom line

1. **e-prop is the only publicly established eligibility-based three-factor learning rule that demonstrably trains nontrivial sequence behavior.** The exact per-synapse formulation is short and fully specified (below). On TIMIT phoneme recognition it "approximates the performance of BPTT" for LSNNs of 800–2400 neurons, and a reward-based variant learns Atari Pong — all online, local, and without storing network history. The NEST implementation (Korcsak-Gorzo et al. 2025) makes it event-driven, strictly local, delay-correct, and demonstrates scaling to **2 million neurons with plastic synapses** on CPU clusters.
2. **Neither HuMAM nor DTB learns anything.** HuMAM (3.47M LIF neurons / 42.8B synapses, one hemisphere, 34 areas) has an explicit `Plasticity: none` row in its model summary and runs resting-state dynamics under Poisson drive. DTB (up to 20B LIF neurons / 2T synapses) fits *input currents* to reproduce a subject's task BOLD ("assimilation") — task scores are predicted from evoked dynamics, not trained. The 2026 "multitask behavior" DTB paper does learn behavior, but via a **backprop-trained hypernetwork+RNN**, not inside a spiking brain model.
3. **For path 2 (developmental connectivity shaping via passage statistics), e-prop supplies the credit-assignment template; HuMAM/DTB supply scale-engineering patterns but no learning template.** The adopt/adapt/avoid table below maps each onto our procedural address-space runtime.

---

## 1. NEST e-prop: the validated three-factor rule at scale

### 1.1 Network architecture (four co-designed models)

An e-prop model in NEST is not one model but a wiring pattern of four ([model index](https://nest-simulator.readthedocs.io/en/stable/models/eprop_iaf_bsshslm_2020.html)):

| Component | Model | Role |
| --- | --- | --- |
| Recurrent/input neuron | `eprop_iaf_bsshslm_2020`, `eprop_iaf_adapt_bsshslm_2020` | LIF (± adaptive threshold ALIF), archives continuous quantities needed by synapses |
| Readout neuron | `eprop_readout_bsshslm_2020` | Non-spiking graded readout; computes error/learning signal |
| Plastic synapse | `eprop_synapse_bsshslm_2020` | Collects presynaptic spikes, triggers gradient computation, applies optimizer |
| Third-factor channel | `eprop_learning_signal_connection_bsshslm_2020` | Dedicated edge carrying learning signal L_j^t from readout to every recurrent neuron |

Two families exist: `_bsshslm_2020` (Bellec et al. 2020 formulation) and the plain names (extended/biologized variant of Korcsak-Gorzo et al. 2025, "e-prop+").

### 1.2 Exact eligibility-trace formulation per synapse (from official model docs)

Gradient decomposition — every weight update is **learning signal × filtered eligibility trace**:

```
dE/dW_ji = Σ_t L_j^t · ē_ji^t
```

Non-adaptive LIF (`eprop_iaf`): the eligibility trace is just surrogate-gradient × filtered presynaptic spike trace:

```
e_ji^t = ψ_j^t · z̄_i^{t-1}
z̄_i^t = α·z̄_i^{t-1} + z_i^t          (α = exp(−Δt/τ_m))
ē_ji^t = F_κ(e_ji^t),  κ = exp(−Δt/τ_m,out)   (low-pass filter with readout's membrane constant)
```

Adaptive-threshold ALIF (`eprop_iaf_adapt`) adds a recursive second-order term so adaptation dynamics contribute to credit:

```
e_ji^t = ψ_j^t · ( z̄_i^{t-1} − β·ε_{ji,a}^{t-1} )
ε_{ji,a}^{t-1} = ψ_j^{t-1}·z̄_i^{t-2} + (ρ − ψ_j^{t-1}·β)·ε_{ji,a}^{t-2}     (ρ = exp(−Δt/τ_a))
A_j^t = v_th + β·a_j^t,   a_j^t = ρ·a_j^{t-1} + z_j^{t-1}
```

ψ_j^t is one of four built-in surrogate gradients (piecewise-linear à la Bellec, exponential, fast-sigmoid derivative, arctan derivative). A firing-rate regularization gradient is added to the objective: `(c_reg/T·n_trial)·Σ_t (f_target − f_av_j)·e_ji^t`. Weight updates go through pluggable optimizers (gradient descent or Adam; per-synapse `optimizer` dict). Documented caveat: the rule ignores precise spike timing within a solver step.

**Multi-compartment / recurrent topologies:** e-prop as implemented does *not* do multi-compartment neurons. Recurrence is handled at the network level — the derivation treats each recurrent connection like a feedforward one whose "presynaptic term" happens to be another neuron's spike train; delays are arbitrary per synapse (`delay d_ji`), decoupled from the solver step (the original Bellec supplement assumed a single-step delay; the NEST version generalizes and repairs the missing end-of-sample learning signals by accumulating gradients over left-open intervals). Dendritic morphology would have to be expressed as multiple point-neuron compartments connected by plastic or static links — which matches our branch-as-cell construction rather than offering anything beyond it.

### 1.3 Event-driven implementation (the engineering core — Korcsak-Gorzo et al. 2025)

The TensorFlow original updates everything at every time step. The NEST port converts this into a hybrid scheme ([arXiv:2511.21674](https://arxiv.org/abs/2511.21674)):

- **Synapses (event-driven)** store only presynaptic traces z_i^{t−1}; they wake up on spikes.
- **Neurons (time-driven)** store the continuous quantities ψ_j^t and L_j^t in bounded archives.
- **Update triggering:** the first spike of sample n+1 triggers retrieval of the archived history for sample n and computation of that sample's gradients (the "flush event" mechanism). Histories are cleared once all synapses have retrieved their data or had no presynaptic spikes — memory stays proportional to live samples, not to total simulation length.
- Gradients accumulate over mini-batches; Adam/GD produce the actual update.
- Strict locality enforced throughout: each synapse touches only presynaptic state, postsynaptic state, and the postsynaptic neuron's learning signal.
- e-prop+ variants push toward biology: continuous-time dynamics and continuous weight updates, weight update on every spike, eligibility filter decoupled from the readout time constant, smooth surrogate gradients, dynamic firing-rate regularization. Reported result: convergence speed and final test error match vanilla e-prop on N-MNIST.

**Scale results:** runtime measurements up to **2 million plastic neurons** show near-linear weak scaling and good strong scaling (body text says super-linear; the figure caption says sub-linear — cite conservatively as sub/near-linear). Plasticity measurably costs more than static synapses, and the third factor costs more than two-factor STDP, but the overhead is bounded. Benchmark configuration: 20 s biological time = 20,000 × 1 ms steps, 32 cores/task, 4 tasks/node.

### 1.4 Biological-plausibility claims (and their honest limits)

- The plausibility argument is about **what is stored and when**: BPTT must archive all neuron states and process them offline backward, which the authors argue makes BPTT brain-implausible and hardware-unfriendly; e-prop needs only forward-computed eligibility traces plus a locally available third factor (Bellec et al. 2020).
- Neuron-specific feedback (readout → each neuron's L_j^t) outperforms global-signal baselines on TIMIT; random vs symmetric vs adaptive feedback performed similarly, which softens the weight-transport objection without eliminating it (readout still must compute a loss-derived signal).
- Reward-based e-prop replaces teacher targets with actor-critic + a global reward-prediction-error broadcast — the closest published form of "three factors with a diffuse neuromodulatory third factor."
- Limits we should keep in view: ψ (surrogate gradient) and the exact L_j^t functional form remain engineered constructs; "biologically plausible" here means *local, online, low-memory*, not experimentally verified. e-prop+ moves further but is still gradient descent in disguise.

### 1.5 Performance vs BPTT: can eligibility-based learning train sequence behavior? Yes.

From Bellec et al. 2020 (*Nature Communications* 11:3625):

| Task | Network | Result |
| --- | --- | --- |
| TIMIT framewise phoneme classification (61 classes, 10 ms frames) | LSNN, 800 neurons (LIF+ALIF) | e-prop ≈ BPTT; pure-LIF RSNNs don't reach good TIMIT even under BPTT — adaptation (ALIF) is load-bearing |
| TIMIT sequence transcription (whole sentences) | LSNN, 2400 neurons | e-prop ≈ BPTT again; solved with no neuron above 12 Hz (energy-efficient regime) |
| Handwriting generation (regression) | LSNN | reproduced from the original TF implementation; also a standard NEST tutorial |
| Atari Pong (deep RL) | LSNN actor-critic | reward-based e-prop wins games; episode-length/LR schedules replace the biologically implausible parallel-learner trick |
| Attributed evidence accumulation, pattern generation | small nets | proof-of-concept reproductions |

The decisive architectural finding for us: **adaptation (ALIF thresholds) is what buys long-horizon temporal credit** — the ε_a recursive trace term carries the extra gradient. Our GLIF5 after-spike-currents are a richer analogue; if we ever want gradient-style diagnostics, the ALIF term shows exactly how adaptation enters eligibility.

NEST-side verification (Korcsak-Gorzo): event-driven scheme replicates the time-driven results on pattern generation, evidence accumulation, and N-MNIST classification (10 seeds; mean prediction errors reported graphically).

---

## 2. HuMAM: human-cortex structure at 3.47M neurons (no learning)

Source: Pronold et al. 2024, *Cerebral Cortex* 34(10):bhae409 ([PMC11491286](https://pmc.ncbi.nlm.nih.gov/articles/PMC11491286/)), Jülich IAS-6.

**What it is:** one cortical hemisphere, Desikan–Killiany parcellation, 34 areas × mostly 8 populations (L23/L56 × E/I) = 254 populations; ~3.47M neurons ("about 3.5 million") and 42.8B ("43 billion") model-internal synapses at **full natural density** for the modeled layers. Point LIF neurons with exponential PSCs and fixed absolute refractory periods. Connectivity is random *within* area/layer/population but parameterized per area/population (degrees, weights, delays all data-derived). Input is independent homogeneous Poisson drive standing in for the unmodeled rest of the brain. The model summary table literally reads `Plasticity: none`.

**How it reaches that scale (engineering patterns):**

- **Populations, not positions:** neurons carry no spatial coordinates; all neurons in an area/layer/population are statistically equivalent. Connectivity becomes ~a table of population-pair parameters instead of billions of individual edges — this is the same move as our procedural topology (connectivity as function of identity, not stored adjacency).
- **Standard HPC NEST machinery:** NEST 2.20.2 on JURECA-DC; MPI-rank partitioning with structured workflows (Snakemake); mean-field cross-checks via NNMT.
- **Concrete budget:** simulating **10 s of biological time takes ≈200 core-hours** — "(1 min build phase + 15 min for 10 s biological time on 768 cores)" (numbers recovered from the paper's inline equation graphics). That is a resting-state-grade workload: no plasticity bookkeeping, tiny state per neuron.
- **Validation discipline:** the connectome is validated against features *not built in* — held-out human spiking statistics (medial frontal cortex), resting-state fMRI FC, single-spike perturbation propagation. FC only matches when inter-areal synaptic strength is scaled up beyond the anatomical baseline — an instructive example of a fitted fudge factor being reported openly.

**Human-data integration method:** multi-modal fusion — electron-microscopy-derived synapse densities, electrophysiology, morphological reconstructions, and DTI tractography for inter-areal pathways; gaps filled using statistical regularities transferred from other species (notably laminar origin/target distributions).

**Learned behavior:** none. Resting state, perturbation propagation, and FC comparison only. It is a structural/dynamical reference, not a training system.

## 3. Digital Twin Brain: 20B neurons by assimilation (no learning either)

Source: Lu, Zeng, Wang, Xiang, Qi, Zheng, Xu, Feng 2024, *National Science Review* 11(5):nwae080 ([PMC11129584](https://pmc.ncbi.nlm.nih.gov/articles/PMC11129584/)), Fudan FTBI.

**What it is:** subject-personalized cortico-subcortical spiking model: up to **20 billion LIF neurons and 2 trillion synapses at 1.73% connection density**, 374 regions (HCPex parcellation incl. subcortex) over 16,043 voxels. Each voxel is a micro-column: cortex 8 populations, subcortex 2, E:I ≈ 4:1; AMPA + GABA synapses; Ornstein–Uhlenbeck background current; simulated firing rates pushed through a Balloon–Windkessel model to generate synthetic BOLD.

**How it reaches 20B (engineering patterns):**

- **GPU clusters** run the spiking network (hardware/performance details are relegated to supplementary material, not in the main text — flagged honestly).
- **Voxel-partitioned microcolumns** keep local circuits dense-but-small and long-range connectivity sparse (1.73%): global structure is again parameters (tract probabilities between voxel columns), not stored edges.
- **Assimilation loop (Vw-dHMDA)** — voxel-wise diffusion hierarchical mesoscale data assimilation: fit each voxel's input-current distribution so simulated BOLD tracks recorded BOLD, hierarchically from region level down to voxels. This is how "human data" enters: as a fitting target for inputs and architecture, not as supervision for weights.

**Human-data integration:** multimodal — T1 sMRI (geometry), multi-shell DWI tractography (long-range structure), SV2A PET (synaptic density per voxel), task/rest fMRI (functional target). Single-subject personalization end to end.

**Key scientific result worth stealing:** similarity to the biological counterpart (FC correlation, BOLD time courses, avalanche criticality, synchrony) **increases monotonically with neuron count and average in-degree**, in rest and in tasks — an explicit empirical scaling law connecting simulation scale to fidelity. Conversely, progressively rewiring the DWI graph toward local-only connectivity degrades similarity, and digital lesioning of V1→ventral/dorsal pathways selectively damages hippocampal involvement in visual tasks.

**Learned behavior:** none in this paper. Tasks (auditory, visual evaluation) are handled by assimilating input-region currents (Gamma-distributed drive fitted at A1/V1) until simulated task BOLD matches the subject's; predicted task scores correlate with experimental scores (auditory task: voxel-mean PCC 0.570; score correlation r = 0.585, p = 0.001). Impressive, but it is stimulus-response curve-fitting — no synaptic variable changes with experience.

**The confusing sibling:** a 2026 BMEF paper, "Digital Twin Brain: Generating Multitask Behavior from Connectomes for Personalized Therapy" ([PMC12895553](https://pmc.ncbi.nlm.nih.gov/articles/PMC12895553/)), *does* predict learned behavior (>90% choice accuracy, RT r>0.85, BOLD r=0.84 across 228 subjects) and runs in-silico interventions. But its architecture is a **hypernetwork that personalizes a main RNN from a resting-state connectome, trained end-to-end by gradient backpropagation**. It shares branding and team lineage, not mechanism, with the 20B spiking model. Cite carefully: there is still no instance of learned behavior inside a billion-scale spiking model.

---

## 4. Integration verdict for path 2 and the 100B address space

Path 2 context (from [roadmap-reading-to-coding.md](roadmap-reading-to-coding.md): R3 episodic store; r0c strategy: associative-memory-through-race): connectivity shaped developmentally by passage statistics — "trained-by-construction" rather than trained-by-loss. What do the three peers actually give us?

**(a) e-prop as learning-rule template — adopt the skeleton, replace the third factor.**
Our GLIF5 lane already keeps eligibility traces; e-prop contributes the *validated decomposition*: per-synapse trace e_ji (pre-trace × post-surrogate-gradient) multiplied at update time by a third factor routed through a dedicated connection class. For passage-statistics shaping, the first two factors transfer unchanged; the supervised L_j^t = dE/dz_j^t does not fit our cell-only contract. Two defensible substitutions preserve the template's value:
  - a *local novelty/consolidation factor* computed from race outcomes and replay statistics (closer to reward-based e-prop's global RPE broadcast than to per-neuron gradients); or
  - use e-prop proper only as an **offline diagnostic**: run it on small resident subsets to estimate whether candidate developmental rules move weights in a direction compatible with sequence performance (cheap oracle, never part of inference).
Also directly reusable: rate regularization as a homeostat term (matches our dormant activity-EMA homeostat plan), the ALIF ε_a recursion showing how adaptive-threshold state enters eligibility, and the flush-event archive discipline that caps learning-memory by touched cells.

**(b) Scale engineering — both platforms confirm our existing contract and add two numbers.**
HuMAM and DTB never scan populations, store no spatial coordinates, and treat connectivity as parameterized generators over statistically equivalent populations — the same design our procedural runtime already enforces (O(R_capacity + Q_capacity + S_capacity + F + D_capacity), `configured_ids_scanned = 0`). New quantitative anchors:
  - **DTB: 1.73% connection density at 20B neurons** supports ultra-sparse procedural fan-out for the 100B address space, and their monotone similarity-vs-(neurons × in-degree) scaling law gives us a citable precedent for expecting fidelity gains from resident growth even before any learning.
  - **HuMAM: ≈200 core-hours per 10 s biological** for 3.47M static LIF neurons is a sobering cost floor for dense-rate point cells; it prices how much cheaper our event-frontier execution must stay, and how much plasticity overhead (per Korcsak-Gorzo: measurable but bounded vs static synapses) we can afford at R4/R5 scale ladders.
Neither peer requires changing our declared addressable/materialized/updated/spiking accounting — if anything their reporting style (population-level summaries, explicit fudge factors) reinforces the repo's honesty contract.

**(c) Human-data integration — assimilation, not supervision, is the transferable pattern.**
DTB's Vw-dHMDA fits *inputs* (current distributions) against observed outputs while leaving weights structural; HuMAM validates structure against held-out dynamics. For path 2, passage statistics should likewise enter as **drive statistics and growth biases** (which branches sprout, which assemblies get reinforced), validated against held-out behavioral probes — not as per-token target signals. That preserves the cell-only output contract while still being "human-data-constrained."

### Adopt / adapt / avoid

| Practice | Source | Verdict | How / why |
| --- | --- | --- | --- |
| Three-factor decomposition dE/dW = Σ L_j^t · ē_ji^t | e-prop (Bellec 2020; NEST docs) | **adopt (structure)** | Keep trace × signal product as the universal update shape for path-2 shaping; substitute a local passage-statistics/novelty factor for the supervised L. |
| Per-synapse eligibility trace e_ji^t = ψ_j^t·z̄_i^{t-1} (+ ALIF ε_a recursion) | NEST `eprop_iaf(_adapt)` docs | **adopt (formulas)** | Exact, cheap, already close to our GLIF5 trace lane; ε_a term is the missing piece for adaptation-bearing cells. |
| Event-driven synapse storage + time-driven neuron archives + flush-event retrieval | Korcsak-Gorzo 2025 | **adopt (pattern)** | Bounds learning memory by touched cells; maps 1:1 onto our resident-frontier/event-queue runtime. |
| Transmission-delay handling decoupled from solver step; left-open-interval gradient accumulation | Korcsak-Gorzo 2025 | **adopt (discipline)** | Any future trace-based rule in our heterogeneous-delay topology needs the same repair logic. |
| Firing-rate regularization term (homeostatic third component) | Bellec 2020 / NEST docs | **adapt** | Repurpose as the stabilization homeostat scheduled for R3+; keep c_reg small and local. |
| Surrogate-gradient menu (4 functions) | NEST docs | **adapt** | Only for offline diagnostic estimates of developmental-rule quality; never in the production cell-only path. |
| Reward-based e-prop (actor-critic + global RPE, schedule-instead-of-parallelism) | Bellec 2020 | **adapt** | Template for a diffuse third factor compatible with race outcomes (win/lose as scalar broadcast). |
| Population-parameterized connectivity; no spatial coordinates; statistically equivalent neurons | HuMAM | **adopt (already ours — keep)** | Confirms procedural topology as the scale-proven way to hold 100B identities without storing edges. |
| Ultra-sparse long-range density (~1.73% at 20B) + microcolumn partitioning | DTB | **adopt (parameters)** | Sets the sparse-fanout prior for 100B addressable / small-resident operation. |
| Similarity-increases-with-scale law (FC/criticality vs neurons & in-degree) | DTB | **adapt** | Reframe behaviorally: predict capacity/transfer improvements along our R4 scale ladder as resident count grows; citable precedent. |
| Assimilation-style human-data integration (fit drives, validate on held-out dynamics) | DTB Vw-dHMDA; HuMAM validation | **adopt (method)** | Passage statistics → drive/growth-bias fitting; validate against held-out QA/replay probes, not token-level targets. |
| Supervised L_j^t from a differentiable loss inside the production loop | e-prop | **avoid (in production)** | Violates the cell-only/local-objective contract; permissible only as offline diagnostic. |
| Expecting scale platforms to provide learned behavior | HuMAM, DTB | **avoid (claim-wise)** | Neither trains its spiking population; the 2026 multitask DTB is a backprop-trained RNN hypernetwork — don't let branding blur that. |
| Dense per-step archiving of continuous variables for all neurons | TF e-prop; partially NEST time-driven side | **avoid** | Unbounded memory at our scales; keep archives bounded by touched/resident cells with explicit flush. |
| Population-wide scans during build/update | neither peer does this | **avoid** | Both peers are partition-and-parameter architectures; our `configured_ids_scanned = 0` invariant stays. |

### What would change this assessment

- A published e-prop (or equivalent) run on a *language-generation* objective rather than TIMIT/N-MNIST-class tasks — none found as of 2026-08-25.
- Any peer integrating real plasticity at HuMAM/DTB scales — none found; NEST e-prop's 2M-neuron plastic scaling is the current ceiling.
- Access to DTB's supplementary hardware/performance file (main text defers GPU specifics there) — would sharpen the cost model but unlikely to change the verdict.

## Sources

- NEST e-prop examples index: https://nest-simulator.readthedocs.io/en/stable/auto_examples/eprop_plasticity/index.html
- NEST model docs (formulations quoted above): https://nest-simulator.readthedocs.io/en/stable/models/eprop_synapse_bsshslm_2020.html , .../eprop_iaf_bsshslm_2020.html , .../eprop_iaf_adapt_bsshslm_2020.html
- Bellec, Scherr, Subramoney, Hajek, Salaj, Legenstein, Maass (2020). A solution to the learning dilemma for recurrent networks of spiking neurons. *Nat Commun* 11:3625. https://doi.org/10.1038/s41467-020-17236-y
- Korcsak-Gorzo, Espinoza Valverde, Stapmanns, Plesser, Dahmen, Bolten, van Albada, Diesmann (2025). Event-driven eligibility propagation in large sparse networks: efficiency shaped by biological realism. arXiv:2511.21674. https://doi.org/10.48550/arXiv.2511.21674
- Pronold et al. (2024). Multi-scale spiking network model of human cerebral cortex (HuMAM). *Cereb Cortex* 34(10):bhae409. https://pmc.ncbi.nlm.nih.gov/articles/PMC11491286/
- Lu et al. (2024). Imitating and exploring the human brain's resting and task-performing states via brain computing: scaling and architecture (DTB). *Natl Sci Rev* 11(5):nwae080. https://academic.oup.com/nsr/article/11/5/nwae080/7617698 (full text via PMC11129584)
- DTB multitask-behavior follow-up (hypernetwork/RNN lineage): *BMEF* (2026), doi:10.34133/bmef.0231, https://pmc.ncbi.nlm.nih.gov/articles/PMC12895553/
