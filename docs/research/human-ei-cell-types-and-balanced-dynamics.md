# Human E/I cell families and balanced recurrent dynamics

**Date:** 2026-08-21

**Status:** Primary-source research and a preregistered implementation plan; no model artifact was downloaded and no product code or capability benchmark was changed.

**Code baseline:** `acc2246` after this documentation branch is rebased.
**Scope boundary:** This note concerns the `Human Brain` GLIF5 recurrent path only. It does not concern `Fly Brain`, and it does not inspect hidden simple-SWE or SWE-bench evaluation data.

## Decision

Add a minimal, explicitly typed E/I circuit before adding many cortical subtypes. Keep the existing human spiny GLIF5 fit—Allen neuronal model `523000114`, specimen `508378649`—as the project's excitatory proxy. Add Allen human **aspiny** GLIF5 model `520757748`, specimen `508400236`, as the first inhibitory proxy because it matches the current specimen's donor, cortical area, layer, hemisphere, and disease context. This is a stronger controlled contrast than choosing the fastest-spiking record. It is still a proxy: Allen's human records do not molecularly establish that this aspiny cell is GABAergic, and they do not identify it as PV, SST, or VIP.

Make source cell type determine the sign of every recurrent contact (Dale's-law invariant). Use fixed-indegree sparse `E->E`, `E->I`, `I->E`, and `I->I` blocks, separate excitatory and inhibitory postsynaptic-current traces, and block-shared parameters. Lay the two cell families out in contiguous ranges inside the existing persistent CUDA execution. Do **not** create Python cell objects, per-edge event objects, a dense matrix, or a kernel launch per family.

The inhibitory fraction, block strengths, time constants, delays, and group structure are hypotheses to calibrate on public training/development inputs, not human cortical truths. Add them one at a time with edge-, activity-, data-, and compute-matched controls. A condition is worth carrying forward only if it retains the `>100,000` fully updated-cell and `>100` end-to-end-token/s gates, passes stability/activity gates, increases usable state rank or readout coverage, and improves family-disjoint development prediction. Freeze the winning rule and checkpoint before one held-out evaluation.

## What the current circuit does

At commit `acc2246`, [`AllenGlif5Parameters`](../../src/humanbrain/cells.py) is one pinned human middle-temporal-gyrus (MTG) layer-3 spiny fit. [`RecurrentCellCircuit`](../../src/humanbrain/recurrent.py) instantiates that same intrinsic fit for every reservoir cell. A seeded fraction of cells—currently 0.2 by default—gets negative outgoing recurrent weights, but inhibitory and excitatory cells otherwise have identical membrane, threshold, reset, after-spike-current, and synaptic-current dynamics. One signed recurrent-current array decays with one time constant. Recurrent spikes affect their targets on the following 0.05-ms integration step; there is no biological-scale conduction-delay model.

The optimized CUDA path is a persistent cooperative kernel: one launch advances all 100 steps in the 5-ms token interval while updating every enabled fitted cell and scanning sparse incoming edges. The registered 200,000-cell result is 144.03 end-to-end tokens/s for the reservoir-only full-touch protocol; the current hybrid sparse-output artifact reports 158.18 inference tokens/s and 150.00 online-teaching tokens/s after reducing the eligibility transfer ([simulation protocol and result](../simulation-speed.md), [project README](../../README.md)). These are the relevant engineering baselines. They do not show that the sparsely recruited large circuit has useful language state.

This design has a valid Dale-like sign at runtime, but its cell-type semantics are weak: “inhibitory” is a random outgoing-sign label applied to a pyramidal intrinsic model. It also makes excitation and inhibition indistinguishable after their signed currents have been summed. The proposed change separates those concerns without replacing the working execution architecture.

## Official Allen GLIF5 inventory

### Selection and evidence level

Allen's [official template record](https://api.brain-map.org/api/v2/data/NeuronalModelTemplate/395310498.json) identifies neuronal-model template `395310498` as **“5 LIF-R-ASC + Threshold Adaptation (LIF-R-ASC-A)”**: voltage, reset behavior, two after-spike currents, and spike-threshold adaptation ([Allen GLIF documentation](https://alleninstitute.github.io/AllenSDK/glif_models.html), [Teeter et al. 2018](https://doi.org/10.1038/s41467-017-02717-4)). The model IDs below are the intersection of official Cell Types specimen metadata with:

- organism `Homo sapiens`;
- dendrite type `aspiny`;
- an attached neuronal model using template `395310498` (GLIF5); and
- an attached `NeuronalModelParameters` well-known file.

That query returns six records. The official AllenSDK example explains the database's spiny/aspiny split as *presumably* excitatory/inhibitory, while the Human Cell Types overview says human cells were selected by soma shape and cortical layer rather than the marker-line targeting used for mouse inhibitory classes ([AllenSDK Cell Types notebook](https://alleninstitute.github.io/AllenSDK/_static/examples/nb/cell_types.html), [Allen Human Cell Types overview](https://celltypes.brain-map.org/)). In these human records, line-name fields are blank and reporter status is absent. Therefore this report uses **aspiny inhibitory proxy**, never “molecularly verified GABAergic interneuron.” Morphology can be an imperfect classifier, and these surgical specimens come from epilepsy or tumor cases.

### Candidate models and configuration IDs

| Neuronal model | Specimen | Donor / tissue provenance | Structure | Dendrite / reconstruction | Parameter-file ID |
|---|---:|---|---|---|---:|
| [`543285649`](https://api.brain-map.org/api/v2/data/NeuronalModel/543285649.json?include=neuronal_model_template%2Cwell_known_files%28well_known_file_type%29%2Cspecimen%28donor%28age%2Corganism%29%2Cstructure%2Ccortex_layer%2Cspecimen_tags%29) | [`488418027`](https://api.brain-map.org/api/v2/data/ApiCellTypesSpecimenDetail/query.json?criteria=%5Bspecimen__id%24eq488418027%5D) | H15.06.017, 65 M, epilepsy | MTG L3 right | aspiny; full reconstruction | [`568872458`](https://api.brain-map.org/api/v2/well_known_file_download/568872458) |
| **[`520757748`](https://api.brain-map.org/api/v2/data/NeuronalModel/520757748.json?include=neuronal_model_template%2Cwell_known_files%28well_known_file_type%29%2Cspecimen%28donor%28age%2Corganism%29%2Cstructure%2Ccortex_layer%2Cspecimen_tags%29)** | **[`508400236`](https://api.brain-map.org/api/v2/data/ApiCellTypesSpecimenDetail/query.json?criteria=%5Bspecimen__id%24eq508400236%5D)** | **H16.06.004, 37 M, epilepsy** | **MTG L3 right** | **aspiny; no reconstruction record** | **[`568879989`](https://api.brain-map.org/api/v2/well_known_file_download/568879989)** |
| [`522987536`](https://api.brain-map.org/api/v2/data/NeuronalModel/522987536.json?include=neuronal_model_template%2Cwell_known_files%28well_known_file_type%29%2Cspecimen%28donor%28age%2Corganism%29%2Cstructure%2Ccortex_layer%2Cspecimen_tags%29) | [`520631985`](https://api.brain-map.org/api/v2/data/ApiCellTypesSpecimenDetail/query.json?criteria=%5Bspecimen__id%24eq520631985%5D) | H16.03.002, 41 F, epilepsy | MTG L5 left | aspiny; no reconstruction record | [`568822886`](https://api.brain-map.org/api/v2/well_known_file_download/568822886) |
| [`529910131`](https://api.brain-map.org/api/v2/data/NeuronalModel/529910131.json?include=neuronal_model_template%2Cwell_known_files%28well_known_file_type%29%2Cspecimen%28donor%28age%2Corganism%29%2Cstructure%2Ccortex_layer%2Cspecimen_tags%29) | [`527890930`](https://api.brain-map.org/api/v2/data/ApiCellTypesSpecimenDetail/query.json?criteria=%5Bspecimen__id%24eq527890930%5D) | H16.06.008, 24 F, epilepsy | MTG L3 left | aspiny; no reconstruction record | [`568880518`](https://api.brain-map.org/api/v2/well_known_file_download/568880518) |
| [`543899047`](https://api.brain-map.org/api/v2/data/NeuronalModel/543899047.json?include=neuronal_model_template%2Cwell_known_files%28well_known_file_type%29%2Cspecimen%28donor%28age%2Corganism%29%2Cstructure%2Ccortex_layer%2Cspecimen_tags%29) | [`541536216`](https://api.brain-map.org/api/v2/data/ApiCellTypesSpecimenDetail/query.json?criteria=%5Bspecimen__id%24eq541536216%5D) | H16.03.007, 28 F, epilepsy | MTG L3 left | aspiny; full reconstruction | [`568883599`](https://api.brain-map.org/api/v2/well_known_file_download/568883599) |
| [`591249886`](https://api.brain-map.org/api/v2/data/NeuronalModel/591249886.json?include=neuronal_model_template%2Cwell_known_files%28well_known_file_type%29%2Cspecimen%28donor%28age%2Corganism%29%2Cstructure%2Ccortex_layer%2Cspecimen_tags%29) | [`576084692`](https://api.brain-map.org/api/v2/data/ApiCellTypesSpecimenDetail/query.json?criteria=%5Bspecimen__id%24eq576084692%5D) | H17.06.007, 42 F, tumor | MFG L3 right | aspiny; full reconstruction | [`591282864`](https://api.brain-map.org/api/v2/well_known_file_download/591282864) |

All six parameter files are named `{neuronal_model_id}_neuron_config.json` and are typed `NeuronalModelParameters` in the model record. The links in the final column are direct official download endpoints, but **none was opened or downloaded for this research commit**.

### Comparison with the current spiny fit

The current model is Allen neuronal model [`523000114`](https://api.brain-map.org/api/v2/data/NeuronalModel/523000114.json?include=neuronal_model_template%2Cwell_known_files%28well_known_file_type%29%2Cspecimen%28donor%28age%2Corganism%29%2Cstructure%2Ccortex_layer%2Cspecimen_tags%29), specimen [`508378649`](https://api.brain-map.org/api/v2/data/ApiCellTypesSpecimenDetail/query.json?criteria=%5Bspecimen__id%24eq508378649%5D), parameter-file ID [`568808790`](https://api.brain-map.org/api/v2/well_known_file_download/568808790). It is a spiny, intact-apical, fully reconstructed MTG L3 right cell from donor H16.06.004. Model `520757748` is the only aspiny candidate from that same donor, area, layer, and hemisphere. Using it first reduces donor, region, laminar, side, and clinical-context confounds; it does not make the pair a matched biological E/I recording.

Official specimen-level ephys summaries also show that “aspiny GLIF5” is not one homogeneous fast-spiking type:

| Model / specimen | Adaptation | Mean firing (Hz) | Input resistance (MOhm) | Membrane tau (ms) | Rheobase (pA) | Up/down ratio | Rest (mV) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Current spiny `523000114` / `508378649`** | 0.06272 | 7.867 | 189.219 | 48.683 | 90 | 3.079 | -71.494 |
| Aspiny `543285649` / `488418027` | not reported | 110.742 | 141.250 | 11.017 | 100 | 1.525 | -72.457 |
| **Matched aspiny `520757748` / `508400236`** | **0.01094** | **30.662** | **167.187** | **15.268** | **140** | **1.472** | **-69.530** |
| Aspiny `522987536` / `520631985` | 0.01089 | 25.023 | 202.187 | 17.583 | 70 | 1.368 | -64.424 |
| Aspiny `529910131` / `527890930` | 0.01145 | 16.687 | 146.429 | 12.222 | 90 | 2.368 | -69.435 |
| Aspiny `543899047` / `541536216` | 0.06279 | 3.202 | 113.281 | 10.832 | 90 | 1.416 | -69.142 |
| Aspiny `591249886` / `576084692` | 0.08725 | 167.785 | 118.125 | 8.899 | 110 | 1.221 | -65.311 |

These values come from the linked official specimen-detail records, not from running the GLIF models. The aspiny specimens have much shorter measured membrane time constants than the current spiny specimen, but their firing and adaptation vary widely. That is a reason to test intrinsic heterogeneity, not to hand-pick the highest firing rate.

This table is **not** a config-to-config parameter comparison. The additional configuration files were deliberately not downloaded, so their GLIF coefficients, numerical method flags, and file hashes remain unknown. Before implementation, the chosen config must pass the same field-by-field unit conversion and AllenSDK trace-parity work already required for a faithful port.

### Reproducible pinning procedure for a later implementation commit

IDs alone do not guarantee immutable bytes. In the implementation PR—but not this report—download each selected file once from its official well-known-file endpoint, keep the raw response bytes unchanged, and record a manifest containing:

1. neuronal-model ID, specimen ID, neuronal-model-template ID `395310498`, well-known-file ID, and official filename;
2. the exact API metadata URLs above and the final resolved artifact URL;
3. UTC retrieval time, byte length, media type, ETag/Last-Modified when supplied, and SHA-256 of the raw config bytes;
4. a SHA-256 of the saved model/specimen metadata JSON or a canonical snapshot of the fields used here;
5. the Allen terms/citation-policy URLs and the date they were reviewed; and
6. the ported constants plus a machine-checkable mapping from every JSON field and unit to the runtime representation.

A future maintainer can perform the byte pin with an explicit destination outside the source tree, then copy only an approved artifact into the project:

```powershell
$wellKnownFileId = 568879989
$destination = "<explicit-staging-directory>\520757748_neuron_config.json"
curl.exe --fail --location "https://api.brain-map.org/api/v2/well_known_file_download/$wellKnownFileId" --output $destination
$hash = (Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash.ToLowerInvariant()
$bytes = (Get-Item -LiteralPath $destination).Length
"sha256=$hash bytes=$bytes"
```

The implementation must reject a hash mismatch rather than silently accepting changed remote content. It should preserve the upstream filename and license/provenance manifest. This report intentionally provides the procedure and IDs without creating an unreviewed data redistribution.

### Access, citation, and license caveats

Allen's [Terms of Use](https://alleninstitute.org/legal/terms-of-use) currently distinguish research/noncommercial use and require permission for some commercial or redistribution uses; the [Citation Policy](https://alleninstitute.org/legal/citation-policy) asks users to cite the relevant dataset/direct URL and associated primary publication. The AllenSDK source has its own license and maintenance status, which should not be assumed to license downloaded Cell Types data. Before committing or distributing any config, re-read the current terms, preserve direct record provenance, cite [Teeter et al.](https://doi.org/10.1038/s41467-017-02717-4), and obtain permission if the intended use exceeds the published terms. This note is not legal advice.

## Established circuit patterns and what they justify

### Balance and inhibitory stabilization

Large sparse E/I networks can enter an asynchronous, irregular regime when strong excitation and inhibition approximately cancel, with fluctuations remaining dynamically important ([van Vreeswijk and Sompolinsky 1996](https://doi.org/10.1126/science.274.5293.1724)). Brunel's sparse integrate-and-fire analysis maps asynchronous irregular, synchronous regular, synchronous irregular, and fast-oscillatory regimes as recurrent balance, delay, and external drive change ([Brunel 2000](https://doi.org/10.1023/A:1008925309027)). These results justify measuring a regime instead of assuming that more cells automatically make it healthy. Their networks are theoretical models, not fitted human GLIF5 language circuits, so their ratios and weights are not parameters to copy literally.

An inhibition-stabilized network (ISN) is a more specific hypothesis: recurrent excitation would be unstable by itself and feedback inhibition stabilizes it. The counterintuitive diagnostic is that a sustained positive drive to inhibitory cells can reduce their steady firing after the network re-equilibrates ([Tsodyks et al. 1997](https://doi.org/10.1523/JNEUROSCI.17-11-04382.1997)). Ozeki et al. reported this inhibitory-stabilization signature in cat V1 surround modulation ([Ozeki et al. 2009](https://doi.org/10.1016/j.neuron.2009.03.028)). It is a useful frozen perturbation probe, not a universal acceptance criterion: a stable, computationally useful circuit need not sit in an ISN regime.

For this project, “balance” should mean separately observable `I_E` and `I_I` contributions that are both nonzero and partially cancel over a public probe, while total drive remains responsive. It should not mean adjusting two large currents until a chosen firing-rate number appears. A slow, local postsynaptic homeostatic gain can be tested after the static typed circuit works, but it must be frozen before evaluation and compared with a no-homeostasis control.

### Cell-type blocks instead of a monolithic random graph

The Potjans-Diesmann cortical microcircuit uses eight layer/type populations—E and I in layers 2/3, 4, 5, and 6—with sparse block-specific connection probabilities, weights, and delays. Its approximately 77,000 neurons and 0.3 billion synapses produce population-specific activity in a target-owned connectivity model ([Potjans and Diesmann 2014](https://doi.org/10.1093/cercor/bhs358), [official reference implementation](https://github.com/INM-6/microcircuit-PD14-model), [official model documentation](https://microcircuit-pd14-model.readthedocs.io/en/stable/index.html)). It establishes the storage and experimental pattern—small shared block tables plus sparse contacts—not that this project's first implementation needs eight populations or those exact probabilities.

Human synaptic physiology supports testing cell-class-dependent dynamics while warning against brittle labels. Campagnola et al. found overlapping but subclass-linked variation in connection strength, variability, and short-term dynamics in mouse and human cortex, including depth variation in human layer-2/3 E-E properties ([Campagnola et al. 2022](https://doi.org/10.1126/science.abj5861)). The six Allen aspiny fits here do not map cleanly onto those transcriptomic subclasses. Use the evidence to motivate heterogeneity and shared blocks, not to invent PV/SST/VIP identities.

Heterogeneous neuronal time constants have improved task learning and robustness in trained spiking networks ([Perez-Nieves et al. 2021](https://doi.org/10.1038/s41467-021-26022-3)), and biological cortical signals exhibit a broad distribution of memory timescales ([Bernacchia, Seo, Lee, and Wang 2011](https://doi.org/10.1038/nn.2752)). Those results motivate a frozen heterogeneity ablation. They do not show that importing all six fits will help this local-learning language model.

### Sparse assemblies, groups, and delays

A modest increase in within-assembly E-E connection probability can create slow, metastable activity in an otherwise balanced network; a surprisingly small structural change can materially alter dynamics ([Litwin-Kumar and Doiron 2012](https://doi.org/10.1038/nn.3220)). This makes grouped topology plausible, but it also makes aggressive grouping dangerous: a few dominant assemblies can reduce coverage and effective rank. First test one modest single-scale E-group quota with the same edge count and weight budget as uniform connectivity. Test nested or spatial groups only after that main effect is known. Keep inhibition broad across groups in the initial grouped condition so it can stabilize rather than merely isolate assemblies.

Conduction delays add temporal state without adding neurons. Delay diversity combined with spike-timing-dependent plasticity can form reproducible nonsynchronous spike groups ([Izhikevich 2006](https://doi.org/10.1162/089976606775093882)). The Potjans-Diesmann family uses longer delays for excitatory than inhibitory sources; an independent implementation reports means of 1.5 ms for excitatory and 0.75 ms for inhibitory projections ([van Albada et al. 2020 implementation analysis](https://pmc.ncbi.nlm.nih.gov/articles/PMC6939236/)). These values are starting hypotheses only. At this project's 0.05-ms step they correspond to 30 and 15 history bins, so a source-type-shared delay can be implemented cheaply and tested before per-edge delay distributions.

The liquid-state-machine result provides the computational test: a recurrent spiking system is useful when different recent inputs produce separable high-dimensional transient states that a memoryless readout can exploit ([Maass, Natschlaeger, and Markram 2002](https://doi.org/10.1162/089976602760407955)). It does not say that raw neuron count, firing, or covariance rank alone constitutes language capability. This project must pair state-rank/separation diagnostics with family-disjoint prediction.

### Established simulator storage patterns

NEST distinguishes connection-local fields from homogeneous parameters shared by a synapse model. Its `static_synapse_hom_w` stores one common weight while each connection retains only target, delay, and receptor information; receptor-specific point cells expose separate synaptic time constants ([NEST synapse specification](https://nest-simulator.readthedocs.io/en/stable/synapses/synapse_specification.html), [`static_synapse_hom_w`](https://nest-simulator.readthedocs.io/en/stable/models/static_synapse_hom_w.html), [`iaf_psc_exp_multisynapse`](https://nest-simulator.readthedocs.io/en/stable/models/iaf_psc_exp_multisynapse.html)). Arbor recipes similarly expose connections owned by each target cell, while runtime communication turns spikes into weight-and-delay events in sorted queues ([Arbor recipe](https://docs.arbor-sim.org/en/latest/cpp/recipe.html), [Arbor spike communication](https://docs.arbor-sim.org/en/latest/dev/communication.html), [Arbor Brunel tutorial](https://docs.arbor-sim.org/en/latest/tutorial/brunel.html)).

The relevant convention is to share immutable type/block parameters and keep only irreducible sparse state per contact. It is not a reason to import a general simulator or its object model into the CUDA hot path.

## Minimal testable architecture for this codebase

### Stage 1: two intrinsic families with hard Dale signs

Use two contiguous cell ranges:

```text
[  spiny-E model 523000114  ][  aspiny-I proxy model 520757748  ]
0                           E_end                                  N
```

The type is derived from `gid < E_end`, so the kernel needs no per-cell type byte for this first experiment. Replace the random inhibitory mask with an immutable source-type sign: every E-source recurrent weight is nonnegative, every I-source recurrent weight is nonpositive, and learning can change magnitude but cannot cross zero. Output-cell races continue to receive nonnegative reservoir contacts unless a separately specified inhibitory output circuit is under test.

Start from the existing 80/20 allocation **only as a continuity hypothesis**, then preregister 90/10, 80/20, and 70/30 on training/development probes. Preserve total cells, incoming degree, total edges, initial absolute recurrent-current budget, input fanout, readout contacts, training tokens, and seeds. Do not call any winning fraction a human cortical estimate.

Store a two-element `GlifConfig` table in constant or read-only device memory. Each integration lane selects the table entry from its contiguous range and runs the same numerical rule. The CPU scalar oracle must do the identical selection. A mixed-family exactness test must cover reset, refractory behavior, both after-spike currents, threshold adaptation, and type-boundary cells.

### Stage 2: typed sparse blocks and separate E/I current traces

Build target-owned incoming CSR with a fixed indegree per block:

```text
K[target_type][source_type]    sparse contact counts
J[target_type][source_type]    shared initial magnitudes
tau[target_type][receptor]     postsynaptic E/I current decay
```

Each target gets exactly the frozen `K_TE`, `K_TI`, `K_IE`, and `K_II` counts selected by its type; source IDs are sampled deterministically inside the appropriate contiguous range. The initial `K` matrix should preserve the current total degree and edge count. Separate `I_E` and `I_I` arrays decay with receptor/block-shared time constants and sum only at membrane integration. This makes cancellation and saturation observable and prevents an excitatory spike from inheriting inhibitory kinetics merely because both used one signed scalar.

For learned recurrent variation, retain a sparse magnitude overlay only on contacts that actually learn. Do not allocate a dense `N x N` matrix. The immutable sign remains derived from source type. The current reverse/incoming CSR can remain the physical edge layout, so topology upload and the persistent cooperative kernel remain recognizable rather than being replaced.

### Stage 3: source-type-shared delays

Add one E delay and one I delay before testing distributions. A bit- or byte-packed spike-history ring of `max_delay_steps x N` permits the recurrent scan to read the appropriate historical source spike. With 30 bins at the 1.5-ms starting hypothesis, a byte ring costs about 5.72 MiB at 200,000 cells; bit packing would cost about 0.72 MiB but may trade arithmetic for bandwidth. Measure both if the byte ring threatens throughput. Do not create a heap event per spike or store a delay on every edge in the first version.

Use a constant block delay only if the source-type scheme helps. A later four-code delay table can be stored in two bits per edge; continuous per-edge delays are not justified until a development result pays for their memory and branch cost.

### Stage 4: heterogeneity and groups, separately

After the two-family circuit passes, compare the matched inhibitory proxy with a preregistered multi-fit condition. Assign the six aspiny fits by a deterministic, balanced hash of stable cell ID; never choose a fit from validation performance. Keep each family contiguous inside a tile if type-major global layout makes topology locality poor. A small `GlifConfig[T]` table and tile/type offsets are sufficient; no virtual dispatch or Python timestep loop is allowed.

Separately compare uniform connectivity with a single-scale grouped condition. Keep E and I fractions, total per-block indegrees, weights, delays, and edge count fixed. Move only a frozen fraction of E-E contacts from between-group to within-group targets. Derive group ID procedurally from cell/tile ID or store a compact integer; use broad I projections across groups. Nested groups, 2D/layered/3D geometry, and mixed membership remain later independent ablations, consistent with the project's existing result that no one geometry wins at every size.

### Hot-path and memory invariants

| State | Storage rule |
|---|---|
| GLIF membrane, reset/threshold state, two ASCs, refractory, eligibility | Existing structure-of-arrays; one value per fully updated cell |
| Cell family | Implicit contiguous range at `T=2`; compact type/tile metadata only when `T>2` |
| E and I synaptic current | Two contiguous arrays per cell; no receptor objects |
| Recurrent contacts | Existing target-owned sparse CSR; source ID per edge; sign derived from source type |
| Initial weights, current time constants, delays | Small type/block tables shared by all matching cells/contacts |
| Learned recurrent change | Sparse magnitude overlay; never dense population-squared state |
| Delay state | Global spike-history ring; no per-spike heap objects |
| Groups | Procedural/tiled IDs and fixed quotas; no population-pair adjacency matrix |

Keep one persistent kernel across families and biological steps. The two current arrays add one extra `f64` per cell relative to the present signed-current representation—about 1.53 MiB at 200,000 cells—before delay history and alignment. Report actual host and device bytes rather than relying on this estimate. If type branches diverge badly, sort work by type-contiguous ranges or fixed tiles; do not duplicate the entire kernel launch.

## Preregistered train/development experiment

Write a versioned JSON protocol and hash it before running any new capability bundle. All topology, parameter, seed, metric, and selection choices below are train/development-only. Do not inspect hidden simple-SWE/SWE-bench tasks, generated hidden test bundles, or the locked evaluation split while choosing them.

### Ordered ablations

| Order | Condition | Single question isolated |
|---:|---|---|
| 0 | Current one-fit circuit with 20% negative-source labels | Locked baseline |
| 1a | Typed E/I topology, both types still use spiny fit | Does hard typing/block construction help independent of intrinsic heterogeneity? |
| 1b | Matched spiny-E + aspiny-I fit, current signed-current dynamics | Does the matched intrinsic contrast help independent of receptor separation? |
| 2 | Winning type setup + separate E/I current traces and 2x2 block table | Do explicit E/I kinetics and balance help? |
| 3 | Stage 2 + source-type delays; compare zero, equal, and frozen E/I delays | Does temporal state help? |
| 4 | Matched I fit versus deterministic six-fit I mixture | Does within-I intrinsic heterogeneity help? |
| 5 | Uniform versus modest single-scale E assembly quota | Do groups help at matched degree and weight budget? |
| 6 | Only after main effects: frozen combination; optional nested/layered/2D/3D controls | Are gains additive and scale-stable? |

Do not silently combine a cell family, new current dynamics, delays, and columns into one “biological” condition. Run at least five paired seeds per main comparison. Predeclare one small calibration size, one development size, and the 200,000-cell engineering size. Treat cell count as an independent axis: compare useful **updated and responsive** cells, not dormant padding, at matched data and per-cell degree.

### Required controls

- **Label shuffle:** preserve family counts and topology but permute GLIF-family assignments. This tests the fitted family contrast against an extra branch or random partition.
- **All-spiny typed control:** preserve E/I blocks and signs while both ranges use model `523000114`.
- **Dale violation diagnostic only:** the old random-sign arrangement remains a baseline, not a candidate allowed to masquerade as typed biology.
- **Layout permutation:** remap stable GIDs and parameter-block order while preserving the graph; outputs must remap exactly. This catches accidental cell-order effects.
- **Degree/weight control:** keep per-target degree, total edge count, and initial absolute current budget fixed when changing groups or type proportions.
- **Delay cost control:** compare delay history with a zero-delay ring that executes the same memory path, plus the current no-ring fast path.
- **Growth/order control:** if developmental recruitment is later combined, compare random, type-stratified, and spatial birth order with the final graph, exposures, active-cell count, and training tokens matched. Natural-looking order is not evidence without this control.
- **No dormant credit:** every cell counted in a scale claim is integrated and reachable; report materialized, updated, spiking, responsive, and readout-contacted counts separately.

### Dynamics and activity gates

Measure these on a frozen public probe set before capability scoring:

- per-type firing rate, inter-spike-interval CV where estimable, refractory occupancy, and spike-count Fano factor;
- `I_E`, `I_I`, and net current distributions; the cancellation index `|I_E + I_I| / (|I_E| + |I_I| + epsilon)`;
- responsive-state fraction, mean and union active-eligibility fractions, sampled pairwise correlation, and participation-ratio effective rank `r_eff = tr(C)^2 / tr(C^2)`;
- prompt-pair state separation/collision rate, responsive cells reachable by each output candidate, and fraction of readout contacts receiving nonconstant presynaptic state;
- per-family and per-group participation entropy so one fit or assembly cannot own nearly all activity; and
- transient recovery after a fixed input pulse, plus the optional frozen ISN perturbation diagnostic.

The exact numerical ranges must be fixed in the protocol in two steps: first run **only** the locked baseline on the public probes and record its paired-seed quantiles; then hash those baseline artifacts and freeze all noninferiority/multiplicative thresholds before running any candidate. This avoids inventing biological firing bands for token-driven human GLIF fits while preventing post-hoc gates. At minimum, a candidate must:

1. produce no NaN/Inf, stuck-refractory population, all-silent type, or self-sustained post-input saturation;
2. retain at least half the locked baseline responsive-state and readout-source fractions, matching the project's existing density guard, and keep median firing within the existing 0.5-80 Hz development guard;
3. keep peak firing/refractory occupancy and recovery time inside the frozen baseline-derived safety envelope;
4. have nonzero E and I current in the typed condition and avoid either no cancellation or near-perfect cancellation with vanishing responsiveness;
5. increase median effective state rank, prompt separation, or readout coverage by the preregistered meaningful-effect threshold—not merely by floating-point noise—without reducing family-disjoint development accuracy; and
6. report the ISN perturbation result as a regime descriptor, not tune toward it.

Rank and activity are diagnostic gates, not optimization targets in isolation. A high-rank noise source that does not improve development prediction fails.

### Speed and exactness gates

For the full 200,000-cell condition, preserve the frozen 0.05-ms step, 5-ms token interval, all 100 biological steps, sparse recurrent propagation, output-cell computation, and all fitted state. Require both:

- more than 100 end-to-end tokens/s for inference and online teaching on the registered workstation; and
- at least 90% of the paired locked-baseline throughput for the same executable, seed, activity trace, and readout path.

Report kernel-only time, end-to-end time, topology-build/upload time, host/device bytes, spikes, responsive cells, and all transfer bytes. Run an activity-matched synthetic trace as well as natural-token input so sparse recruitment cannot hide a slow dense regime. No Python biological-step loop, dormant padding, reduced step count, or unreported CPU fallback qualifies.

Before timing, require exact scalar-CPU/native/CUDA parity for both families and for E/I current, delay-ring, reset, checkpoint continuation, and boundary GIDs. If the official configs use arithmetic rules not represented by the present kernel, implement and verify those rules rather than silently coercing the fit into the current model.

### Capability and anti-overfitting gate

Use only training and family-disjoint development tasks for selection. Keep training examples, token budget, readout capacity (`K` and `C`), structural-update count, and wall-clock or cell-step compute explicitly matched. Report both equal-cell and equal-compute tracks if a candidate is slower. Publish every seed and failed condition.

Select by a frozen lexicographic rule:

1. exactness, Dale, stability/activity, memory, and speed gates all pass;
2. median family-disjoint development prediction improves over the paired baseline, with uncertainty across seeds reported;
3. no task family has a material preregistered regression;
4. among statistically indistinguishable survivors, choose the simpler/faster condition; and
5. use state rank, separation, and coverage only as the final mechanistic tie-breakers.

Then freeze source commit, protocol hash, config hashes, checkpoint hash, seed list, tokenizer, decoder, and stopping rule before a single held-out run. Do not choose the cell ratio, model mixture, delay, group scale, or homeostasis strength from held-out results. If dynamics improve but family-disjoint prediction does not, report the dynamics result and do not promote it as greater intelligence. If more cells help only when data or compute also increases, call that a joint scaling result rather than a neuron-count law.

## Implementation order and stop conditions

1. **Provenance PR:** retrieve only model `520757748` after license review; pin metadata and raw config hashes; add field/unit validation and an AllenSDK trace fixture. Stop if the official config cannot be redistributed under the intended use.
2. **Two-family oracle:** add the matched GLIF table and hard Dale signs to the scalar reference with tiny deterministic tests. Stop if the current numerical contract cannot faithfully express the new config.
3. **Native/CUDA parity:** contiguous type ranges, one persistent kernel, no topology redesign yet. Stop if 200,000-cell throughput falls below either speed gate.
4. **Typed blocks and E/I receptor state:** target-owned fixed indegrees and two current arrays, with current-budget-matched controls.
5. **One temporal mechanism:** source-type delays and its zero-delay cost control.
6. **One structural mechanism:** modest E assemblies, separately from the delay and heterogeneity experiments.
7. **Only after positive main effects:** six-fit heterogeneity, nested groups, laminar blocks, developmental ordering, or combinations.

This layered order keeps a working product after every step. It also gives a negative result a clean interpretation: intrinsic family, receptor separation, delay, grouping, and growth order can fail independently rather than disappearing inside one large architecture change.

## Sources

- Allen Institute, [Generalized leaky integrate-and-fire models](https://alleninstitute.github.io/AllenSDK/glif_models.html), [Cell Types API documentation](https://alleninstitute.github.io/AllenSDK/cell_types.html), [Human Cell Types overview](https://celltypes.brain-map.org/), [Terms of Use](https://alleninstitute.org/legal/terms-of-use), and [Citation Policy](https://alleninstitute.org/legal/citation-policy).
- Teeter et al. (2018), [Generalized leaky integrate-and-fire models classify multiple neuron types](https://doi.org/10.1038/s41467-017-02717-4).
- van Vreeswijk and Sompolinsky (1996), [Chaos in neuronal networks with balanced excitatory and inhibitory activity](https://doi.org/10.1126/science.274.5293.1724).
- Brunel (2000), [Dynamics of sparsely connected networks of excitatory and inhibitory spiking neurons](https://doi.org/10.1023/A:1008925309027).
- Tsodyks et al. (1997), [Paradoxical effects of external modulation of inhibitory interneurons](https://doi.org/10.1523/JNEUROSCI.17-11-04382.1997).
- Ozeki et al. (2009), [Inhibitory stabilization of the cortical network underlies visual surround suppression](https://doi.org/10.1016/j.neuron.2009.03.028).
- Potjans and Diesmann (2014), [The cell-type specific cortical microcircuit](https://doi.org/10.1093/cercor/bhs358).
- Campagnola et al. (2022), [Local connectivity and synaptic dynamics in mouse and human neocortex](https://doi.org/10.1126/science.abj5861).
- Litwin-Kumar and Doiron (2012), [Slow dynamics and high variability in balanced cortical networks with clustered connections](https://doi.org/10.1038/nn.3220).
- Izhikevich (2006), [Polychronization: computation with spikes](https://doi.org/10.1162/089976606775093882).
- Maass, Natschlaeger, and Markram (2002), [Real-time computing without stable states](https://doi.org/10.1162/089976602760407955).
- Perez-Nieves et al. (2021), [Neural heterogeneity promotes robust learning](https://doi.org/10.1038/s41467-021-26022-3).
- Bernacchia, Seo, Lee, and Wang (2011), [A reservoir of time constants for memory traces in cortical neurons](https://doi.org/10.1038/nn.2752).
- NEST Simulator, [Synapse specification](https://nest-simulator.readthedocs.io/en/stable/synapses/synapse_specification.html); Arbor, [Recipes and cell connections](https://docs.arbor-sim.org/en/latest/cpp/recipe.html).
