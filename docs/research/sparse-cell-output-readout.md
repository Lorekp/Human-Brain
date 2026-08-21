# Sparse cell-only output readout: plasticity and locality review

**Date:** 2026-08-21  
**Scope:** Evidence for a vocabulary-sized population of simulated output cells with a fixed sparse afferent budget. This note evaluates `C = min(N, 4K)` potential inputs per output cell, `K` allocated contacts, three-factor weight updates, and replacement of a weak contact by the strongest eligible inactive candidate. It does not claim that any cited circuit is specifically human; the cited mechanisms are mammalian or normative spiking-neuron results.

## Bottom line

Keep the fixed candidate pool and fixed-size storage layout, but describe them as a **geometry-limited pool of potential appositions** and a **capacity of K contact slots**, not as exactly K continuously existing biological synapses. Keep eligibility-times-modulator learning, but make eligibility genuinely pre/post-synaptic. Replace instantaneous margin-triggered top-1 rewiring with slow, seeded stochastic turnover (and retain top-1 only as an explicitly engineering-oriented ablation). Use a shared inhibitory circuit for output competition. These changes keep memory `O(V(C+K)) = O(VK)` because `C <= 4K`; the inference hot path still visits only active slots.

| Proposed part | Evidence verdict | Required interpretation or change |
|---|---|---|
| Fixed pool `C` | Supported in form, not in constant | Fixed potential-contact pools constrained by axon/dendrite appositions have precedent. `4K` and a deterministic ID hash are engineering choices; freeze the pool before labels and derive membership from seeded geometry/layer/type. |
| Exactly `K` instantiated silent/weighted contacts | Budget is defensible; literal anatomy is not | Treat `K` as allocated capacity and permit `0..K` active contacts plus vacant/inactive slots. If kernels require K records, zero entries are storage slots, not claimed synapses. |
| Eligibility x target modulator | Partly supported | A three-factor rule is defensible only when a contact's trace contains presynaptic activity **and** a postsynaptic voltage/spike/calcium gate. The target teaching signal is cell-level/nonlocal supervision and must be modeled explicitly. |
| Weakest active contact removal | Approximate support | Weak contacts disappearing after their weight reaches zero is supported. A cell-wide exact argmin is an efficient resource-competition abstraction, not a demonstrated synaptic rule. Prefer a local low-weight/low-use deletion hazard with a minimum contact age. |
| Strongest eligible absent contact | Engineering abstraction | It is closest to RigL's lowest-magnitude drop/highest-gradient grow rule. A nonexistent synapse cannot possess a synapse-local eligibility trace. Store a coactivity trace on a persistent potential apposition, or reactivate candidates stochastically. |
| Rewire on every margin violation | Unsupported timescale/coupling | Functional plasticity can respond to a teaching event; structural turnover should run on a slower clock and be gated by sustained activity/homeostatic state. |
| Strongest non-target/margin | Nonlocal | Numeric argmax and margin comparison expose all output scores. Make the error event “wrong cell won the spike race” or “target missed its deadline”; deliver target teaching current and a broadcast surprise/reward signal. |

## Evidence and locality limits

### Three-factor traces and e-prop

Izhikevich's dopamine-modulated STDP model retains a synapse-specific trace after pre/post coincidence and converts it into plasticity when delayed dopamine arrives ([DOI 10.1093/cercor/bhl152](https://doi.org/10.1093/cercor/bhl152)). Yagishita et al. experimentally found a short dopamine window after glutamatergic/spiking stimulation for spine enlargement ([DOI 10.1126/science.1255514](https://doi.org/10.1126/science.1255514)). These support a local eligibility tag plus a later third factor; they do not support a vocabulary label or exact multiclass margin being locally available at every contact.

E-prop factorizes updates into a synaptic eligibility trace and a neuron-specific learning signal. For LIF cells, the trace is a filtered presynaptic train multiplied by a postsynaptic pseudo-derivative; its ideal learning signal is unavailable online, so e-prop substitutes a current output-error term routed through feedback weights ([Bellec et al., DOI 10.1038/s41467-020-17236-y](https://doi.org/10.1038/s41467-020-17236-y)). Thus e-prop removes backward-through-time transport, not the need to construct and broadcast an approximate error. The [official NEST e-prop implementation](https://nest-simulator.readthedocs.io/en/stable/auto_examples/eprop_plasticity/index.html) makes the readout, learning-signal connection, neuron, and plastic synapse distinct objects.

Bellec et al. derive and test LIF/ALIF cells, not a human-fitted multicompartment cell. Applying the pattern to such a cell is a new engineering hypothesis: its local eligibility recurrence needs an explicit derivation and finite-difference validation on a tiny circuit before it can inherit even e-prop's limited locality claim.

Consequences for this readout:

- A presynaptic reservoir eligibility copied unchanged to every output contact is not a complete neo-Hebbian trace. Use `e_ji <- decay(e_ji) + pre_trace_i * post_gate_j`, where `post_gate_j` is derived from local membrane voltage, spike, or calcium state.
- Use `delta w_ji = eta * M_j * e_ji`, with bounded, nonnegative excitatory weights. `M_j` may be produced by a target-teaching cell/apical current and a broadcast surprise/reward cell, but this is explicit supervisory circuitry, not synapse-local information.
- Do not normalize an update with a norm over all reservoir traces if claiming strict locality. Use a fixed/saturating learning rate or a postsynaptic homeostatic gain.

### Potential contacts, turnover, and sparse rewiring

Spiess et al. model each neuron pair with a fixed number of potential contacts based on anatomical appositions. Active contacts use local pre/post spike traces; a weight reaching zero makes a contact inactive; inactive potential contacts reactivate spontaneously; heterosynaptic terms stabilize firing and total weight ([DOI 10.1093/cercor/bhx339](https://doi.org/10.1093/cercor/bhx339)). This is the closest primary precedent for `C`: persistent potential sites can remain inactive while preserving an anatomical opportunity. It does **not** select the inactive site with the largest task eligibility, nor hold the active count exactly constant.

The NEST structural-plasticity model grows and removes axonal/dendritic “synaptic elements” according to a cell's activity relative to a homeostatic target, then matches vacant compatible elements. Matching requires global knowledge in that implementation and is deliberately much slower than electrical dynamics ([Diaz-Pier et al., DOI 10.3389/fnana.2016.00057](https://doi.org/10.3389/fnana.2016.00057); [official NEST example](https://nest-simulator.readthedocs.io/en/main/auto_examples/structural_plasticity.html)). Local element growth is therefore biologically motivated; exact global matching is an implementation limitation, not proof of locality.

The proposed weakest-to-strongest operation is algorithmically close to RigL: infrequently drop smallest-magnitude live weights and grow absent weights with largest instantaneous gradients while preserving a fixed parameter count ([Evci et al., ICML 2020](https://proceedings.mlr.press/v119/evci20a.html); [official code](https://github.com/google-research/rigl)). DEEP R likewise demonstrates that strict connection budgets and stochastic rewiring are useful engineering constraints ([ICLR 2018 paper](https://openreview.net/forum?id=BJ_wN01C-); [official code](https://github.com/guillaumeBellec/deep_rewiring)). Neither establishes biological locality.

### Stabilization and output competition

Synapse-specific positive feedback needs slower stabilizers. Neocortical cultures show cell-wide, approximately multiplicative scaling of excitatory quantal amplitudes after prolonged activity perturbation ([Turrigiano et al., DOI 10.1038/36103](https://doi.org/10.1038/36103)). Inhibitory plasticity can balance excitation and inhibition and stabilize sparse asynchronous activity in spiking-network models ([Vogels et al., DOI 10.1126/science.1211095](https://doi.org/10.1126/science.1211095)). Use fast intrinsic threshold adaptation/shared inhibition plus slower cell-wide weight scaling and slower still structural turnover; do not collapse all three timescales into an error-triggered swap.

Biologically grounded output competition can be implemented with output excitatory cells recruiting a common inhibitory interneuron, so the first/strongest cell suppresses the rest ([Coultrip et al., DOI 10.1016/S0893-6080(05)80006-1](https://doi.org/10.1016/S0893-6080(05)80006-1)). Integrate-and-fire analysis shows a first spike can complete a winner-take-all computation under suitable excitation/inhibition ([Jin and Seung, DOI 10.1103/PhysRevE.65.051922](https://doi.org/10.1103/PhysRevE.65.051922)). A shared inhibitory population costs `O(V)` connections, not `O(V^2)` lateral edges.

## Recommended bounded rule

For each output cell `j`:

1. Freeze a seeded, geometry/type-derived candidate list `P_j`, `|P_j| = C <= 4K`, before labels are observed. Store a small coactivity state for every potential apposition. This costs at most `4VK` candidate records. Require `K <= N` if presynaptic partners must be unique; otherwise model and report multapses (multiple contacts from one partner) explicitly.
2. Store `K` active-capacity slots separately. A slot can be vacant. Inference traverses only occupied slots and therefore remains `O(VK)`.
3. On neural events, update contact eligibility locally: `e_ji <- lambda_e e_ji + pre_trace_i * post_gate_j`. For an inactive potential site call the analogous state “apposition coactivity,” not synaptic eligibility.
4. On a teaching event, update only existing contacts with `eta * M_j * e_ji`; clamp excitatory contacts at zero and implement negative competition through inhibitory cells or depression toward zero, preserving Dale's law.
5. Let shared inhibition produce a spike-race winner. A wrong winner or target timeout emits a cell-level teaching/surprise event. Do not compute “strongest non-target” inside the plasticity rule.
6. Every `T_struct` training events, not per violation, consider turnover. Delete only contacts that are old enough and persistently weak/unused, using a seeded stochastic hazard. Fill vacancies by seeded stochastic activation among compatible candidates, optionally biased by local apposition coactivity and the output cell's homeostatic resource signal. One-for-one replacement may remain as a fixed-resource abstraction.
7. Keep deterministic top-eligible regrowth as an **engineering lane** and compare it with spontaneous/geometry-matched and coactivity-biased stochastic lanes. Freeze structure before validation and benchmark evaluation.

Only structural epochs scan `C`; choosing `T_struct` makes that work amortized. Counter-based seeded randomness preserves reproducibility without sorting candidates. The hot inference loop and deployable model still hold no dense `V x N` matrix.

## Required tests and preregistered ablations

- **Locality:** a contact update may read only its own pre trace, its output cell's post state/modulator, and its weight. Perturbing an unrelated output cell has no effect except through explicit inhibitory spikes. Assert no dense score/error vector, global norm, or plasticity-side argmax.
- **Eligibility timing:** pre/post coincidence plus an in-window modulator changes weight; pre alone, modulator alone, or an out-of-window modulator does not.
- **Human-cell trace validation:** on one fitted cell and then a tiny circuit, compare the proposed local trace with finite differences under reset, adaptation, and dendritic state; quantify the approximation rather than labeling it e-prop by analogy.
- **Candidate semantics:** inactive scores arise only from stored potential-apposition variables. Pool membership is label-independent and immutable after initialization.
- **Biophysical signs:** excitatory weights never become negative; inhibitory effects originate only from inhibitory cells.
- **Competition:** measure winner count, co-winner rate, first-spike latency, and close-drive behavior; removing shared inhibition must increase co-winners.
- **Homeostasis:** under sustained high and low input, firing rate, total incoming excitatory weight, and contact count remain bounded. Removing scaling/inhibition should produce the predicted instability or quiescence.
- **Turnover:** active contacts never exceed K; deletion respects minimum age; churn is bounded; survival increases with weight/use; no validation/test examples influence rewiring.
- **Complexity/native parity:** account for candidate IDs and traces; verify memory stays a constant times `VK`, inference visits at most K contacts, structural scans allocate nothing per example, and reference/native state transitions match for a fixed seed.
- **Frozen ablations:** no rewiring; random geometry-matched reactivation; low-weight deletion only; coactivity-biased stochastic reactivation; top-eligible/RigL-like reactivation; candidate ratios `C/K` of 2, 4, and 8; immediate versus slow structural clock. Tune only on training/development data.

## Primary sources and official implementations

- Bellec et al. (2020), e-prop: [paper](https://doi.org/10.1038/s41467-020-17236-y), [original code](https://github.com/IGITUGraz/eligibility_propagation), [NEST implementation](https://nest-simulator.readthedocs.io/en/stable/auto_examples/eprop_plasticity/index.html).
- Izhikevich (2007), dopamine-modulated STDP: [paper](https://doi.org/10.1093/cercor/bhl152).
- Yagishita et al. (2014), dopamine timing and spine plasticity: [paper](https://doi.org/10.1126/science.1255514).
- Spiess et al. (2018), multicontact structural plasticity: [paper](https://doi.org/10.1093/cercor/bhx339).
- Diaz-Pier et al. (2016), NEST structural plasticity: [paper](https://doi.org/10.3389/fnana.2016.00057), [official example](https://nest-simulator.readthedocs.io/en/main/auto_examples/structural_plasticity.html).
- Turrigiano et al. (1998), synaptic scaling: [paper](https://doi.org/10.1038/36103).
- Vogels et al. (2011), inhibitory plasticity: [paper](https://doi.org/10.1126/science.1211095).
- Coultrip et al. (1992) and Jin & Seung (2002), spiking WTA: [paper](https://doi.org/10.1016/S0893-6080(05)80006-1), [paper](https://doi.org/10.1103/PhysRevE.65.051922).
- Evci et al. (2020), RigL: [paper](https://proceedings.mlr.press/v119/evci20a.html), [official code](https://github.com/google-research/rigl).
- Bellec et al. (2018), DEEP R: [paper](https://openreview.net/forum?id=BJ_wN01C-), [official code](https://github.com/guillaumeBellec/deep_rewiring).
