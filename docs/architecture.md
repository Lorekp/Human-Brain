# Human Brain architecture

`Human Brain` is a sibling of `Fly Brain`. The projects share no runtime
package and neither imports the other.

The computational object is a sparse simulation of human-data-fitted cortical
cells. A token activates a token cell. Replayed sequences grow local synapses
onto dendritic branches. At recall, matching branches converge from short basal
contexts to specific apical contexts. Distinct continuations determine how much
lower-order current escapes into the next level, preventing a sparse long
branch from discarding reliable shorter evidence. The resulting NMDA-like
plateau currents drive postsynaptic human GLIF3 cells, and a fast-spiking
basket-cell circuit suppresses later competitors. The first pyramidal spike is
the next token.

Before that first spike, the fitted GLIF3 membrane and exponentially decaying
NMDA current form a linear system. The runtime compiles its exact forward-Euler
unit response when the cortex is created, then scales that response for the
active synaptic currents during each race. This changes neither simulated spike
time nor winner; it removes repeated integration of identical subthreshold
dynamics.

Stateful GLIF5 reservoir tokens use a separate native sparse kernel. Python
owns tokenization, topology construction, and checkpoint arrays, but one Rust
C-ABI call advances the entire biological token interval. The kernel applies
the same human-fitted membrane, adaptive-threshold, after-spike, reset, and
refractory equations and scatters only emitted spike events over fixed sparse
edges. It has no ANN runtime or dense cell-by-cell inference matrix.

Readout learning has an explicit seam with two rules. Exact ridge remains the
default comparison. The scalable `three_factor` rule combines the presynaptic
reservoir eligibility trace, the active postsynaptic output-cell competitor,
and a signed target neuromodulatory signal. Its native update changes only the
target and competitor synapses, stores no optimizer state, and requires no
reservoir-sized normal matrix or solve. Language-model growth freezes either
rule before validation and checkpointing. See `docs/local-readout-learning.md`
for the equation, locality argument, complexity, and recorded limitations.

The external interface is deliberately small:

```python
model = CellLanguageModel.grow(corpus)
prediction = model.predict_next(prompt)
text = model.complete(prompt, max_new_tokens=32)
```

Behind that seam, the implementation owns token/cell identity, structural
plasticity, sparse branch lookup, membrane integration, inhibition, and spike
readout. Tokenization is permitted non-cell plumbing. Repository retrieval,
patch application, and benchmark execution are separate mini-swe-agent
adapters. They do not get access to branch strengths or choose output tokens.
Decoding may stop after the first complete bash action, but the action text
itself is emitted token by token by cell spike races.

## Recurrent scaling layer

The scaling path uses Allen human GLIF5 reservoir cells. Each input token sends
sparse afferent current into the active population; recurrent spikes propagate
over a fixed-fanout topology; exponentially decaying eligibility traces carry
recent cell state into learned output synapses; and the next token is selected
by a race among GLIF output cells. Dormant allocated cells receive no afferent
or recurrent drive and cannot contribute eligibility.

The first reservoir regime was pathologically synchronized: 96 cells produced
only about 1.1 effective state dimensions and fired near 100 Hz. A train-only
dynamics calibration selected sparse afferents, 5 ms token intervals, lower
recurrent current, and a 40 ms eligibility trace. At 96 cells this raises
effective rank to roughly 18-21, reduces mean absolute cell correlation to
about 0.15, and keeps firing near 12-14 Hz. Cell-count conclusions are made
only inside such dynamically matched regimes.

Supported degree-matched topology probes include uniform random connectivity,
single-scale and nested groups, a 1D ring, a 2D sheet, a six-layer sheet, a 3D
volume, and overlapping mixed membership. Calibrated validation v2 found only
the layered sheet met its primary criterion at 96 cells; no structured variant
beat uniform connectivity at 192 cells. These are diagnostic populations, not
claims about brain-scale geometry.

## Native simulation boundary

The recurrent hot loop is a Rust `cdylib` loaded through a small Python FFI.
One native call advances the complete token interval, including 100 GLIF5
steps and sparse recurrent scatter. The Python reference remains executable for
equivalence tests. Native and reference paths are bit-exact across a 64-token
sequence for every persistent cell and synapse state array.

Inference allocates only linear-sized cell, afferent, and sparse recurrent
state. It does not allocate the training-only dense ridge statistics. The
current exact ridge learner is therefore deliberately identified as a separate
`O(N^2)` memory / `O(N^3)` consolidation bottleneck; replacing it with a local
plasticity rule is required before learned language scaling can follow the
98,304-cell simulation result.

## Current layer

The first working layer uses Allen Cell Types human specimen `488392806` and
its fitted GLIF3 model `503612028`. The port retains the artifact's membrane
capacitance, input resistance, threshold, reset, refractory cut, and two
after-spike currents. It is a human-data-fitted point cell, not yet a whole
human brain and not a claim of full Hodgkin–Huxley morphology. The strict
fidelity gate remains the Eyal multicompartment model followed by a validated
Neuron_Reduce/Jaxley port.

## Cell-only contract

- No transformer, dense ANN, pretrained embedding, or ANN runtime participates
  in inference.
- Learned state consists of cell identity, dendritic branches, and local
  synaptic strength.
- A next token must be the winner of a simulated membrane/spike race.
- Tokenization and byte decoding may be ordinary deterministic code.
- SWE adapters may retrieve files and apply emitted edit programs, but
  may not select repairs or rewrite cell output into a different answer.

## SWE episodic layer

For the first end-to-end SWE gate, each teacher interaction is replayed into a
64-token variable-order dendritic memory. At inference, matched context
branches produce recursively discounted synaptic current and start a race among
fitted human GLIF3 token cells; the first spike emits the next token. This layer
contains no recurrent ANN, transformer, pretrained embedding, or external
command selector.

The current SQLFluff checkpoint is deliberately instance-specific. It is a
plumbing and capacity demonstration, not the held-out checkpoint required for
the final Gemma comparison.
