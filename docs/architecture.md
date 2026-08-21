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
