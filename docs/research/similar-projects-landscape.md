# Similar-project landscape: cell-based language and program learning

**Date:** 2026-08-25  
**Status:** primary-source landscape research  
**Scope:** public papers, official repositories, model cards, and project pages

## Bottom line

Yes, there are serious projects doing important parts of the same thing, but
the landscape is fragmented. Spiking-language-model projects pursue
autoregressive text; HTM and recent dendritic SNN work pursue active-dendrite
sequence computation; Assembly Calculus, CABot, ANNABELL, Spaun, and BrainCog
pursue cognition or language in simulated neurons; and HuMAM, Digital Twin
Brain, NEST, and neuromorphic hardware pursue biological or computational
scale.

I did **not** find a public project that combines all of the following in one
system:

- autoregressive text **and** executable-program learning;
- inference performed without a Transformer, dense ANN, pretrained embedding,
  or ANN runtime;
- pinned Allen **human** GLIF3/GLIF5 point-cell dynamics;
- explicit dendritic branches, sparse recurrence, eligibility/local plasticity,
  and structural growth;
- the next token selected by the earliest simulated output-cell spike; and
- native large-cell execution plus a carefully qualified 100-billion-identity
  address-space program.

That makes Human Brain's **combination** unusual, not its individual ideas
unique. The strongest intellectual precedents are Numenta's dendritic sequence
memory and Assembly Calculus/NEMO; the closest task peers are NeuronSpark,
SpikeGPT, and `spiking-reservoir-lm`; the closest scale and human-cortex peers
are NEST, HuMAM, and Digital Twin Brain.

This is an architecture/research-program comparison, not a capability claim.
Human Brain's own [README](../../README.md) records failed free-running and
held-out program-composition gates, and its [architecture](../architecture.md)
explicitly says that it is neither a whole-brain simulation nor a
multicompartment human-neuron model.

## What was compared

The comparison target comes from the repository's [architecture
contract](../architecture.md), [pinned Allen cell provenance](../../data/allen/README.md),
[developmental-growth design](../developmental-growth.md), and [program
map](../program-map.md). Two implemented sequence paths matter:

1. a dendritic GLIF3 path that grows sparse higher-order context branches and
   chooses a token through an inhibitory first-spike race; and
2. a recurrent GLIF5 path with fitted membrane, reset, after-spike-current, and
   adaptive-threshold state, sparse recurrence, eligibility traces, and either
   ridge or a scalable local three-factor readout.

Ordinary tokenization, grammar masking, compilation, and repository adapters
are allowed plumbing, but semantic output choices must come from simulated
cell/synapse state. That boundary is stricter than the usual meaning of
"spiking language model."

## Closest current comparables

| Project | Why it is close | Crucial difference from Human Brain | Evidence status |
| --- | --- | --- | --- |
| [`spiking-reservoir-lm`](https://github.com/alfanowski/spiking-reservoir-lm) | Probably the closest independent architecture: raw UTF-8 byte prediction through a recurrent LIF spiking reservoir, with candid negative results and explicit capacity analysis. | Its reservoir is frozen and random; a trainable byte embedding and backprop-trained causal-attention readout learn the task. It uses generic LIF cells, not fitted human cells, and the attention head—not an output-cell spike race—chooses the byte. | Open repository and long first-party technical report; not peer reviewed. |
| [NeuronSpark](https://arxiv.org/abs/2603.16148) ([public model](https://huggingface.co/Brain2nd/NeuronSpark-0.9B)) | A 0.9B-parameter, from-random-initialization SNN trained for next-token prediction; the released model uses PLIF membrane dynamics and a selective state-space backbone rather than attention. | It is trained with surrogate-gradient backpropagation and retains a continuous residual stream, RMSNorm, learned parameter matrices, and an ordinary language-model generation interface. It has no Allen cell fit, active dendritic growth, local learning contract, or membrane-race vocabulary. | March 2026 preprint plus public model/code files; not peer reviewed. |
| [SpikeGPT](https://openreview.net/pdf?id=gcf1anBL9e) ([code](https://github.com/ridgerchu/SpikeGPT)) | The best-established direct precedent for an autoregressive generative SNN: binary event-driven activations, sequential token processing, and 45M/216M variants. | It is an RWKV/modified-Transformer-lineage model trained end to end with backpropagation. It does not model human-fitted cells, growing dendritic contexts, local structural learning, or an output-cell latency race. | Peer reviewed in TMLR (2024); code repository archived on 2026-06-02. |
| [Assembly Calculus / NEMO](https://www.pnas.org/doi/10.1073/pnas.2001893117) ([code](https://github.com/dmitropolsky/assemblies)) | The closest match to the overall scientific philosophy: cognition is reduced to randomly connected excitatory neurons, local inhibition, Hebbian plasticity, and assemblies. The current language-acquisition model learns word meaning, noun/verb role, word order, and generates novel grounded sentences from a small exposure set. | Its neurons advance in stylized population rounds with top-*k* selection; it is an assembly-level language model, not open-domain next-token learning with fitted continuous-time membrane dynamics. It has no Allen identity, explicit active-dendrite model, or source-program decoder. | Foundational Assembly Calculus is peer reviewed in PNAS; the [2025 language-acquisition work](https://arxiv.org/abs/2507.11788) is a preprint. |
| [Project Nord](https://github.com/gtausa197-svg/-Project-Nord-Spiking-Neural-Network-Language-Model) | An independent open spiking text project that reports from-scratch training, zonal specialization, persistent memory, and reward-modulated STDP. | Its own architecture includes embeddings, feed-forward/MoE blocks, attention, GRU/RAG components, surrogate-gradient backpropagation, and a conventional language-model head. Its STDP operates alongside backprop rather than replacing it. | Open code/model artifacts, but results are self-reported and have no peer-reviewed evaluation; the README describes both from-scratch and later distillation modes. |
| [CTP-Hybrid](https://doi.org/10.5281/zenodo.20077294) | Especially relevant because it explicitly proposes **first-spike competitive decoding** for output tokens in a Transformer-free spiking language architecture. | It uses learned spike encodings and surrogate-gradient training rather than Allen-fitted physiology, local dendritic/developmental learning, or the repository's cell-only learned-state contract. I did not locate the promised official code in the primary-source search. | May 2026 Zenodo preprint; unreviewed and currently low confidence. |

The result is a useful distinction: `spiking-reservoir-lm` is closest in
topology, NeuronSpark and SpikeGPT are closest in task, NEMO is closest in
learning philosophy, and CTP-Hybrid is closest in output-decoding language.
None is an exact match.

## Mechanism-level relatives

### Numenta HTM / Temporal Memory: closest dendritic ancestor

Numenta's sequence-memory model is the clearest precedent for the dendritic
GLIF3 lane. It represents context with sparse cell populations, grows local
synapses on active distal dendritic segments, puts matching cells into a
predictive/depolarized state, and uses inhibition to distinguish predicted from
bursting activity. The authors explicitly frame active dendrites as a way for
one pyramidal neuron to recognize many sequence contexts ([peer-reviewed
paper](https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2016.00023/full);
[community implementation](https://github.com/htm-community/htm.core)).

The difference is not cosmetic. HTM cells and segments are algorithmic binary
objects, not Allen-fitted GLIF membranes evolving in biological time, and the
system was developed mainly for sequence prediction/anomaly detection rather
than autoregressive language and program generation. Numenta now labels the
original HTM material [legacy, unmaintained
research](https://www.numenta.com/resources/htm/htmschool/), although the
community implementation remains active.

### DendriCL: direct dendritic computation, but a synthetic task

[DendriCL](https://arxiv.org/abs/2607.02283) is a July 2026 preprint showing a
single-layer compartmental SNN whose apical subthreshold recurrence is
constructed to implement leaky online Widrow-Hoff/LMS learning during the
forward pass. It reports stable performance on the Garg-2022 synthetic
in-context regression benchmark and recovery of the reference LMS trajectory
from apical state.

This is a close mechanism neighbor because computation lives in dendritic
state rather than only in weights or a deep attention stack. It is not yet a
language model, a human-fitted cell simulation, or a local-plasticity training
system. Its evidence is a new preprint, so it is a research direction to watch,
not an established comparator.

### CABot and ANNABELL: earlier neuron-level language systems

The 2009 [CABot2 parser](https://pmc.ncbi.nlm.nih.gov/articles/PMC2777190/)
parsed natural language entirely with 30,000 simulated fatiguing LIF neurons,
cell assemblies, and winner-like competition, including semantic resolution of
prepositional-phrase ambiguity. It is a genuine historical precedent for
"language processing in simulated spiking cells." It is, however, a designed
parser producing semantic frames rather than a learned autoregressive generator
or program synthesizer.

[ANNABELL](https://pmc.ncbi.nlm.nih.gov/articles/PMC4641699/)
([code](https://github.com/golosio/annabell)) learned to communicate through
natural language from an initially empty knowledge base using sparse artificial
neuron maps, *k*-winner-take-all competition, Hebbian weight updates, and
rewarded sequences of internal mental actions. The paper explicitly says its
neuron design prioritized computational efficiency over biological detail. It
therefore overlaps Human Brain's incremental/local-learning ambition, but not
its spiking physiology or cell-only output rule.

## Broader adjacent systems

### Functional spiking cognitive architectures

- [Spaun](https://pubmed.ncbi.nlm.nih.gov/23197532/)
  ([Spaun 2.0 code](https://github.com/xchoo/spaun2.0)) joined 2.5 million
  spiking neurons into one model that performed eight perception, memory,
  reasoning, question-answering, and motor tasks. [Nengo
  SPA](https://www.nengo.ai/nengo-spa/) still provides spiking cognitive models
  for associative memory, question answering, sequencing, and command parsing.
  Spaun uses Neural Engineering Framework/Semantic Pointer designs and
  hand-composed task structure; it is not an acquired autoregressive language
  model. [BioSpaun](https://arxiv.org/abs/1602.05220) moved Spaun toward more
  detailed compartmental neurons, but not toward Human Brain's text-learning
  contract.

- [BrainCog](https://www.cell.com/patterns/fulltext/S2666-3899(23)00144-7)
  ([code](https://github.com/BrainCog-X/Brain-Cog)) is a broad, actively
  developed spiking cognitive-intelligence and brain-simulation platform. It
  includes neuron models, STDP and backprop-based learning, structural
  development, reasoning, decision-making, social cognition, and sequence/music
  generation. It is a toolkit and family of models, not one cell-only
  autoregressive language/program system; its published implementation also
  uses deep-learning components.

These projects show that multi-task cognition in spiking networks is established
precedent. Human Brain differs by making generic token emission and held-out
program composition the target rather than wiring a fixed battery of cognitive
tasks.

### Spiking LLMs designed mainly for efficiency

There is now a lively but materially different spiking-LM cluster:

- [SpikeLLM](https://proceedings.iclr.cc/paper_files/paper/2025/file/510e7d39fce008a3e31de54b8f5be9ac-Paper-Conference.pdf)
  redesigns/quantizes 7B--70B LLaMA-family models with spiking mechanisms;
- [SpikingBrain](https://arxiv.org/abs/2509.05276) uses adaptive spiking
  neurons, linear/hybrid attention, conversion-based training, and 7B/76B
  models; and
- the 2026 [Winner-Take-All Spiking
  Transformer](https://arxiv.org/abs/2604.11321) replaces softmax inside
  attention with spike-driven WTA and trains encoder/decoder Transformers end
  to end.

These are important task baselines, but their objective is largely efficient
large-model training or deployment. "Winner take all" in a spiking attention
block is not the same as a vocabulary of fitted output cells whose membrane
latencies determine the next token. Conversion, distillation, attention,
ordinary dense parameter matrices, and global gradient training also put them
outside this repository's cell-only contract.

### Human-data and scale peers

- The peer-reviewed [Human Multi-Area Model
  (HuMAM)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11491286/)
  ([EBRAINS model](https://search.kg.ebrains.eu/instances/5ba4cf48-296e-43b8-a12d-0ff6d822c821?noSilentSSO=true))
  simulates 34 human cortical areas with 3.47 million LIF neurons and 42.8
  billion synapses, integrating human anatomy, electrophysiology, morphology,
  tractography, and resting-state data. It is a much stronger precedent for
  human cortical structure and network scale, but it studies spontaneous and
  evoked dynamics, not language learning.

- Fudan's peer-reviewed [Digital Twin
  Brain](https://academic.oup.com/nsr/article/11/5/nwae080/7617698) reports
  personalized, imaging-constrained spiking cortico-subcortical simulations up
  to 20 billion LIF neurons for resting-state, visual, and auditory analyses.
  This is far beyond Human Brain's current resident-cell scale, but it does not
  train the spiking population to generate language or programs.

- NEST's exascale work reports [1.51 billion resident neurons and 16.8 trillion
  synapses](https://www.frontiersin.org/journals/neuroinformatics/articles/10.3389/fninf.2018.00002/full),
  while its official [e-prop
  examples](https://nest-simulator.readthedocs.io/en/stable/auto_examples/eprop_plasticity/index.html)
  implement eligibility-based three-factor learning. NEST is therefore a
  strong runtime and learning-rule precedent, not a competing language model.

Large biological simulation is thus not novel by itself. Human Brain's 100B
work should remain phrased as an addressable identity space with a declared
resident/updated subset, exactly as its [100-billion-cell architecture
document](100-billion-cell-architecture.md) already requires.

## Feature-by-feature assessment

| Human Brain feature | Serious public precedent? | Closest source-backed peer | Assessment |
| --- | --- | --- | --- |
| Autoregressive generation with spiking units | Yes | SpikeGPT, NeuronSpark, `spiking-reservoir-lm` | Active field, not unusual alone. |
| Local/Hebbian learning in neuron-level language | Yes, at toy/cognitive scale | NEMO, ANNABELL | Established precedent, but not modern open-domain LM capability. |
| Active dendrites for contextual sequence computation | Yes | HTM, DendriCL | Strong mechanism precedent; rarely joined to language generation. |
| Human-data-constrained spiking cortex | Yes | HuMAM, Digital Twin Brain, Allen ecosystem | Strong neuroscience precedent; not trained for language/program output. |
| Earliest-spike vocabulary decoding | A very recent claim exists | CTP-Hybrid | Still rare; Human Brain's fitted-cell latency race remains materially different. |
| No ANN/attention/embedding in inference | Rare among generative SNN LMs | NEMO and HTM satisfy related constraints but are not general LMs | Clear differentiator. |
| Learned source-program composition from cells | No close public match found | Assembly Calculus planning is adjacent, but its programs are authored in the calculus | Clearest apparent whitespace; absence claim is search-bounded, not proof of uniqueness. |
| Native million-cell and larger address-space engineering tied to language | Runtime scale exists, task coupling does not | NEST, HuMAM, Digital Twin Brain | Unusual integration, not a new scaling technique by itself. |

## Defensible conclusion

The accurate answer is **"yes, but nobody found here is doing the exact same
combination."** Human Brain sits at the intersection of four communities that
usually do not meet:

1. spiking language models;
2. biologically local/dendritic sequence learning;
3. human-data-constrained cortical simulation; and
4. large sparse simulation/runtime engineering.

Its nearest conceptual ancestor is HTM; its nearest cognitive-language cousin
is Assembly Calculus/NEMO; its nearest architectural implementation is
`spiking-reservoir-lm`; its nearest current task peers are NeuronSpark and
SpikeGPT; and its scale peers are NEST, HuMAM, and Digital Twin Brain.

The synthesis appears genuinely unusual in public work as of 2026-08-25. The
claim should remain **"unusual combination"**, not **"first"** or **"unique"**:
the search cannot prove a negative, several 2026 systems are unreviewed and
changing quickly, and CTP-Hybrid already shows that even first-spike token
decoding is no longer exclusive.
