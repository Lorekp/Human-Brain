# One fixed multiscale neuronal writer comparison

Status: prospective design review, 2026-09-06. No implementation, new model
observations or acquisition was performed for this review. The
[roadmap](roadmap-reading-to-coding.md) and
[active plan](persistent-memory-byte-development.md) own subsequent execution.

## Decision and evidence

Select a fixed heterogeneous update fraction within the existing 64 writer
neurons: indices 0–31 use `alpha=0.5`, 32–47 use `0.125`, and 48–63 use
`0.03125`. These are three generic temporal populations, not semantic roles.
Every key, value and write-strength map retains access to every writer neuron;
the full recurrent matrix remains trainable under the existing local rule.
Keep the 32-neuron query cell at its current homogeneous `0.5` fraction. Do not
learn the fractions, rescale their activity, add a token parser, or change the
tokenization, write boundary, read rule, output module or learning rate in this
comparison. This fixed geometrically spaced choice retains half the original
fast population while allocating the other half to two slower responses. It is
a prospective engineering choice, not a literature-derived optimum or a sweep.

The [binding diagnostic](../../experiments/r4_integrated/local_memory_binding_v1.md)
at the current value-credit checkpoint establishes the following limited facts.
Its saved audit is `.pytest-tmp/local-memory-binding-v1-independent-audit.json`,
SHA256 `bd5c86fbf800519ada5053d1ead22c5ec78ba3062b058f58e9e2e628e6bf0b4c`.
Tagged writer identifier contrast retains 10.576%/9.864% of its key-end norm
before the value arrives, and 3.596–4.558% at the first document's EOS. At the
second value byte, write-key differences are 0.045808–0.090723 for different
identifiers carrying the same value, versus 0.686867–0.738152 for different
values under the same identifier. The frozen writer shows the same qualitative
pattern. The historical checkpoint's pre-value retention was 10.853%/10.139%;
the repaired query/value teaching has not visibly removed this attenuation.

The source-swap response is already mostly common to both questions at the first
value byte, before the second document can overwrite it. At the final read,
tagged common-component norms are 0.433311–0.476839 versus 0.062026–0.088728 for
the half difference-of-differences. All eight paired questions per mode preserve
the same output when the ordered values are held fixed and their record IDs are
exchanged. Nevertheless, both modes solve 5/16 individual questions, and one
original-order question solves both assignments. This is evidence of weak
record-identity binding on an exposed development roster, not absence of all
binding or irreversible information loss. The audit's residual-accounting
identity does not independently certify that all discrepancies are rounding.

The selected question is whether a generic temporal state can make earlier
context more available when later values drive the writer. Address learning,
write-strength learning and interference remain alternative or interacting
limitations. Neither a new boundary gate nor a more exact controller signal is
proved unnecessary. They are deferred so this comparison changes one design.

## Primary-source basis and its limits

Perez-Nieves et al. report benefits from heterogeneous neuronal time constants
in temporal tasks, including fixed heterogeneous settings. Their surrogate
gradient experiments use BPTT; their other learning regime is FORCE. This
supports heterogeneity as a representation hypothesis, not the legitimacy or
success of our local training rule, and does not select our fractions or bank
sizes. [Primary paper](https://www.nature.com/articles/s41467-021-26022-3),
[authors' code](https://github.com/npvoid/neural_heterogeneity).

Bellec et al. separate synaptic eligibility from neuronal learning signals.
That distinction supports stating precisely which derivative our trace keeps
and which credit remains approximate; it does not make this custom dense rate
cell or its plastic-memory controller an exact e-prop implementation.
[Primary paper](https://www.nature.com/articles/s41467-020-17236-y).
The derivation below follows the actual
[rate-cell source](../../src/humanbrain/plastic_memory.py) and is independent
algebra, not a borrowed convergence result. The graph service returned
`Transport closed`; source inspection used the known repository file.

## Forward rule and local eligibility

For fixed per-neuron fractions `0 < alpha_j <= 1`, define

```text
s_j(t) = sum_i W_ji x_i(t) + sum_k U_jk h_k(t-1) + b_j
a_j(t) = tanh(s_j(t))
h_j(t) = h_j(t-1) + alpha_j [a_j(t) - h_j(t-1)]
d_j(t) = alpha_j [1 - a_j(t)^2]
c_j(t) = (1 - alpha_j) + d_j(t) U_jj
```

For a synapse terminating at neuron `j`, retain the same own-neuron recurrent
path as the current implementation:

```text
e^W_ji(t) = c_j(t) e^W_ji(t-1) + d_j(t) x_i(t)
e^U_jk(t) = c_j(t) e^U_jk(t-1) + d_j(t) h_k(t-1)
e^b_j(t)  = c_j(t) e^b_j(t-1)  + d_j(t)
```

These expressions follow by differentiating the additive membrane carry and
the neuron's own tanh drive while holding other neurons' indirect parameter
dependence outside the retained path. All off-diagonal **forward** recurrent
connections remain active. Their derivative paths through other neurons remain
omitted. No history-wide reverse pass, external per-synapse derivative, optimizer
or `.grad` is introduced. The neuronal teaching signal and existing
`-eta * signal_j * eligibility_ji` publication remain unchanged.

The formulas apply to each batch row and preserve all old activity and traces
on masked steps, including padding with nonfinite ignored inputs. They use the
previous activity as recurrent presynaptic input, not the newly published one.
The input, activity, eligibility and proposed parameter finite checks remain.
Fractions are fixed, finite, correctly shaped, and follow the model's device and
floating dtype. They have no learning signal or retained gradient. With zero
initial state and finite tanh drive, fractions in `(0,1]` keep forward activity
in the coordinatewise interval `[-1,1]`; this does not bound eligibility after
the trainable recurrent weights move. Keep the existing overflow rejection.

The memory's writer parameter tags continue to consume these raw local traces.
Their existing controller/key/strength approximations do not become exact.
The direct value-neuron signal retains its exact conditional value-map
interpretation during a fixed-weight ingestion interval, with the same
truncation at tag retirement. Actual read-path query credit, pre-update LMS
context teaching, float64 sum/mean and single checked upstream cast remain
unchanged. No extra alpha factor belongs in the later value signal or tag update:
the changed cell susceptibility is already present in the controller trace.

When every fraction is `0.5`, the forward expression is exactly the current
`state + 0.5*(activation-state)` operation, and `d` and `c` reduce to the current
`.5*(1-a^2)` and `.5+d*U_jj`. Preserve this operation ordering; do not replace it
with a reassociated weighted sum and claim bitwise equality. Ordinary synthetic
float32/float64 QA must establish exact default forward, masks, traces and
teaching trajectories against the frozen old implementation. The algebra is
real arithmetic; discrete floating outputs and exceptions require those checks.

## Retention is not absolute signal or learning gain

For the leak-only linear example, an isolated unit input contributes
`alpha*(1-alpha)^lag`. Reporting only `(1-alpha)^lag` would omit the reduced
injection. At lag 11 the leak-only impulse magnitudes for fractions
`0.5, 0.125, 0.03125` are approximately `0.000244, 0.0288, 0.02204` respectively.
This is an illustrative analytic kernel, not a prediction of the trained cell.
With the actual initial `U_jj=0.9` and unsaturated tanh, the retained self-path
carry is instead `1-0.1*alpha`, approximately `0.95, 0.9875, 0.996875`; the
input susceptibility is still proportional to alpha. Hence one cannot infer
the actual memory horizon from alpha alone, and a slower bank can have smaller
activity, lower credit injection, slower acquisition and more cross-document
carry. The same learning rate is intentional; no inverse-alpha compensation or
map normalization should conceal this tradeoff.

More directly, the near-zero linearized impulse response with self-weight `u`
is `alpha*(1-alpha+alpha*u)^gap`. At initial `u=0.9` and gap 11, the three
absolute responses are approximately `0.28440, 0.10885, 0.03019`: the slower
banks retain a larger fraction but have **smaller absolute responses** in this
linearized example. The nonlinear driven network can change that balance.
The candidate is therefore a test of temporal diversity, not a mathematical
guarantee of stronger identifier activity over the observed filler interval.

Any already authorized representation receipt must report absolute identifier
and value contrasts, activity norms, write-key norms/cosines and absolute local
susceptibility along with retention ratios, separately for the fixed banks.
Zero denominators stay undefined. Norm per square-root bank width can make the
32/16/16 sizes explicit, but must not replace raw norms. Do not introduce new
checkpoint probes merely to populate these measures: a new observation budget
requires its own declaration. The integrated action criterion below, not a
retention threshold or fitted host decoder, determines usefulness.

## Why keep the population size instead of adding traces

The selected rule changes no trainable parameter shape, activity dimension,
connectivity or local eligibility/tag storage. A fixed writer vector costs 64
floating scalars: 256 bytes in float32, 512 in float64. If implementation also
materializes immutable default vectors on other cells, count those separately;
they are not new learned parameters. The writer still has 64 simulated neurons.
Per stream its cell state has 64 activities and 6,208 eligibility entries:
2,048 input, 4,096 recurrent and 64 bias entries. The fixed fractions need no
per-stream history or new eligibility tensor. Elementwise multiplications
replace scalar multiplications; actual whole-workload cost must be measured.

An explicit extra 64-unit filtered activity bank would preserve all original
fast units but add 64 activations and a second update path. Giving all three
writer maps access to the concatenated 128 units adds at least
`64*(64+64+1)=8,256` learned map weights (33,024 float32 bytes). Under the current
64-direction parameter-tag layout, those map weights alone add 528,384 tag
entries, or 2,113,536 bytes per tagged retained state. Filtering learned activity
also requires eligibility for its upstream dependence; carrying one additional
own-path eligibility bank already adds 6,208 entries before any richer coupling.
These are concrete costs of this alternative, not a theorem for every possible
trace architecture. Input-only traces could choose different costs but change
the interface and features too. The within-64 option is the smaller isolating
experiment. Three rates rather than two add no tensor dimensions or asymptotic
work, while preserving a fixed fast half and broader generic timescales.

## Exact reference and one integrated execution contract

Use six arms: heterogeneous tagged, aggregate, writer-maps-frozen, frozen,
writes-disabled, and immediate homogeneous-current tagged. Candidate frozen
controls must retain the **candidate** writer fractions: otherwise a tagged
advantage would conflate learning with the temporal prior. Their forward
trajectories are not required to match the old scalar-writer controls. All
candidate arms share the same initial arrays and fractions. The homogeneous
reference differs only in the declared fractions and its reference metadata.
Its input codes, maps, recurrent weights, query parameters, output weights and
fixed remaining teaching broadcasts start seed-identically to the predecessor.

Preserve scalar fractions as ordinary constructor scalars under one unified
arithmetic law; only a per-neuron vector needs a registered fixed buffer. The
default scalar `.5` then retains the old state-dictionary keys and exact
arithmetic, while the selected writer vector moves/casts with its model and
has explicit learner metadata. The query stays scalar `.5`. This is a genuine
scalar/vector cell API, not a state-loading shim. Expose explicit writer fractions
through the joined learner too, so other supported neuron counts and authored
tiny QA can declare their own scalar/vector configuration. Omitted configuration
selects the reviewed bank at 64 neurons; do not invent a proportional allocation
rule for other sizes. The frozen comparison uses only the declared 64-vector.
Production uses the selected
fixed heterogeneous writer; do not
add a retired learner compatibility mode, migrate old checkpoints, invent a
reference-only feedback buffer or load Git source during evaluation. The
experiment-only reference may use the common scalar-default law after true
old-source trajectory parity is established before freezing. State custody must
explicitly identify the rates; initial tensor matching must permit only that
declared difference. Do not silently reuse the preceding harness assumption
that every comparison differs by one obsolete feedback tensor.

Freeze one fresh seed-733 acquisition with the existing six-arm shared schedule:
1,024 episodes per arm, rate 0.01, fresh zero LMS output weights, 32 input-code
features, 64 key/value dimensions, twelve retained memories per arm, two generated
bytes plus learned EOS, 8 fit and 4 each novel-value/delay/interference actions.
Use the existing raw documents, independent key/value assignments, same-question
swaps, source-free worker and real Python motor. Writer carry and credit still
cross document terminators and purge only after the whole document list. No
epoch, seed, leak, tokenizer, delimiter or gate sweep is authorized by this
review. The motor remains supplied host action-selection scaffolding.

Before any acquisition, require scalar numerical eligibility QA for the selected
fractions, diagonal-only exact derivative witnesses, masking, finite proposal
failures, and default `.5` parity; these are authored unit arrays, not protected
or endpoint model probes. Verify the same raw local traces reach memory tags,
that frozen fractions never learn, and that no query-cell rate changes. Then
freeze a fresh complete cost workload: 24 first attempts, 72 teacher offers,
60 enabled output opportunities, 120 action attempts, two motor demonstrations
and six capped predictions. Use the existing twice-learning plus twice-other-time
plus 30-second projection and original 900-second/2,048-MiB ceiling. Do not reuse
value-credit admission or omit eligibility validation to make the cost pass.
If rejected, preserve the cost evidence and stop rather than relaxing the cap.

If admitted, execute the single full workload and saved-only audit. Usefulness
still requires tagged 8/8 fit and 4/4 in each other stratum, with strict novel
success above candidate writer-frozen, frozen, writes-disabled and the immediate
homogeneous reference; report aggregate separately. A loss gain, slower norm
decay or fit-only improvement cannot establish learned writing or qualify the
gate. This comparison also changes the population's average rate, so even a
useful result would not uniquely identify heterogeneity versus a slower average
as the cause. Do not add another homogeneous leak arm after seeing the outcome.
Failure closes this fixed representation test for reassessment; success returns
directly to broader document–memory–action learning rather than another component
queue. No result would yet establish general coding or the long-run model targets.
