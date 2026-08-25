# Neural Observatory

Neural Observatory is the native experiment workbench for Human Brain. It
reads the canonical ladder result layout directly:

```text
experiments/<experiment_id>/<arm>.json
```

It is a Dear PyGui desktop application, not a browser or local web server. The
interface keeps the experiment cohort, comparison plot, run matrix, and one
selected run visible together. This follows the proven overview-and-details
pattern used by scientific tools such as Cytoscape, Gephi, and ParaView while
remaining small enough for the current research workflow.

## Install and launch

Use the repository's accepted Python 3.11 interpreter:

```powershell
uv --cache-dir .uv-cache pip install --python .venv\Scripts\python.exe -e ".[visualizer]"
& ".venv\Scripts\python.exe" -m humanbrain.visualizer.cli experiments --experiment visualizer_demo
```

The path can point at the full `experiments` tree or a smaller tree with the
same `<experiment>/<arm>.json` layout. The optional `--experiment` selects the
initial cohort. Arm-name search and plot controls are available in the left
rail.

Generate fresh input with the standard harness:

```powershell
& ".venv\Scripts\python.exe" benchmarks\ladder_experiments.py `
  --experiment observatory_demo `
  --arm similarity-1024 `
  --N 1024 `
  --seed 7 `
  --codebook similarity `
  --with-state-separation `
  --record-observatory `
  --observatory-samples 128 `
  --observatory-state-stride 20
```

Restart the workbench after generating a run. Results and artifacts remain
read-only; the visualizer never rewrites experiment JSON.

## Measurements

The workbench reports only measurements supported by the loaded result. It
does not infer spike activity from configuration values.

| Measurement | Definition |
| --- | --- |
| Train, calibration, replay rate | Observed successes divided by cases, with Wilson 95% score intervals |
| Generalization gap | Train proportion minus calibration proportion, with a Newcombe hybrid-score 95% interval |
| Decision margin | Recorded median and minimum output-cell race margin in pA |
| Representation separation | Median pairwise `1 - cosine similarity` for recorded eligibility and synaptic-current states |
| E:I composition | Enabled excitatory and inhibitory population counts and their ratio |
| Input assembly sparsity | Assembly size divided by configured population |
| Recurrent density | Recurrent fanout divided by configured population |
| Contact estimates | Input, recurrent, and readout contacts implied by the effective configuration |
| Population activity reference | Two percent of enabled neurons, clearly labelled as a capacity reference rather than an observed firing rate |
| Population firing rate | Exact spike count during a token interval divided by enabled neurons and biological interval seconds |
| Neuron-step activity density | Exact token spike count divided by enabled neurons and integration steps |
| Sampled live state | Actual relative membrane voltage, synaptic current, and eligibility for deterministic address-space-wide GIDs at the configured state cadence |

The current recorder is truthful at token resolution. It does not yet receive
the identity and integration-step time of every spike from every execution
kernel. Therefore inter-spike-interval CV, Fano factor, spike-time tiling
coefficient (STTC), and a spike raster remain unavailable rather than being
approximated from token totals. This follows the monitor/recorder split used by
[NEST](https://nest-simulator.readthedocs.io/en/stable/devices/record_from_simulations.html)
and [Brian](https://brian2.readthedocs.io/en/2.7.0/reference/brian2.monitors.html).

The Fano factor is a useful trial-to-trial variability measure, but its window
and alignment matter ([Churchland et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC2828350/)).
STTC is preferred over a raw correlation score for sparse spike trains because
it is less confounded by firing rate
([Cutts and Eglen](https://www.repository.cam.ac.uk/items/a21a8743-f2e0-432f-8d3c-9fdbfbfb1a84)).
Participation ratio provides a spectrum-based effective dimensionality rather
than a hard component cutoff
([Gao et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC6655892/)).

## Inspect network structure

The Structure workspace uses an address-space lens instead of a force-directed
view of the entire graph. Enter any GID and choose one to three outgoing hops.
The center node, returned neighbors, edge slots, direction, and E/I roles are
exact. The UI renders at most 160 nodes in one lens and always reports
`shown / configured`; changing the configured population from 512 to one
billion does not change that memory bound.

Current materialized ladder results are reconstructed exactly up to the CPU
science-run cap of 4,096 neurons. Larger materialized results must record a
queryable topology representation; aggregate configuration is not enough to
recover arbitrary RNG rows without materializing or replaying the graph.

Procedural networks use the existing constant-space `TopologyManifest` and
`ProceduralTopology` oracle. If `config_dump.procedural_topology_manifest` is
present, the selected run opens it automatically. A direct manifest can also
be inspected:

```json
{
  "cell_count": 1000000000,
  "fanout": 8,
  "topology": "uniform_no_self",
  "seed": 7,
  "protocol": "random123-philox2x64-10-v2"
}
```

```powershell
& ".venv\Scripts\python.exe" -m humanbrain.visualizer.cli experiments `
  --structure-manifest path\to\manifest.json
```

The connection-density matrix is exact when every source row is available
within the reconstruction cap. At larger procedural scale it uses 4,096
deterministically distributed source GIDs and labels the result as sampled;
every sampled edge is exact, but the matrix is not presented as a census.

## Watch a running experiment

`--record-observatory` opens this flush-visible stream before training:

```text
experiments/<experiment_id>/.observatory/<arm>.jsonl
```

The visualizer can wait for and tail that path even before the metrics JSON
exists. Start it in one terminal:

```powershell
& ".venv\Scripts\python.exe" -m humanbrain.visualizer.cli experiments `
  --activity-stream experiments\live_run\.observatory\arm.jsonl
```

Then run the ladder command with `--experiment live_run --arm arm` and
`--record-observatory` in another terminal. The Activity workspace polls the
file every 250 ms and can play or scrub the data while the writer continues.
The reader retains a rolling 20,000-frame decoded window for live monitoring,
but separately indexes every complete frame with a 64-bit file offset. Frames
outside the window are loaded directly from the append-only JSONL stream, so
the start of a long experiment remains replayable after the live window has
moved on. Large existing traces are indexed in bounded chunks so the desktop
UI can continue rendering while it catches up.

Spike totals, firing rate, density, phase, and token identity are written for
every token. Full sampled state is larger, so
`--observatory-state-stride 20` records it every 20 tokens by default. The UI
identifies the exact state frame being displayed when the activity cursor lies
between sampled-state frames. Playback advances discrete scientific frames at
the selected biological-time scale; there is no visual interpolation.

## Replay an experiment

Select a run with an Observatory stream and open the Activity workspace. The
full-run slider seeks to any indexed token frame. `Prev frame` and `Next frame`
advance exactly one recorded token boundary. Playback speeds range from
`0.001×`—five seconds per frame for the default 5 ms token interval—to `10×`.
Changing speed retimes the next discrete advance; it does not synthesize
intermediate voltage or spike state.

`Prev event` and `Next event` can navigate all events or one exact event class:

- **Phase change** — the first recorded frame in a new experiment phase.
- **State sample** — a frame containing recorded voltage, current, and
  eligibility samples.
- **Zero-spike token** — a token interval whose recorded population spike
  total is zero.

The selected-frame inspector reports replay position, original frame index,
biological time, phase, token, spike count, firing rate, activity density, and
the age of the sampled state being drawn. The firing-rate overview uses a
min/max envelope when a trace exceeds the plotting budget, preserving brief
peaks and quiet intervals instead of simply dropping every nth frame.

Replay reads the recorded trace; it does not re-execute a frozen experiment or
alter its metrics artifact. Compact offsets and numeric timeline/event indexes
scale with token-frame count, while decoded sampled-state objects stay bounded.
Configured neuron count does not affect replay index size.

## Scale and model boundaries

The UI cost is proportional to result files, plotted observations, the bounded
neighborhood, and the bounded live-state sample—not to configured neuron count.
The visualizer never materializes one UI object per neuron or synapse. The
Population Ledger uses the same truthful vocabulary as the procedural runtime:
configured, enabled, estimated contacts, and the current concurrent-activity
reference remain distinct.

The reader is the only current-schema adapter. It normalizes ladder JSON into
model-neutral `ExperimentRun`, `PopulationSummary`, outcome, and diagnostic
records before the native UI sees the data. A future cell implementation should
add or replace a reader adapter while keeping those observation concepts; no
GLIF equation or simulator object is coupled to the presentation layer.

Malformed or unsupported JSON is counted in Catalog Health instead of being
silently guessed into the ladder schema. This protects comparisons as result
formats grow.

## Source layout

- `src/humanbrain/visualizer/catalog.py` — discovery, strict validation, and
  current ladder-schema adaptation.
- `src/humanbrain/visualizer/model.py` — model-neutral observations and
  statistical estimates.
- `src/humanbrain/visualizer/structure.py` — exact bounded structure-query
  interface and dense/procedural adapters.
- `src/humanbrain/observatory.py` — deterministic flush-visible activity
  recorder.
- `src/humanbrain/visualizer/activity.py` — incremental, disk-backed full-run
  replay index and exact event navigation.
- `src/humanbrain/visualizer/app.py` — native Dear PyGui workbench.
- `src/humanbrain/visualizer/cli.py` — launch boundary.
- `tests/test_experiment_visualizer.py` — canonical demo and validation tests.
- `tests/test_observatory.py` — exact topology and live-stream tests, including
  a one-billion-neuron procedural fixture.
