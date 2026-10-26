# Human Brain

Human Brain is a research codebase for learning and text prediction with simulated cortical cells. It combines Allen human-data-fitted GLIF cell equations, sparse dendritic associations, recurrent state, local plasticity, and experiments that connect retained knowledge to software actions.

The useful starting point is a small text learner: grow context associations from a corpus, select the next token through a simulated cell race, generate a continuation, and save the learned state. Larger experiments investigate persistent reading and editing. This is an experimental learning system, not a whole-brain simulation or a demonstrated general-purpose coding agent.

## What is implemented

- A byte/phrase tokenizer and `CellLanguageModel` API for growth, incremental learning, prediction, completion, evaluation, and checkpoint persistence.
- A dendritic context memory whose Witten–Bell-style interpolation supplies synaptic drive to fitted-cell races. This makes the statistical baseline part of the interpretation of results.
- GLIF3 and adaptive GLIF5 cell models, sparse recurrent/readout machinery, and a Rust native kernel with optional CUDA execution paths.
- Research harnesses for bounded scaling, typed programs, persistent memory, and actual execution of small software edits.
- Source-bound experiment protocols, result artifacts, baseline comparisons, and public developer QA tiers.
- An optional Dear PyGui visualizer for stored experiment structure and activity.

## Getting started

Python 3.11 is the tested development baseline; package metadata accepts Python 3.11 or later. The maintained bootstrap below uses Windows PowerShell. Native QA also needs Rust/Cargo and the MSVC C++ build tools with the Windows SDK. CUDA and the visualizer are optional.

From the repository root:

```powershell
./tools/setup.ps1
```

Setup creates or reuses `.venv`, installs `.[dev]` using `tools/dev-constraints.txt`, verifies the environment, and runs the public smoke suite. Package installation requires an accessible package index or populated cache. The constraints pin a direct development baseline; they are not a complete transitive dependency lock. To select a Python interpreter when creating the environment, use `./tools/setup.ps1 -Python 'C:/path/to/python.exe'`.

### A small language example

Start the checkout's interpreter:

```powershell
./.venv/Scripts/python.exe
```

Then enter:

```python
from humanbrain import CellLanguageModel, ModelConfig

corpus = (
    "the membrane predicts the next word.\n"
    "the membrane predicts the next word.\n"
    "the cortex predicts the next token.\n"
)
model = CellLanguageModel.grow([corpus], ModelConfig(max_order=8))
prediction = model.predict_next("the membrane predicts the next ")
print(prediction.text, prediction.spike_time_ms)
print(model.complete("the membrane ", max_new_tokens=8))

model.save("cell-model.npz")
restored = CellLanguageModel.load("cell-model.npz")
print(restored.predict_next("the membrane predicts the next ").text)
```

This example follows the fixture in `tests/test_language.py`, which expects the next token to be `word` and checks checkpoint round-trip equality. It illustrates recall from a tiny repeated corpus, not held-out language understanding. It requires no external text dataset.

## Development checks

```powershell
./.venv/Scripts/python.exe -m pytest tests/test_language.py
./.venv/Scripts/python.exe tools/dev.py qa native
./.venv/Scripts/python.exe tools/dev.py qa integration
```

Native QA prepares and verifies the compiled library; integration QA exercises selected public headless fixtures. The Windows Developer QA workflow also checks developer tooling and runs Rust tests with locked, vendored dependencies. Passing these tiers does not mean every research protocol or test in the repository passes.

See [development workflow](docs/development.md) for test selection, build identity, local cache configuration, and QA receipts. Frozen experiment checks can intentionally reject changed sources. Review and deliberately update the relevant protocol rather than disabling its digest checks. Some cohorts also bind the Python minor version or require optional dependencies and separately prepared corpora.

## Current research status

The central open problem is complete, useful learning across reading, retention, and software action. The [current roadmap](docs/research/roadmap-reading-to-coding.md) is the entry point for active work; [experiment history](docs/research/roadmap-experiment-history.md) preserves earlier directions.

The [bounded repeated-teaching study](experiments/r4_integrated/repeated_teaching_acquisition_v1_result.md) reports that the online learner acquired and retained the intended edit boundaries, but both online and learning-disabled routes still failed the complete reading and editing tasks after 32 corrections per skill. Generated payloads remained wrong. This supports a narrow local-learning result, not successful integrated acquisition. A later learning-rule revision has mechanics and preservation evidence; the proposed larger acquisition comparison has no released execution result in the current roadmap.

Earlier experiments include useful small prediction and program-reuse results alongside failed free-running generation and cross-structure composition. High teacher-forced accuracy, successful replay, and an instance-specific software checkpoint are different claims from generalization. Likewise, configured cell address space and measured simulation throughput do not establish useful recruitment, biological fidelity, or cognitive ability.

## Repository map

| Path | Purpose |
| --- | --- |
| `src/humanbrain/language.py`, `cortex.py`, `tokenizer.py` | Small text-learning interface and dendritic context memory |
| `src/humanbrain/cells.py`, `recurrent.py`, `readout.py` | Fitted-cell equations, recurrent state, and sparse output learning |
| `src/humanbrain/_native_src/` | Rust kernel, Cargo lockfile, and vendored dependencies |
| `src/humanbrain/simple_swe/` | Small software-task protocols, compilation, and execution support |
| `experiments/` | Experiment runners and result narratives |
| `benchmarks/`, `artifacts/` | Protocols, measurements, and saved research evidence |
| `tests/`, `tools/` | Regression checks, bootstrap, and developer QA |
| `data/allen/` | Pinned fitted-cell inputs and provenance |
| `docs/` | Architecture, development guide, and research plans |

## Provenance and licensing

The cell equations and fitted parameters build on Allen Cell Types data and AllenSDK model rules. See [pinned cell provenance](data/allen/README.md) for the exact models and checksums. The project-specific contribution is their integration into learning circuits, execution machinery, and experimental protocols; the underlying biological models and statistical techniques are prior work.

Python dependencies and optional integrations are declared in `pyproject.toml`; native dependencies are declared in `src/humanbrain/_native_src/Cargo.toml`. Preserve the license notices shipped with vendored components. No repository-wide license grant is declared; obtain the owner's terms before reuse or redistribution, and check dataset and dependency terms separately.
