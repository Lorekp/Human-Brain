"""Execute below-STOP raw candidate leaders during teaching, keeping evaluation fixed.

The fixture links openly reused teaching and declares fresh transfer instances.
Launch with CPU and checkout native-cache environment set before Python starts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.r4_integrated import contextual_binding_python_session as binding

SOURCE_PATHS = (
    *binding.SOURCE_PATHS,
    "experiments/r4_integrated/contextual_exploration_python_session.py",
    "experiments/r4_integrated/contextual_exploration_cases.json",
    "experiments/r4_integrated/contextual_exploration_protocol.md",
)


def run(output: Path, model_dir: Path) -> dict:
    return binding.run(
        output,
        model_dir,
        fixture_path=binding.HERE / "contextual_exploration_cases.json",
        exploration_policy="raw_candidate",
        source_paths=SOURCE_PATHS,
        include_rank_diagnostics=True,
        report_labels={
            "kind": "public development; execution-grounded exploration of raw candidate leaders",
            "teaching_exposure": "sixteen paired-entity teaching requests and 48-entry schedule "
            "openly reused via linked fixture; fresh transfer substitutions declared separately",
            "feedback_contract": "after neuronal STOP execute unique raw candidate leader, even "
            "at or below STOP; cosine only on candidate ambiguity; evaluation abstention unchanged",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=binding.ROOT / ".models/all-MiniLM-L6-v2")
    args = parser.parse_args()
    result = run(args.output, args.model_dir)
    print(binding._json({"terminal": result["terminal"], "output": str(args.output)}))
