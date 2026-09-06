"""Direct value-neuron credit versus its immediate query-repaired predecessor."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r4_integrated import local_memory_query_credit_v1 as previous  # noqa: E402
from experiments.r4_integrated import local_memory_six_arm_harness as harness  # noqa: E402

ARMS = previous.ARMS
file_hash, write_json = previous.file_hash, previous.write_json
duration_clock, validate_cost_launch = previous.duration_clock, previous.validate_cost_launch
roster, teaching_schedule = previous.roster, previous.teaching_schedule
project_cost, qualifies = previous.project_cost, previous.qualifies
detached_state, digest = previous.detached_state, previous.digest


@dataclass(frozen=True)
class ValueCreditSpec(previous.QueryCreditSpec):
    """One fixed six-arm comparison; the CLI exposes no configuration hooks."""

    @property
    def config(self):
        return {
            **super().config,
            "query_credit": dict.fromkeys(ARMS, "actual_instantaneous_read_path"),
            "value_credit": {
                arm: "fixed_random_context_projection" if arm == "current_tagged"
                else "direct_read_neuron_error" for arm in ARMS
            },
            "current_reference_revision": "f195a36323a4450b40e454e6a0d0df11ba126626",
            "reference_only_buffer": "value_feedback",
        }


SPEC = ValueCreditSpec(
    schema="local-memory-value-credit-v1",
    module="experiments.r4_integrated.local_memory_value_credit_v1",
    entrypoint=Path(__file__).resolve(),
    recipe=Path(__file__).with_suffix(".md").resolve(),
)


def sources(spec=SPEC):
    paths = (
        ROOT / "experiments/r4_integrated/local_memory_query_credit_v1.py",
        ROOT / "experiments/r4_integrated/local_memory_query_credit_v1.md",
        ROOT / "experiments/r4_integrated/local_memory_six_arm_harness.py",
        ROOT / "experiments/r4_integrated/local_memory_value_credit_reference.py",
        ROOT / "docs/research/local-memory-writer-credit-review.md",
        ROOT / "docs/research/local-memory-writer-credit-sources.md",
    )
    return {
        **previous.sources(spec),
        **{p.relative_to(ROOT).as_posix(): file_hash(p) for p in paths},
    }


def new_model(spec, arm):
    if arm not in ARMS:
        raise ValueError("unknown value-credit experiment arm")
    if arm == "current_tagged":
        from experiments.r4_integrated.local_memory_value_credit_reference import LocalMemoryLearner
    else:
        from humanbrain.local_memory_learner import LocalMemoryLearner
    config = spec.config
    return LocalMemoryLearner(
        mode="tagged" if arm == "current_tagged" else arm,
        seed=config["seed"], input_code_size=config["input_code_size"],
        hidden_size=config["hidden_size"], key_size=config["key_size"],
        value_size=config["value_size"],
    ).eval()


EXPERIMENT = harness.Experiment(spec=SPEC, new_model=new_model, sources=sources)


if __name__ == "__main__":
    harness.main(EXPERIMENT)
