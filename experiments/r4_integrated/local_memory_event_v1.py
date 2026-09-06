"""One fixed byte-event neuronal encoder versus the immediate byte-input learner."""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from experiments.r4_integrated import local_memory_multiscale_v1 as previous  # noqa: E402
from experiments.r4_integrated import local_memory_six_arm_harness as harness  # noqa: E402

ARMS = previous.ARMS
file_hash, write_json = previous.file_hash, previous.write_json
duration_clock, validate_cost_launch = previous.duration_clock, previous.validate_cost_launch
roster, teaching_schedule = previous.roster, previous.teaching_schedule
project_cost, qualifies = previous.project_cost, previous.qualifies
detached_state, digest = previous.detached_state, previous.digest


@dataclass(frozen=True)
class EventSpec(previous.MultiscaleSpec):
    @property
    def config(self):
        config = super().config
        del config["candidate_only_buffer"]
        return {
            **config,
            "current_reference_revision": "a1ab5ee0d82c5cc047392ec8e6b22be3a4386438",
            "candidate_only_initial_tensor_keys": [
                "event_encoder.weight", "event_encoder.recurrent", "event_encoder.bias"],
            "writer_update_fraction": {
                arm: [0.5] * 32 + [0.125] * 16 + [0.03125] * 16 for arm in ARMS},
            "writer_timescale": dict.fromkeys(ARMS, "fixed_multiscale_32_16_16"),
            "input_encoding": {
                arm: "fixed_byte_codes" if arm == "current_tagged"
                else "fixed_rate_cell_byte_events" for arm in ARMS},
            "event_encoder": {"hidden_size": 32, "seed_offset": 4, "update_fraction": 0.5,
                              "cap_bytes": 16,
                              "policy": "ascii_word_whitespace_runs_other_singletons"},
        }


SPEC = EventSpec(
    schema="local-memory-event-v1", module="experiments.r4_integrated.local_memory_event_v1",
    entrypoint=Path(__file__).resolve(), recipe=Path(__file__).with_suffix(".md").resolve())


def sources(spec=SPEC):
    paths = (
        ROOT / "experiments/r4_integrated/local_memory_multiscale_v1.py",
        ROOT / "experiments/r4_integrated/local_memory_multiscale_v1.md",
        ROOT / "experiments/r4_integrated/local_memory_event_reference.py",
        ROOT / "src/humanbrain/byte_events.py",
        ROOT / "docs/research/local-token-event-interface.md",
    )
    return {**previous.sources(spec),
            **{path.relative_to(ROOT).as_posix(): file_hash(path) for path in paths}}


def new_model(spec, arm):
    if arm not in ARMS:
        raise ValueError("unknown event experiment arm")
    if arm == "current_tagged":
        from experiments.r4_integrated.local_memory_event_reference import LocalMemoryLearner
    else:
        from humanbrain.local_memory_learner import LocalMemoryLearner
    config = spec.config
    return LocalMemoryLearner(
        mode="tagged" if arm == "current_tagged" else arm, seed=config["seed"],
        input_code_size=config["input_code_size"], hidden_size=config["hidden_size"],
        key_size=config["key_size"], value_size=config["value_size"]).eval()


EXPERIMENT = harness.Experiment(spec=SPEC, new_model=new_model, sources=sources)


if __name__ == "__main__":
    harness.main(EXPERIMENT)
