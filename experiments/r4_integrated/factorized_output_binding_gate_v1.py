"""Immutable one-shot gate for factorized output construction and fresh binding.

The runner consumes only the frozen public fixture, grows one simulated-cell
model, freezes treatment/control decisions before opening scoring-only fields,
and writes exactly one exclusive artifact.  Importing this module is inert; run
it once with ``python -m experiments.r4_integrated.factorized_output_binding_gate_v1``
to spend the gate.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.factorized_output_binding_fixture_v1 import (
    FACTOR_OUTPUT_BINDING_FIXTURE_SHA256,
    FRESH_CASES,
    PUBLIC_LESSONS,
    FreshCompositionCaseV1,
)
from humanbrain.factorized_pointer import (
    BINDING_OFF,
    FRAGMENT_PROJECTION_OFF,
    STOP_OFF,
    WHOLE_EPISODE_ONLY,
    FactorizedCompletionTrace,
    FactorizedPointerCellModel,
    FactorizedPointerTrainingExample,
)
from humanbrain.language import ModelConfig
from humanbrain.pointer_binding import SymbolInventory

SCHEMA = "factorized-output-binding-gate-v1"
RESERVATION_SCHEMA = "factorized-output-binding-gate-reservation-v1"
MAX_NEW_TOKENS = 64
MODEL_CONFIG = ModelConfig(
    max_order=8,
    max_phrases=64,
    min_phrase_frequency=2,
    random_seed=7,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "benchmarks" / "r4_factorized_output_binding_gate_v1.json"
FIXTURE_SOURCE_PATH = (
    REPO_ROOT
    / "experiments"
    / "r4_integrated"
    / "factorized_output_binding_fixture_v1.py"
)
MODEL_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "factorized_pointer.py"
CORTEX_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "cortex.py"
LANGUAGE_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "language.py"
POINTER_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "pointer_binding.py"
TOKENIZER_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "tokenizer.py"

_OUTPUT = re.compile(
    r"let (?P<local>[A-Za-z_][A-Za-z0-9_]*) = "
    r"(?P<factor_a>[A-Za-z_][A-Za-z0-9_]*)"
    r"\((?P<input>[A-Za-z_][A-Za-z0-9_]*)\); send "
    r"(?P<factor_b>[A-Za-z_][A-Za-z0-9_]*)\((?P=local)\)"
)
_UNKNOWN_PROMPT = "unknown-factor-control"
_UNKNOWN_CUE_TEXT = "opaque unfamiliar"
_UNKNOWN_INPUT_SYMBOL = "novel_unknown"


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _digest(namespace: bytes, value: object) -> str:
    return sha256(
        namespace + b"\0" + _canonical_json(value).encode("ascii")
    ).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _cue_text(cue_a: str, cue_b: str) -> str:
    return f"{cue_a} {cue_b}"


def _config_record() -> dict[str, object]:
    return {
        "max_order": MODEL_CONFIG.max_order,
        "max_phrases": MODEL_CONFIG.max_phrases,
        "min_phrase_frequency": MODEL_CONFIG.min_phrase_frequency,
        "random_seed": MODEL_CONFIG.random_seed,
        "cell_parameters": asdict(MODEL_CONFIG.cell_parameters),
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def _training_examples() -> tuple[FactorizedPointerTrainingExample, ...]:
    return tuple(
        FactorizedPointerTrainingExample(
            prompt=lesson.lesson_id,
            output=lesson.demonstrated_output,
            inventory=SymbolInventory((lesson.input_symbol,)),
            cue_text=_cue_text(lesson.cue_a, lesson.cue_b),
            provenance_id=lesson.lesson_id,
        )
        for lesson in PUBLIC_LESSONS
    )


def _grow_model() -> FactorizedPointerCellModel:
    return FactorizedPointerCellModel.grow(_training_examples(), MODEL_CONFIG)


def _trace_record(trace: FactorizedCompletionTrace) -> dict[str, object]:
    record = {
        "query_sha256": trace.query_sha256,
        "model_digest": trace.model_digest,
        "ablation": trace.ablation,
        "cue_words": trace.cue_words,
        "fragment_steps": tuple(asdict(step) for step in trace.fragment_steps),
        "binding_races": tuple(asdict(race) for race in trace.binding_races),
        "emitted_fragment_cells": trace.emitted_fragment_cells,
        "emitted_token_ids": trace.emitted_token_ids,
        "selected_stop": trace.selected_stop,
        "terminal_stop": trace.terminal_stop,
        "output": trace.output,
    }
    record["trace_sha256"] = _digest(b"factorized-completion-trace-v1", record)
    return record


def _physical_trace(trace: FactorizedCompletionTrace) -> bool:
    return bool(trace.fragment_steps) and all(
        step.spike_time_ms > 0.0 and step.simulated_steps > 0
        for step in trace.fragment_steps
    ) and all(
        race.spike_time_ms > 0.0 and race.simulated_steps > 0
        for race in trace.binding_races
    )


def _required_trace(model: FactorizedPointerCellModel) -> FactorizedCompletionTrace:
    trace = model.last_trace
    if trace is None:
        raise RuntimeError("factorized completion did not retain a physical trace")
    return trace


@dataclass(frozen=True, slots=True)
class _FrozenDecision:
    case: FreshCompositionCaseV1
    observation: dict[str, str]
    treatment_output: str
    treatment_trace: FactorizedCompletionTrace
    whole_episode_output: str
    whole_episode_trace: FactorizedCompletionTrace
    binding_off_output: str
    binding_off_trace: FactorizedCompletionTrace
    projection_off_output: str
    projection_off_trace: FactorizedCompletionTrace
    stop_off_output: str
    stop_off_trace: FactorizedCompletionTrace


def _freeze_decisions(
    model: FactorizedPointerCellModel,
) -> tuple[_FrozenDecision, ...]:
    decisions: list[_FrozenDecision] = []
    for case in FRESH_CASES:
        observation = case.public_observation()
        prompt = observation["case_id"]
        inventory = SymbolInventory((observation["input_symbol"],))
        cues = _cue_text(observation["cue_a"], observation["cue_b"])

        treatment = model.complete(
            prompt,
            inventory,
            cue_text=cues,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        treatment_trace = _required_trace(model)

        whole_episode = model.complete(
            prompt,
            inventory,
            cue_text=cues,
            max_new_tokens=MAX_NEW_TOKENS,
            ablation=WHOLE_EPISODE_ONLY,
        )
        whole_episode_trace = _required_trace(model)

        binding_off = model.complete(
            prompt,
            inventory,
            cue_text=cues,
            max_new_tokens=MAX_NEW_TOKENS,
            ablation=BINDING_OFF,
        )
        binding_off_trace = _required_trace(model)

        projection_off = model.complete(
            prompt,
            inventory,
            cue_text=cues,
            max_new_tokens=MAX_NEW_TOKENS,
            ablation=FRAGMENT_PROJECTION_OFF,
        )
        projection_off_trace = _required_trace(model)

        stop_off = model.complete(
            prompt,
            inventory,
            cue_text=cues,
            max_new_tokens=MAX_NEW_TOKENS,
            ablation=STOP_OFF,
        )
        stop_off_trace = _required_trace(model)

        decisions.append(
            _FrozenDecision(
                case=case,
                observation=observation,
                treatment_output=treatment,
                treatment_trace=treatment_trace,
                whole_episode_output=whole_episode,
                whole_episode_trace=whole_episode_trace,
                binding_off_output=binding_off,
                binding_off_trace=binding_off_trace,
                projection_off_output=projection_off,
                projection_off_trace=projection_off_trace,
                stop_off_output=stop_off,
                stop_off_trace=stop_off_trace,
            )
        )
    return tuple(decisions)


def _completion_record(
    output: str,
    trace: FactorizedCompletionTrace,
    *,
    fixture_exact: bool,
) -> dict[str, object]:
    return {
        "output": output,
        "output_sha256": sha256(output.encode("utf-8")).hexdigest(),
        "fixture_exact": fixture_exact,
        "physical_trace": _physical_trace(trace),
        "trace": _trace_record(trace),
    }


def _score_decision(decision: _FrozenDecision) -> dict[str, object]:
    case = decision.case
    treatment_exact = case.accepts_treatment_output(decision.treatment_output)
    whole_episode_exact = case.accepts_treatment_output(
        decision.whole_episode_output
    )
    binding_off_exact = case.accepts_treatment_output(decision.binding_off_output)
    projection_off_exact = case.accepts_treatment_output(
        decision.projection_off_output
    )
    stop_off_exact = case.accepts_treatment_output(decision.stop_off_output)
    expected_binding_off = case.binding_off_output()

    match = _OUTPUT.fullmatch(decision.treatment_output)
    local_symbol = match.group("local") if match is not None else None
    input_symbol = match.group("input") if match is not None else None
    binding_races = decision.treatment_trace.binding_races
    binding_winner = binding_races[0].symbol if len(binding_races) == 1 else None
    public_symbols = frozenset(decision.observation.values())
    fresh_local_binding = bool(
        treatment_exact
        and local_symbol is not None
        and local_symbol == binding_winner
        and local_symbol not in public_symbols
    )

    return {
        "case_id": decision.observation["case_id"],
        "public_observation": decision.observation,
        "scoring_custody": case.scoring_record(),
        "fresh_local_binding": {
            "passed": fresh_local_binding,
            "local_symbol": local_symbol,
            "input_symbol": input_symbol,
            "local_reused_by_output_grammar": match is not None,
            "binding_race_count": len(binding_races),
            "binding_winner_symbol": binding_winner,
            "local_not_public_symbols": (
                local_symbol is not None and local_symbol not in public_symbols
            ),
        },
        "treatment": _completion_record(
            decision.treatment_output,
            decision.treatment_trace,
            fixture_exact=treatment_exact,
        ),
        "whole_episode_only": _completion_record(
            decision.whole_episode_output,
            decision.whole_episode_trace,
            fixture_exact=whole_episode_exact,
        ),
        "binding_off": {
            **_completion_record(
                decision.binding_off_output,
                decision.binding_off_trace,
                fixture_exact=binding_off_exact,
            ),
            "expected_placeholder_output": expected_binding_off,
            "placeholder_exact": decision.binding_off_output
            == expected_binding_off,
        },
        "projection_off": _completion_record(
            decision.projection_off_output,
            decision.projection_off_trace,
            fixture_exact=projection_off_exact,
        ),
        "stop_off": _completion_record(
            decision.stop_off_output,
            decision.stop_off_trace,
            fixture_exact=stop_off_exact,
        ),
    }


def _unknown_stop_control(model: FactorizedPointerCellModel) -> dict[str, object]:
    inventory = SymbolInventory((_UNKNOWN_INPUT_SYMBOL,))
    stopped = model.complete(
        _UNKNOWN_PROMPT,
        inventory,
        cue_text=_UNKNOWN_CUE_TEXT,
        max_new_tokens=MAX_NEW_TOKENS,
    )
    stopped_trace = _required_trace(model)
    stop_off = model.complete(
        f"{_UNKNOWN_PROMPT}-stop-off",
        inventory,
        cue_text=_UNKNOWN_CUE_TEXT,
        max_new_tokens=MAX_NEW_TOKENS,
        ablation=STOP_OFF,
    )
    stop_off_trace = _required_trace(model)
    passed = bool(
        stopped.startswith("<stop>")
        and stopped_trace.selected_stop
        and stopped_trace.fragment_steps[-1].winner_kind == "stop"
        and not stop_off.startswith("<stop>")
        and not stop_off_trace.selected_stop
        and stopped_trace.model_digest == stop_off_trace.model_digest
        and _physical_trace(stopped_trace)
        and _physical_trace(stop_off_trace)
    )
    return {
        "passed": passed,
        "prompt": _UNKNOWN_PROMPT,
        "cue_text": _UNKNOWN_CUE_TEXT,
        "inventory_sha256": inventory.sha256,
        "stop_enabled": _completion_record(
            stopped,
            stopped_trace,
            fixture_exact=False,
        ),
        "stop_off": _completion_record(
            stop_off,
            stop_off_trace,
            fixture_exact=False,
        ),
    }


def _reservation_record(model: FactorizedPointerCellModel) -> dict[str, object]:
    config = _config_record()
    training_records = tuple(lesson.record() for lesson in PUBLIC_LESSONS)
    return {
        "schema": RESERVATION_SCHEMA,
        "status": "RUNNING",
        "artifact_path": ARTIFACT_PATH.relative_to(REPO_ROOT).as_posix(),
        "fixture_sha256": FACTOR_OUTPUT_BINDING_FIXTURE_SHA256,
        "fixture_training_count": len(PUBLIC_LESSONS),
        "fixture_case_count": len(FRESH_CASES),
        "training_examples_sha256": _digest(
            b"factorized-output-binding-public-training-v1",
            training_records,
        ),
        "model_config": config,
        "model_config_sha256": _digest(
            b"factorized-output-binding-model-config-v1",
            config,
        ),
        "initial_model": {
            "training_digest": model.training_digest,
            "cell_count": model.cell_count,
            "grown_branches": model.grown_branches,
            "vocabulary_size": model.vocabulary_size,
        },
        "source_custody": {
            "gate_sha256": _sha256_file(Path(__file__)),
            "fixture_source_sha256": _sha256_file(FIXTURE_SOURCE_PATH),
            "factorized_pointer_sha256": _sha256_file(MODEL_SOURCE_PATH),
            "cortex_sha256": _sha256_file(CORTEX_SOURCE_PATH),
            "language_sha256": _sha256_file(LANGUAGE_SOURCE_PATH),
            "pointer_binding_sha256": _sha256_file(POINTER_SOURCE_PATH),
            "tokenizer_sha256": _sha256_file(TOKENIZER_SOURCE_PATH),
            "python_executable_sha256": _sha256_file(Path(sys.executable)),
        },
    }


def run_gate(model: FactorizedPointerCellModel) -> dict[str, object]:
    initial_digest = model.training_digest
    decisions = _freeze_decisions(model)
    unknown_stop = _unknown_stop_control(model)

    # Scoring-only fixture fields are first opened here, after every treatment
    # and causal-control decision above has been frozen.
    cases = tuple(_score_decision(decision) for decision in decisions)
    case_count = len(FRESH_CASES)
    treatment_exact_count = sum(bool(case["treatment"]["fixture_exact"]) for case in cases)
    fresh_local_binding_count = sum(
        bool(case["fresh_local_binding"]["passed"]) for case in cases
    )
    whole_episode_stop_count = sum(
        bool(case["whole_episode_only"]["trace"]["selected_stop"])
        for case in cases
    )
    whole_episode_exact_count = sum(
        bool(case["whole_episode_only"]["fixture_exact"]) for case in cases
    )
    whole_episode_single_step_count = sum(
        len(case["whole_episode_only"]["trace"]["fragment_steps"]) == 1
        for case in cases
    )
    binding_off_placeholder_count = sum(
        bool(case["binding_off"]["placeholder_exact"]) for case in cases
    )
    binding_off_exact_count = sum(
        bool(case["binding_off"]["fixture_exact"]) for case in cases
    )
    binding_off_binding_race_count = sum(
        len(case["binding_off"]["trace"]["binding_races"]) for case in cases
    )
    projection_off_stop_count = sum(
        bool(case["projection_off"]["trace"]["selected_stop"])
        for case in cases
    )
    projection_off_exact_count = sum(
        bool(case["projection_off"]["fixture_exact"]) for case in cases
    )
    stop_off_exact_count = sum(
        bool(case["stop_off"]["fixture_exact"]) for case in cases
    )
    treatment_terminal_stop_count = sum(
        bool(case["treatment"]["trace"]["terminal_stop"]) for case in cases
    )
    stop_off_terminal_stop_count = sum(
        bool(case["stop_off"]["trace"]["terminal_stop"]) for case in cases
    )
    physical_trace_count = sum(
        bool(case[arm]["physical_trace"])
        for case in cases
        for arm in (
            "treatment",
            "whole_episode_only",
            "binding_off",
            "projection_off",
            "stop_off",
        )
    )
    final_digest = model.training_digest
    passed = bool(
        treatment_exact_count == case_count
        and fresh_local_binding_count == case_count
        and whole_episode_stop_count == case_count
        and whole_episode_exact_count == 0
        and whole_episode_single_step_count == case_count
        and binding_off_placeholder_count == case_count
        and binding_off_exact_count == 0
        and binding_off_binding_race_count == 0
        and projection_off_stop_count == case_count
        and projection_off_exact_count == 0
        and treatment_terminal_stop_count == case_count
        and stop_off_terminal_stop_count == 0
        and physical_trace_count == case_count * 5
        and unknown_stop["passed"] is True
        and final_digest == initial_digest
    )
    return {
        "schema": SCHEMA,
        "status": (
            "PASS_FACTORIZED_OUTPUT_BINDING_GATE_V1"
            if passed
            else "FAIL_FACTORIZED_OUTPUT_BINDING_GATE_V1"
        ),
        "claim_boundary": (
            "public arbitrary-token microfixture evidence for neuron-raced "
            "two-factor output construction, query-symbol projection, fresh local "
            "binding, and explicit STOP only; no Python execution, V4 task input, "
            "persistent continual-learning, SWE, protected evaluation, or broad "
            "intelligence claim"
        ),
        "fixture_sha256": FACTOR_OUTPUT_BINDING_FIXTURE_SHA256,
        "case_count": case_count,
        "training_example_count": len(PUBLIC_LESSONS),
        "initial_model_digest": initial_digest,
        "final_model_digest": final_digest,
        "inference_digest_unchanged": final_digest == initial_digest,
        "scores": {
            "treatment_exact_count": treatment_exact_count,
            "fresh_local_binding_count": fresh_local_binding_count,
            "whole_episode_stop_count": whole_episode_stop_count,
            "whole_episode_exact_count": whole_episode_exact_count,
            "whole_episode_single_step_count": whole_episode_single_step_count,
            "binding_off_placeholder_count": binding_off_placeholder_count,
            "binding_off_exact_count": binding_off_exact_count,
            "binding_off_binding_race_count": binding_off_binding_race_count,
            "projection_off_stop_count": projection_off_stop_count,
            "projection_off_exact_count": projection_off_exact_count,
            "stop_off_exact_count": stop_off_exact_count,
            "treatment_terminal_stop_count": treatment_terminal_stop_count,
            "stop_off_terminal_stop_count": stop_off_terminal_stop_count,
            "physical_trace_count": physical_trace_count,
            "physical_trace_expected": case_count * 5,
        },
        "unknown_stop_control": unknown_stop,
        "sandbox_executions": 0,
        "v4_task_inputs": 0,
        "persistent_learner_inputs": 0,
        "target_teacher_inputs": 0,
        "private_or_protected_inputs": 0,
        "cases": cases,
    }


def _write_exclusive(path: Path, record: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()


def _write_terminal(path: Path, record: dict[str, object]) -> None:
    path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    model = _grow_model()
    reservation = _reservation_record(model)
    _write_exclusive(ARTIFACT_PATH, reservation)
    try:
        result = run_gate(model)
    except BaseException as error:
        failed = dict(reservation)
        failed["status"] = "FAILED"
        failed["error_type"] = type(error).__name__
        failed["error"] = str(error)
        _write_terminal(ARTIFACT_PATH, failed)
        raise
    completed_reservation = dict(reservation)
    completed_reservation["status"] = "COMPLETED"
    result["reservation"] = completed_reservation
    _write_terminal(ARTIFACT_PATH, result)
    print(f"wrote {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
