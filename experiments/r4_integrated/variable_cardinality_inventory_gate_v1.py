"""Immutable one-shot artifact for learned variable-cardinality emission.

The runner grows one simulated-cell model solely from the frozen public lessons,
freezes every treatment/control/STOP decision from public observations, and only
then opens scoring custody.  The same public fixture is also a focused development
test, so this is causal developmental evidence rather than a novel held-out claim.
Importing this module is inert; invoke ``main`` only once to reserve the artifact.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated.variable_cardinality_inventory_fixture_v1 import (
    BINDING_OFF as FIXTURE_BINDING_OFF,
)
from experiments.r4_integrated.variable_cardinality_inventory_fixture_v1 import (
    DEMONSTRATED_ARITY,
    FIXED_ARITY_LOOP_OFF,
    FRESH_CARDINALITIES,
    FRESH_CASES,
    PHYSICAL_DECISION_CONTRACT,
    PUBLIC_LESSONS,
    STOP_PROBES,
    VARIABLE_CARDINALITY_FIXTURE_SHA256,
    CardinalityStopProbeV1,
    FreshCardinalityCaseV1,
    authenticate_fixture,
)
from humanbrain.factorized_pointer import (
    BINDING_OFF as MODEL_BINDING_OFF,
)
from humanbrain.factorized_pointer import (
    INVENTORY_LOOP_OFF,
    FactorizedCompletionTrace,
    FactorizedPointerCellModel,
    FactorizedPointerTrainingExample,
)
from humanbrain.language import ModelConfig
from humanbrain.pointer_binding import SymbolInventory

SCHEMA = "variable-cardinality-inventory-gate-v1"
RESERVATION_SCHEMA = "variable-cardinality-inventory-gate-reservation-v1"
MAX_NEW_TOKENS = 64
MODEL_CONFIG = ModelConfig(
    max_order=8,
    max_phrases=64,
    min_phrase_frequency=2,
    random_seed=7,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    REPO_ROOT / "benchmarks" / "r4_variable_cardinality_inventory_gate_v1.json"
)
SOURCE_PATHS = {
    "runner_sha256": Path(__file__),
    "fixture_source_sha256": (
        REPO_ROOT
        / "experiments"
        / "r4_integrated"
        / "variable_cardinality_inventory_fixture_v1.py"
    ),
    "factorized_pointer_sha256": (
        REPO_ROOT / "src" / "humanbrain" / "factorized_pointer.py"
    ),
    "cortex_sha256": REPO_ROOT / "src" / "humanbrain" / "cortex.py",
    "language_sha256": REPO_ROOT / "src" / "humanbrain" / "language.py",
    "pointer_binding_sha256": (
        REPO_ROOT / "src" / "humanbrain" / "pointer_binding.py"
    ),
    "tokenizer_sha256": REPO_ROOT / "src" / "humanbrain" / "tokenizer.py",
    "python_executable_sha256": Path(sys.executable),
}


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


def _source_custody() -> dict[str, str]:
    return {name: _sha256_file(path) for name, path in SOURCE_PATHS.items()}


def _authenticate_source_custody(expected: dict[str, str]) -> None:
    actual = _source_custody()
    if actual != expected:
        changed = tuple(
            name for name in SOURCE_PATHS if actual.get(name) != expected.get(name)
        )
        raise RuntimeError(f"gate source custody changed during spend: {changed}")


def _config_record() -> dict[str, object]:
    return {
        "max_order": MODEL_CONFIG.max_order,
        "max_phrases": MODEL_CONFIG.max_phrases,
        "min_phrase_frequency": MODEL_CONFIG.min_phrase_frequency,
        "random_seed": MODEL_CONFIG.random_seed,
        "cell_parameters": asdict(MODEL_CONFIG.cell_parameters),
        "max_new_tokens": MAX_NEW_TOKENS,
    }


def _authenticate_inputs() -> None:
    authenticate_fixture()
    if INVENTORY_LOOP_OFF != FIXED_ARITY_LOOP_OFF:
        raise RuntimeError("fixture and model loop-off ablations disagree")
    if MODEL_BINDING_OFF != FIXTURE_BINDING_OFF:
        raise RuntimeError("fixture and model binding-off ablations disagree")


def _training_examples() -> tuple[FactorizedPointerTrainingExample, ...]:
    examples: list[FactorizedPointerTrainingExample] = []
    for lesson in PUBLIC_LESSONS:
        record = lesson.record()
        inventory = record["inventory"]
        if not isinstance(inventory, tuple):
            raise RuntimeError("public lesson inventory lost tuple custody")
        examples.append(
            FactorizedPointerTrainingExample(
                prompt=str(record["lesson_id"]),
                output=str(record["demonstrated_output"]),
                inventory=SymbolInventory(inventory),
                cue_text=str(record["cue"]),
                provenance_id=str(record["lesson_id"]),
            )
        )
    return tuple(examples)


def _grow_model() -> FactorizedPointerCellModel:
    _authenticate_inputs()
    return FactorizedPointerCellModel.grow(_training_examples(), MODEL_CONFIG)


def _required_trace(model: FactorizedPointerCellModel) -> FactorizedCompletionTrace:
    trace = model.last_trace
    if trace is None:
        raise RuntimeError("completion did not retain a physical trace")
    return trace


def _trace_summary(trace: FactorizedCompletionTrace) -> dict[str, object]:
    physical_fragment_count = sum(
        step.spike_time_ms > 0.0 and step.simulated_steps > 0
        for step in trace.fragment_steps
    )
    physical_loop_count = sum(
        race.spike_time_ms > 0.0 and race.simulated_steps > 0
        for race in trace.inventory_loop_races
    )
    stop_steps = tuple(
        step for step in trace.fragment_steps if step.winner_kind == "stop"
    )
    return {
        "fragment_race_count": len(trace.fragment_steps),
        "physical_fragment_race_count": physical_fragment_count,
        "fragment_winner_kinds": tuple(
            step.winner_kind for step in trace.fragment_steps
        ),
        "inventory_loop_race_count": len(trace.inventory_loop_races),
        "physical_inventory_loop_race_count": physical_loop_count,
        "inventory_loop_item_indices": tuple(
            race.item_index for race in trace.inventory_loop_races
        ),
        "inventory_loop_decisions": tuple(
            race.decision for race in trace.inventory_loop_races
        ),
        "continue_count": sum(
            race.decision == "continue" for race in trace.inventory_loop_races
        ),
        "exit_count": sum(
            race.decision == "exit" for race in trace.inventory_loop_races
        ),
        "stop_winner_count": len(stop_steps),
        "physical_stop_winner_count": sum(
            step.spike_time_ms > 0.0 and step.simulated_steps > 0
            for step in stop_steps
        ),
        "selected_stop": trace.selected_stop,
        "terminal_stop": trace.terminal_stop,
        "physical_custody": bool(
            trace.fragment_steps
            and physical_fragment_count == len(trace.fragment_steps)
            and physical_loop_count == len(trace.inventory_loop_races)
        ),
    }


def _trace_record(trace: FactorizedCompletionTrace) -> dict[str, object]:
    record = {
        "query_sha256": trace.query_sha256,
        "model_digest": trace.model_digest,
        "ablation": trace.ablation,
        "cue_words": trace.cue_words,
        "fragment_steps": tuple(asdict(step) for step in trace.fragment_steps),
        "binding_races": tuple(asdict(race) for race in trace.binding_races),
        "inventory_loop_races": tuple(
            asdict(race) for race in trace.inventory_loop_races
        ),
        "emitted_fragment_cells": trace.emitted_fragment_cells,
        "emitted_token_ids": trace.emitted_token_ids,
        "selected_stop": trace.selected_stop,
        "terminal_stop": trace.terminal_stop,
        "output": trace.output,
        "summary": _trace_summary(trace),
    }
    record["trace_sha256"] = _digest(
        b"variable-cardinality-completion-trace-v1", record
    )
    return record


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
        "trace_output_exact": trace.output == output,
        "physical_custody": _trace_summary(trace)["physical_custody"],
        "trace": _trace_record(trace),
    }


def _observation_query(
    observation: dict[str, object],
    *,
    id_key: str,
) -> tuple[str, SymbolInventory, str]:
    inventory = observation.get("inventory")
    cue = observation.get("cue")
    query_id = observation.get(id_key)
    if (
        not isinstance(inventory, tuple)
        or not isinstance(cue, str)
        or not isinstance(query_id, str)
    ):
        raise RuntimeError("public observation has invalid query custody")
    return query_id, SymbolInventory(inventory), cue


@dataclass(frozen=True, slots=True)
class _FrozenCaseDecision:
    case: FreshCardinalityCaseV1
    observation: dict[str, object]
    treatment_output: str
    treatment_trace: FactorizedCompletionTrace
    loop_off_output: str
    loop_off_trace: FactorizedCompletionTrace
    binding_off_output: str
    binding_off_trace: FactorizedCompletionTrace


@dataclass(frozen=True, slots=True)
class _FrozenStopDecision:
    probe: CardinalityStopProbeV1
    observation: dict[str, object]
    output: str
    trace: FactorizedCompletionTrace


def _freeze_decisions(
    model: FactorizedPointerCellModel,
) -> tuple[tuple[_FrozenCaseDecision, ...], tuple[_FrozenStopDecision, ...]]:
    cases: list[_FrozenCaseDecision] = []
    for case in FRESH_CASES:
        observation = case.public_observation()
        prompt, inventory, cue = _observation_query(observation, id_key="case_id")

        treatment = model.complete(
            prompt,
            inventory,
            cue_text=cue,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        treatment_trace = _required_trace(model)
        loop_off = model.complete(
            prompt,
            inventory,
            cue_text=cue,
            max_new_tokens=MAX_NEW_TOKENS,
            ablation=INVENTORY_LOOP_OFF,
        )
        loop_off_trace = _required_trace(model)
        binding_off = model.complete(
            prompt,
            inventory,
            cue_text=cue,
            max_new_tokens=MAX_NEW_TOKENS,
            ablation=MODEL_BINDING_OFF,
        )
        binding_off_trace = _required_trace(model)
        cases.append(
            _FrozenCaseDecision(
                case=case,
                observation=observation,
                treatment_output=treatment,
                treatment_trace=treatment_trace,
                loop_off_output=loop_off,
                loop_off_trace=loop_off_trace,
                binding_off_output=binding_off,
                binding_off_trace=binding_off_trace,
            )
        )

    stops: list[_FrozenStopDecision] = []
    for probe in STOP_PROBES:
        observation = probe.public_observation()
        prompt, inventory, cue = _observation_query(observation, id_key="probe_id")
        output = model.complete(
            prompt,
            inventory,
            cue_text=cue,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        stops.append(
            _FrozenStopDecision(
                probe=probe,
                observation=observation,
                output=output,
                trace=_required_trace(model),
            )
        )
    return tuple(cases), tuple(stops)


def _decision_freeze_record(
    cases: tuple[_FrozenCaseDecision, ...],
    stops: tuple[_FrozenStopDecision, ...],
) -> dict[str, object]:
    record = {
        "case_decisions": tuple(
            {
                "public_observation": decision.observation,
                "treatment_output": decision.treatment_output,
                "treatment_trace_sha256": _trace_record(
                    decision.treatment_trace
                )["trace_sha256"],
                "fixed_arity_loop_off_output": decision.loop_off_output,
                "fixed_arity_loop_off_trace_sha256": _trace_record(
                    decision.loop_off_trace
                )["trace_sha256"],
                "binding_off_output": decision.binding_off_output,
                "binding_off_trace_sha256": _trace_record(
                    decision.binding_off_trace
                )["trace_sha256"],
            }
            for decision in cases
        ),
        "stop_decisions": tuple(
            {
                "public_observation": decision.observation,
                "output": decision.output,
                "trace_sha256": _trace_record(decision.trace)["trace_sha256"],
            }
            for decision in stops
        ),
    }
    record["decision_freeze_sha256"] = _digest(
        b"variable-cardinality-decision-freeze-v1", record
    )
    return record


def _loop_contract(
    trace: FactorizedCompletionTrace,
    cardinality: int,
) -> dict[str, object]:
    races = trace.inventory_loop_races
    expected_decisions = ("continue",) * (cardinality - 1) + ("exit",)
    expected_indices = tuple(range(cardinality))
    physical_count = sum(
        race.spike_time_ms > 0.0 and race.simulated_steps > 0 for race in races
    )
    decision_winners_match = all(
        race.winner_cell
        == race.candidate_cells[0 if race.decision == "continue" else 1]
        for race in races
        if race.decision in {"continue", "exit"}
    ) and all(race.decision in {"continue", "exit"} for race in races)
    passed = bool(
        tuple(race.decision for race in races) == expected_decisions
        and tuple(race.item_index for race in races) == expected_indices
        and physical_count == cardinality
        and decision_winners_match
    )
    return {
        "passed": passed,
        "cardinality": cardinality,
        "expected_continue_count": cardinality - 1,
        "expected_exit_count": 1,
        "expected_decisions": expected_decisions,
        "observed_decisions": tuple(race.decision for race in races),
        "observed_item_indices": tuple(race.item_index for race in races),
        "physical_race_count": physical_count,
    }


def _score_case(decision: _FrozenCaseDecision) -> dict[str, object]:
    case = decision.case
    inventory = decision.observation["inventory"]
    if not isinstance(inventory, tuple):
        raise RuntimeError("frozen case inventory lost tuple custody")
    treatment_exact = case.accepts_treatment_output(decision.treatment_output)
    loop_off_exact = case.accepts_treatment_output(decision.loop_off_output)
    binding_placeholder_exact = case.accepts_binding_off_output(
        decision.binding_off_output
    )
    binding_target_exact = case.accepts_treatment_output(decision.binding_off_output)
    treatment_loop = _loop_contract(decision.treatment_trace, len(inventory))
    binding_loop = _loop_contract(decision.binding_off_trace, len(inventory))
    return {
        "case_id": decision.observation["case_id"],
        "public_observation": decision.observation,
        "scoring_custody": case.scoring_record(),
        "treatment_loop_contract": treatment_loop,
        "binding_off_loop_contract": binding_loop,
        "treatment": _completion_record(
            decision.treatment_output,
            decision.treatment_trace,
            fixture_exact=treatment_exact,
        ),
        "fixed_arity_loop_off": {
            **_completion_record(
                decision.loop_off_output,
                decision.loop_off_trace,
                fixture_exact=loop_off_exact,
            ),
            "inventory_loop_race_count": len(
                decision.loop_off_trace.inventory_loop_races
            ),
        },
        "binding_off": {
            **_completion_record(
                decision.binding_off_output,
                decision.binding_off_trace,
                fixture_exact=binding_target_exact,
            ),
            "expected_placeholder_output": case.binding_off_output(),
            "placeholder_exact": binding_placeholder_exact,
            "binding_race_count": len(decision.binding_off_trace.binding_races),
        },
    }


def _score_stop(decision: _FrozenStopDecision) -> dict[str, object]:
    trace = decision.trace
    summary = _trace_summary(trace)
    passed = bool(
        decision.probe.accepts_output(decision.output)
        and trace.selected_stop
        and not trace.terminal_stop
        and trace.fragment_steps
        and trace.fragment_steps[-1].winner_kind == "stop"
        and not trace.inventory_loop_races
        and summary["physical_custody"] is True
    )
    return {
        "probe_id": decision.observation["probe_id"],
        "public_observation": decision.observation,
        "scoring_custody": decision.probe.scoring_record(),
        "passed": passed,
        "completion": _completion_record(
            decision.output,
            trace,
            fixture_exact=False,
        ),
    }


def _reservation_record(
    model: FactorizedPointerCellModel,
    source_custody: dict[str, str],
    *,
    status: str,
) -> dict[str, object]:
    config = _config_record()
    public_training = tuple(lesson.record() for lesson in PUBLIC_LESSONS)
    return {
        "schema": RESERVATION_SCHEMA,
        "status": status,
        "artifact_path": ARTIFACT_PATH.relative_to(REPO_ROOT).as_posix(),
        "artifact_absent_at_check": not ARTIFACT_PATH.exists(),
        "fixture_sha256": VARIABLE_CARDINALITY_FIXTURE_SHA256,
        "cohort_novel_at_artifact_spend": False,
        "evaluation_status": "public_developmental_fixture",
        "prior_development_probe": (
            "tests/test_factorized_pointer_inventory_loop.py"
        ),
        "demonstrated_arity": DEMONSTRATED_ARITY,
        "fresh_cardinalities": FRESH_CARDINALITIES,
        "public_training_count": len(PUBLIC_LESSONS),
        "fresh_case_count": len(FRESH_CASES),
        "stop_probe_count": len(STOP_PROBES),
        "public_training_sha256": _digest(
            b"variable-cardinality-public-training-v1", public_training
        ),
        "model_config": config,
        "model_config_sha256": _digest(
            b"variable-cardinality-model-config-v1", config
        ),
        "initial_model": {
            "training_digest": model.training_digest,
            "cell_count": model.cell_count,
            "grown_branches": model.grown_branches,
            "vocabulary_size": model.vocabulary_size,
        },
        "source_custody": source_custody,
        "source_custody_sha256": _digest(
            b"variable-cardinality-source-custody-v1", source_custody
        ),
    }


def preflight() -> dict[str, object]:
    """Authenticate and grow from public lessons without writing or scoring."""

    if ARTIFACT_PATH.exists():
        raise FileExistsError(f"gate artifact already exists: {ARTIFACT_PATH}")
    model = _grow_model()
    custody = _source_custody()
    return _reservation_record(model, custody, status="READY_UNSPENT")


def run_gate(model: FactorizedPointerCellModel) -> dict[str, object]:
    initial_digest = model.training_digest
    frozen_cases, frozen_stops = _freeze_decisions(model)
    decision_freeze = _decision_freeze_record(frozen_cases, frozen_stops)

    # Scoring-only target strings and expected behavior first open here, after
    # every treatment, causal-ablation, binding, and STOP decision is frozen.
    cases = tuple(_score_case(decision) for decision in frozen_cases)
    stops = tuple(_score_stop(decision) for decision in frozen_stops)
    case_count = len(cases)
    decision_outputs = tuple(
        output_and_trace
        for decision in frozen_cases
        for output_and_trace in (
            (decision.treatment_output, decision.treatment_trace),
            (decision.loop_off_output, decision.loop_off_trace),
            (decision.binding_off_output, decision.binding_off_trace),
        )
    ) + tuple((decision.output, decision.trace) for decision in frozen_stops)
    decision_traces = tuple(trace for _, trace in decision_outputs)

    treatment_exact_count = sum(
        case["treatment"]["fixture_exact"] is True for case in cases
    )
    loop_off_exact_count = sum(
        case["fixed_arity_loop_off"]["fixture_exact"] is True for case in cases
    )
    loop_off_zero_race_count = sum(
        case["fixed_arity_loop_off"]["inventory_loop_race_count"] == 0
        for case in cases
    )
    binding_off_placeholder_count = sum(
        case["binding_off"]["placeholder_exact"] is True for case in cases
    )
    binding_off_target_exact_count = sum(
        case["binding_off"]["fixture_exact"] is True for case in cases
    )
    binding_off_zero_binding_race_count = sum(
        case["binding_off"]["binding_race_count"] == 0 for case in cases
    )
    treatment_loop_contract_count = sum(
        case["treatment_loop_contract"]["passed"] is True for case in cases
    )
    binding_loop_contract_count = sum(
        case["binding_off_loop_contract"]["passed"] is True for case in cases
    )
    treatment_terminal_stop_count = sum(
        case["treatment"]["trace"]["terminal_stop"] is True for case in cases
    )
    binding_terminal_stop_count = sum(
        case["binding_off"]["trace"]["terminal_stop"] is True for case in cases
    )
    stop_pass_count = sum(stop["passed"] is True for stop in stops)
    physical_trace_count = sum(
        _trace_summary(trace)["physical_custody"] is True
        and trace.output == output
        for output, trace in decision_outputs
    )
    trace_digest_match_count = sum(
        trace.model_digest == initial_digest for trace in decision_traces
    )
    final_digest = model.training_digest
    expected_physical_traces = case_count * 3 + len(stops)
    passed = bool(
        treatment_exact_count == case_count == 4
        and loop_off_exact_count == 0
        and loop_off_zero_race_count == case_count
        and binding_off_placeholder_count == case_count
        and binding_off_target_exact_count == 0
        and binding_off_zero_binding_race_count == case_count
        and treatment_loop_contract_count == case_count
        and binding_loop_contract_count == case_count
        and treatment_terminal_stop_count == case_count
        and binding_terminal_stop_count == case_count
        and stop_pass_count == len(stops) == 2
        and physical_trace_count == expected_physical_traces
        and trace_digest_match_count == expected_physical_traces
        and final_digest == initial_digest
    )
    return {
        "schema": SCHEMA,
        "status": (
            "PASS_VARIABLE_CARDINALITY_INVENTORY_GATE_V1"
            if passed
            else "FAIL_VARIABLE_CARDINALITY_INVENTORY_GATE_V1"
        ),
        "claim_boundary": (
            "public arbitrary-token microfixture evidence for learned recurrent "
            "variable-cardinality inventory traversal, physical continue/exit/STOP "
            "decisions, learned separators, and pointer binding only; no Python, "
            "novel held-out cohort, integrated V5, continual-learning, SWE, "
            "protected evaluation, or broad intelligence claim"
        ),
        "cohort_novel_at_artifact_spend": False,
        "evaluation_status": "public_developmental_fixture",
        "fixture_sha256": VARIABLE_CARDINALITY_FIXTURE_SHA256,
        "physical_decision_contract": PHYSICAL_DECISION_CONTRACT,
        "decision_freeze": decision_freeze,
        "decisions_frozen_before_scoring": True,
        "initial_model_digest": initial_digest,
        "final_model_digest": final_digest,
        "inference_digest_unchanged": final_digest == initial_digest,
        "scores": {
            "treatment_exact_count": treatment_exact_count,
            "treatment_expected": 4,
            "fixed_arity_loop_off_exact_count": loop_off_exact_count,
            "fixed_arity_loop_off_expected": 0,
            "fixed_arity_loop_off_zero_race_count": loop_off_zero_race_count,
            "binding_off_placeholder_count": binding_off_placeholder_count,
            "binding_off_placeholder_expected": 4,
            "binding_off_target_exact_count": binding_off_target_exact_count,
            "binding_off_zero_binding_race_count": (
                binding_off_zero_binding_race_count
            ),
            "treatment_loop_contract_count": treatment_loop_contract_count,
            "binding_off_loop_contract_count": binding_loop_contract_count,
            "treatment_terminal_stop_count": treatment_terminal_stop_count,
            "binding_off_terminal_stop_count": binding_terminal_stop_count,
            "stop_pass_count": stop_pass_count,
            "stop_expected": 2,
            "physical_trace_count": physical_trace_count,
            "physical_trace_expected": expected_physical_traces,
            "trace_digest_match_count": trace_digest_match_count,
        },
        "sandbox_executions": 0,
        "integrated_task_inputs": 0,
        "persistent_learner_inputs": 0,
        "target_teacher_inputs": 0,
        "private_or_protected_inputs": 0,
        "cases": cases,
        "stop_probes": stops,
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
    source_custody = _source_custody()
    reservation = _reservation_record(model, source_custody, status="RUNNING")
    _write_exclusive(ARTIFACT_PATH, reservation)
    try:
        result = run_gate(model)
        _authenticate_source_custody(source_custody)
        authenticate_fixture()
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
