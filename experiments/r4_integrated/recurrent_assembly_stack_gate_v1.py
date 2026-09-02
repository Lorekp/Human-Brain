"""Immutable one-shot gate for opaque recurrent assembly-stack composition.

``--preflight-schema`` is deliberately inert: it authenticates fixture and
source custody without importing or constructing the model.  ``--run`` is the
only path that can reserve and spend the artifact.  Fresh expected responses
remain scoring-only until every neuronal response and causal control has been
frozen and neutrally resolved.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, is_dataclass
from hashlib import sha256
from pathlib import Path
from typing import NoReturn

from experiments.r4_integrated.recurrent_assembly_stack_fixture_v1 import (
    FRESH_ATOM_CASE,
    FRESH_NESTED_CASES,
    PUBLIC_EXPERIENCES,
    RECURRENT_ASSEMBLY_STACK_FIXTURE_SHA256,
    UNKNOWN_CUE_PROBE,
    FreshAssemblyCaseV1,
    authenticate_fixture,
)

SCHEMA = "recurrent-assembly-stack-gate-v1"
RESERVATION_SCHEMA = "recurrent-assembly-stack-gate-reservation-v1"
EVALUATION_STATUS = "public_developmental_mechanism_fit"
CLAIM_BOUNDARY = (
    "public developmental causal mechanism-fit evidence that a bounded physical "
    "assembly stack is necessary for two visible-scoring, model-training-unseen "
    "three-factor recombinations, with local PUSH/POP/STOP custody and direct-route "
    "retention; the scorer was visible during implementation, so this is not fresh "
    "developer-heldout generalization, Python, persistent integrated V5/V6, "
    "protected evaluation, SWE, cross-domain transfer, or broad intelligence evidence"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = (
    REPO_ROOT / "benchmarks" / "r4_recurrent_assembly_stack_gate_v1.json"
)
SOURCE_PATHS = {
    "runner_sha256": Path(__file__),
    "fixture_source_sha256": (
        REPO_ROOT
        / "experiments"
        / "r4_integrated"
        / "recurrent_assembly_stack_fixture_v1.py"
    ),
    "model_source_sha256": (
        REPO_ROOT / "src" / "humanbrain" / "recurrent_assembly_stack.py"
    ),
    "cortex_source_sha256": REPO_ROOT / "src" / "humanbrain" / "cortex.py",
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
    missing = tuple(
        name for name, path in SOURCE_PATHS.items() if not path.is_file()
    )
    if missing:
        raise FileNotFoundError(f"gate source custody is incomplete: {missing}")
    return {name: _sha256_file(path) for name, path in SOURCE_PATHS.items()}


def _authenticate_source_custody(expected: dict[str, str]) -> None:
    actual = _source_custody()
    if actual != expected:
        changed = tuple(
            name for name in SOURCE_PATHS if actual.get(name) != expected.get(name)
        )
        raise RuntimeError(f"gate source custody changed during spend: {changed}")


def _fixture_counts() -> dict[str, int]:
    return {
        "fresh_atom_case_count": 1,
        "fresh_nested_case_count": len(FRESH_NESTED_CASES),
        "public_experience_count": len(PUBLIC_EXPERIENCES),
        "unknown_cue_probe_count": 1,
    }


def preflight_schema() -> dict[str, object]:
    """Validate/hash the unspent schema without importing the model module."""

    if ARTIFACT_PATH.exists():
        raise FileExistsError(f"gate artifact already exists: {ARTIFACT_PATH}")
    fixture = authenticate_fixture()
    source_custody = _source_custody()
    return {
        "schema": RESERVATION_SCHEMA,
        "status": "READY_SCHEMA_UNSPENT",
        "artifact_absent_at_check": True,
        "artifact_path": ARTIFACT_PATH.relative_to(REPO_ROOT).as_posix(),
        "fixture": fixture,
        "fixture_sha256": RECURRENT_ASSEMBLY_STACK_FIXTURE_SHA256,
        **_fixture_counts(),
        "claim_boundary": CLAIM_BOUNDARY,
        "cohort_novel_at_artifact_spend": False,
        "developmental_mechanism_family_visible": True,
        "fresh_combinations_unseen_by_model_training": True,
        "evaluation_status": EVALUATION_STATUS,
        "protected_evidence": False,
        "target_teacher_inputs": 0,
        "model_imported": False,
        "model_constructed": False,
        "source_custody": source_custody,
        "source_custody_sha256": _digest(
            b"recurrent-assembly-stack-source-custody-v1",
            source_custody,
        ),
    }


def _plain(value: object) -> object:
    """Convert frozen trace values to canonical JSON data."""

    if is_dataclass(value) and not isinstance(value, type):
        return _plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "value"):
        return _plain(value.value)
    raise TypeError(f"trace contains unsupported value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class _FrozenResponse:
    tokens: tuple[int, ...]
    trace: object
    prior_model_digest: str
    resolved_model_digest: str
    consequence_update: object


@dataclass(frozen=True, slots=True)
class _FrozenNestedDecision:
    case: FreshAssemblyCaseV1
    observation: dict[str, object]
    treatment: _FrozenResponse
    stack_off: _FrozenResponse


@dataclass(frozen=True, slots=True)
class _FrozenDirectDecision:
    case: FreshAssemblyCaseV1
    observation: dict[str, object]
    treatment: _FrozenResponse
    stack_off: _FrozenResponse


@dataclass(frozen=True, slots=True)
class _FrozenStopDecision:
    observation: dict[str, object]
    response: _FrozenResponse


@dataclass(frozen=True, slots=True)
class _ModelTypes:
    AssemblyAblation: type
    AssemblyStackConfig: type
    RecurrentAssemblyStack: type
    Valence: type


def _load_model_types() -> _ModelTypes:
    """Import the model only on the explicit irreversible run path."""

    from humanbrain.recurrent_assembly_stack import (
        AssemblyAblation,
        AssemblyStackConfig,
        RecurrentAssemblyStack,
        Valence,
    )

    return _ModelTypes(
        AssemblyAblation=AssemblyAblation,
        AssemblyStackConfig=AssemblyStackConfig,
        RecurrentAssemblyStack=RecurrentAssemblyStack,
        Valence=Valence,
    )


def _token_id(token: str) -> int:
    digest = sha256(
        b"humanbrain-recurrent-assembly-stack-opaque-token-v1\0"
        + token.encode("utf-8")
    ).digest()
    value = int.from_bytes(digest[:4], "big") & 0x7FFF_FFFF
    return value or 1


def _encode_tokens(tokens: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(_token_id(token) for token in tokens)


def _authenticate_token_encoding() -> None:
    # Expected response tokens are intentionally absent.  Every fresh expected
    # structural token is already public training material, while each fresh
    # payload first enters through its public observation.
    public_tokens = tuple(
        token
        for experience in PUBLIC_EXPERIENCES
        for sequence in (
            experience.observation_tokens,
            experience.response_tokens,
        )
        for token in sequence
    ) + tuple(
        token
        for observation in (
            *(case.observation_tokens for case in FRESH_NESTED_CASES),
            FRESH_ATOM_CASE.observation_tokens,
            UNKNOWN_CUE_PROBE.observation_tokens,
        )
        for token in observation
    )
    distinct = set(public_tokens)
    if len({_token_id(token) for token in distinct}) != len(distinct):
        raise RuntimeError("opaque fixture token ids collide")


def _config_record(config: object) -> dict[str, object]:
    record = _plain(config)
    if not isinstance(record, dict):
        raise TypeError("assembly stack config must be a dataclass record")
    return record


def _grow_model() -> tuple[object, object, _ModelTypes, int]:
    authenticate_fixture()
    _authenticate_token_encoding()
    types = _load_model_types()
    config = types.AssemblyStackConfig(
        max_cells=2_048,
        max_examples=len(PUBLIC_EXPERIENCES),
        max_programs=256,
        max_cue_tokens=4,
        max_sequence_tokens=7,
        max_depth=3,
        max_steps=96,
        max_output_tokens=7,
        random_seed=23,
    )
    model = types.RecurrentAssemblyStack(config)
    for experience in PUBLIC_EXPERIENCES:
        model.observe_success(
            _encode_tokens(experience.observation_tokens),
            _encode_tokens(experience.response_tokens),
        )
    return model, config, types, len(PUBLIC_EXPERIENCES)


def _action_name(action: object) -> str:
    value = getattr(action, "value", action)
    return str(value).upper()


def _trace_summary(trace: object) -> dict[str, object]:
    steps = tuple(trace.steps)
    actions = tuple(_action_name(step.operation) for step in steps)
    phases = tuple(str(step.phase) for step in steps)
    depths = tuple(int(step.depth) for step in steps)
    physical_count = sum(
        float(step.spike_time_ms) > 0.0
        and int(step.simulated_steps) > 0
        for step in steps
    )
    return {
        "step_count": len(steps),
        "actions": actions,
        "phases": phases,
        "depths": depths,
        "push_count": actions.count("PUSH"),
        "emit_count": actions.count("EMIT"),
        "pop_count": actions.count("POP"),
        "stop_count": actions.count("STOP"),
        "physical_step_count": physical_count,
        "physical_custody": bool(steps and physical_count == len(steps)),
        "max_depth": int(trace.max_depth),
        "selected_stop": bool(trace.selected_stop),
        "simulated_steps": int(trace.simulated_steps),
    }


def _trace_record(trace: object) -> dict[str, object]:
    record = _plain(trace)
    if not isinstance(record, dict):
        raise TypeError("assembly decision trace must be a dataclass record")
    record = {**record, "summary": _trace_summary(trace)}
    record["trace_sha256"] = _digest(
        b"recurrent-assembly-stack-decision-trace-v1",
        record,
    )
    return record


def _tokens_sha256(tokens: tuple[int, ...]) -> str:
    payload = b"".join(token.to_bytes(4, "big") for token in tokens)
    return sha256(payload).hexdigest()


def _freeze_response(
    model: object,
    types: _ModelTypes,
    observation_tokens: tuple[str, ...],
    *,
    ablation: object,
) -> _FrozenResponse:
    before = str(model.model_digest)
    pending = model.respond(
        _encode_tokens(observation_tokens),
        ablation=ablation,
    )
    tokens = tuple(int(token) for token in pending.tokens)
    trace = pending.trace
    update = pending.resolve(types.Valence.NEUTRAL)
    after = str(model.model_digest)
    if update.valence is not types.Valence.NEUTRAL:
        raise RuntimeError("neutral scoring returned a non-neutral consequence")
    if tuple(update.changed_edges):
        raise RuntimeError("neutral frozen scoring changed structural edges")
    if str(update.prior_digest) != before:
        raise RuntimeError("neutral consequence lost its prior digest custody")
    if str(update.model_digest) != after:
        raise RuntimeError("neutral consequence lost its resolved digest custody")
    if after != before:
        raise RuntimeError("neutral frozen scoring changed the model digest")
    return _FrozenResponse(
        tokens=tokens,
        trace=trace,
        prior_model_digest=before,
        resolved_model_digest=after,
        consequence_update=update,
    )


def _freeze_decisions(
    model: object,
    types: _ModelTypes,
) -> tuple[
    tuple[_FrozenNestedDecision, ...],
    _FrozenDirectDecision,
    _FrozenStopDecision,
]:
    nested: list[_FrozenNestedDecision] = []
    for case in FRESH_NESTED_CASES:
        observation = case.public_observation()
        treatment = _freeze_response(
            model,
            types,
            case.observation_tokens,
            ablation=types.AssemblyAblation.NONE,
        )
        stack_off = _freeze_response(
            model,
            types,
            case.observation_tokens,
            ablation=types.AssemblyAblation.STACK_OFF,
        )
        nested.append(
            _FrozenNestedDecision(
                case=case,
                observation=observation,
                treatment=treatment,
                stack_off=stack_off,
            )
        )

    direct_observation = FRESH_ATOM_CASE.public_observation()
    direct_treatment = _freeze_response(
        model,
        types,
        FRESH_ATOM_CASE.observation_tokens,
        ablation=types.AssemblyAblation.NONE,
    )
    direct_stack_off = _freeze_response(
        model,
        types,
        FRESH_ATOM_CASE.observation_tokens,
        ablation=types.AssemblyAblation.STACK_OFF,
    )
    direct = _FrozenDirectDecision(
        case=FRESH_ATOM_CASE,
        observation=direct_observation,
        treatment=direct_treatment,
        stack_off=direct_stack_off,
    )

    stop_observation = UNKNOWN_CUE_PROBE.public_observation()
    stop_response = _freeze_response(
        model,
        types,
        UNKNOWN_CUE_PROBE.observation_tokens,
        ablation=types.AssemblyAblation.NONE,
    )
    stop = _FrozenStopDecision(
        observation=stop_observation,
        response=stop_response,
    )
    return tuple(nested), direct, stop


def _response_record(response: _FrozenResponse) -> dict[str, object]:
    trace = _trace_record(response.trace)
    consequence = _plain(response.consequence_update)
    if not isinstance(consequence, dict):
        raise TypeError("assembly consequence must be a dataclass record")
    return {
        "emitted_token_ids": response.tokens,
        "emitted_token_bytes_sha256": _tokens_sha256(response.tokens),
        "trace": trace,
        "prior_model_digest": response.prior_model_digest,
        "resolved_model_digest": response.resolved_model_digest,
        "inference_digest_unchanged": (
            response.resolved_model_digest == response.prior_model_digest
        ),
        "neutral_consequence": consequence,
    }


def _score_nested(decision: _FrozenNestedDecision) -> dict[str, object]:
    # First scoring-only target access occurs here, after the full freeze.
    expected_strings = decision.case.scoring_only_expected_tokens
    expected = _encode_tokens(expected_strings)
    treatment = _response_record(decision.treatment)
    stack_off = _response_record(decision.stack_off)
    treatment_summary = treatment["trace"]["summary"]
    if not isinstance(treatment_summary, dict):
        raise TypeError("treatment trace summary lost mapping custody")
    structure_passed = bool(
        int(treatment_summary["max_depth"]) >= 2
        and int(treatment_summary["push_count"]) >= 2
        and int(treatment_summary["pop_count"]) >= 2
        and treatment_summary["physical_custody"] is True
    )
    return {
        "case_id": decision.case.case_id,
        "public_observation": decision.observation,
        "scoring_custody": decision.case.scoring_record(),
        "scoring_only_expected_token_ids": expected,
        "treatment": {
            **treatment,
            "fixture_exact": decision.treatment.tokens == expected,
            "nested_physical_contract": structure_passed,
        },
        "stack_off": {
            **stack_off,
            "fixture_exact": decision.stack_off.tokens == expected,
        },
    }


def _score_direct(decision: _FrozenDirectDecision) -> dict[str, object]:
    expected_strings = decision.case.scoring_only_expected_tokens
    expected = _encode_tokens(expected_strings)
    return {
        "case_id": decision.case.case_id,
        "public_observation": decision.observation,
        "scoring_custody": decision.case.scoring_record(),
        "scoring_only_expected_token_ids": expected,
        "treatment": {
            **_response_record(decision.treatment),
            "fixture_exact": decision.treatment.tokens == expected,
        },
        "stack_off": {
            **_response_record(decision.stack_off),
            "fixture_exact": decision.stack_off.tokens == expected,
        },
    }


def _score_stop(decision: _FrozenStopDecision) -> dict[str, object]:
    response = _response_record(decision.response)
    summary = response["trace"]["summary"]
    if not isinstance(summary, dict):
        raise TypeError("STOP trace summary lost mapping custody")
    passed = bool(
        decision.response.tokens == ()
        and summary["selected_stop"] is True
        and int(summary["stop_count"]) >= 1
        and summary["physical_custody"] is True
    )
    return {
        "probe_id": UNKNOWN_CUE_PROBE.probe_id,
        "public_observation": decision.observation,
        "scoring_custody": UNKNOWN_CUE_PROBE.scoring_record(),
        "response": response,
        "passed": passed,
    }


def _decision_freeze_record(
    nested: tuple[_FrozenNestedDecision, ...],
    direct: _FrozenDirectDecision,
    stop: _FrozenStopDecision,
) -> dict[str, object]:
    record = {
        "nested": tuple(
            {
                "public_observation": decision.observation,
                "treatment_token_ids": decision.treatment.tokens,
                "treatment_trace_sha256": _trace_record(
                    decision.treatment.trace
                )["trace_sha256"],
                "treatment_prior_model_digest": (
                    decision.treatment.prior_model_digest
                ),
                "treatment_resolved_model_digest": (
                    decision.treatment.resolved_model_digest
                ),
                "stack_off_token_ids": decision.stack_off.tokens,
                "stack_off_trace_sha256": _trace_record(
                    decision.stack_off.trace
                )["trace_sha256"],
                "stack_off_prior_model_digest": (
                    decision.stack_off.prior_model_digest
                ),
                "stack_off_resolved_model_digest": (
                    decision.stack_off.resolved_model_digest
                ),
            }
            for decision in nested
        ),
        "direct": {
            "public_observation": direct.observation,
            "treatment_token_ids": direct.treatment.tokens,
            "treatment_trace_sha256": _trace_record(direct.treatment.trace)[
                "trace_sha256"
            ],
            "treatment_prior_model_digest": direct.treatment.prior_model_digest,
            "treatment_resolved_model_digest": (
                direct.treatment.resolved_model_digest
            ),
            "stack_off_token_ids": direct.stack_off.tokens,
            "stack_off_trace_sha256": _trace_record(direct.stack_off.trace)[
                "trace_sha256"
            ],
            "stack_off_prior_model_digest": direct.stack_off.prior_model_digest,
            "stack_off_resolved_model_digest": (
                direct.stack_off.resolved_model_digest
            ),
        },
        "unknown_cue": {
            "public_observation": stop.observation,
            "token_ids": stop.response.tokens,
            "trace_sha256": _trace_record(stop.response.trace)["trace_sha256"],
            "prior_model_digest": stop.response.prior_model_digest,
            "resolved_model_digest": stop.response.resolved_model_digest,
        },
    }
    record["decision_freeze_sha256"] = _digest(
        b"recurrent-assembly-stack-decision-freeze-v1",
        record,
    )
    return record


def _reservation_record(
    model: object,
    config: object,
    source_custody: dict[str, str],
    public_observation_count: int,
    *,
    status: str,
) -> dict[str, object]:
    config_record = _config_record(config)
    public_training = tuple(
        experience.record() for experience in PUBLIC_EXPERIENCES
    )
    return {
        "schema": RESERVATION_SCHEMA,
        "status": status,
        "artifact_path": ARTIFACT_PATH.relative_to(REPO_ROOT).as_posix(),
        "artifact_absent_at_check": not ARTIFACT_PATH.exists(),
        "fixture_sha256": RECURRENT_ASSEMBLY_STACK_FIXTURE_SHA256,
        "cohort_novel_at_artifact_spend": False,
        "developmental_mechanism_family_visible": True,
        "fresh_combinations_unseen_by_model_training": True,
        "evaluation_status": EVALUATION_STATUS,
        "protected_evidence": False,
        "claim_boundary": CLAIM_BOUNDARY,
        "target_teacher_inputs": 0,
        **_fixture_counts(),
        "public_training_sha256": _digest(
            b"recurrent-assembly-stack-public-training-v1",
            public_training,
        ),
        "opaque_token_encoding": (
            "sha256(namespace + utf8_token) first 31 nonzero bits"
        ),
        "model_config": config_record,
        "model_config_sha256": _digest(
            b"recurrent-assembly-stack-model-config-v1",
            config_record,
        ),
        "initial_model": {
            "model_digest": str(model.model_digest),
            "public_success_observation_count": public_observation_count,
        },
        "source_custody": source_custody,
        "source_custody_sha256": _digest(
            b"recurrent-assembly-stack-source-custody-v1",
            source_custody,
        ),
    }


def run_gate(model: object, types: _ModelTypes) -> dict[str, object]:
    initial_digest = str(model.model_digest)
    nested_frozen, direct_frozen, stop_frozen = _freeze_decisions(model, types)
    decision_freeze = _decision_freeze_record(
        nested_frozen,
        direct_frozen,
        stop_frozen,
    )

    # Scoring-only strings first open after every response and neutral resolve.
    nested = tuple(_score_nested(decision) for decision in nested_frozen)
    direct = _score_direct(direct_frozen)
    unknown = _score_stop(stop_frozen)
    treatment_exact_count = sum(
        case["treatment"]["fixture_exact"] is True for case in nested
    )
    treatment_structure_count = sum(
        case["treatment"]["nested_physical_contract"] is True for case in nested
    )
    stack_off_nested_exact_count = sum(
        case["stack_off"]["fixture_exact"] is True for case in nested
    )
    direct_treatment_exact = direct["treatment"]["fixture_exact"] is True
    direct_stack_off_exact = direct["stack_off"]["fixture_exact"] is True
    all_response_records = (
        *(
            response
            for case in nested
            for response in (case["treatment"], case["stack_off"])
        ),
        direct["treatment"],
        direct["stack_off"],
        unknown["response"],
    )
    physical_trace_count = sum(
        response["trace"]["summary"]["physical_custody"] is True
        for response in all_response_records
    )
    inference_digest_match_count = sum(
        response["prior_model_digest"] == initial_digest
        and response["resolved_model_digest"] == initial_digest
        and response["inference_digest_unchanged"] is True
        for response in all_response_records
    )
    neutral_resolution_count = sum(
        response["neutral_consequence"]["valence"] == "neutral"
        and tuple(response["neutral_consequence"]["changed_edges"]) == ()
        and response["neutral_consequence"]["prior_digest"] == initial_digest
        and response["neutral_consequence"]["model_digest"] == initial_digest
        for response in all_response_records
    )
    final_digest = str(model.model_digest)
    passed = bool(
        treatment_exact_count == len(nested) == 2
        and treatment_structure_count == len(nested)
        and stack_off_nested_exact_count == 0
        and direct_treatment_exact
        and direct_stack_off_exact
        and unknown["passed"] is True
        and physical_trace_count == len(all_response_records) == 7
        and inference_digest_match_count == len(all_response_records)
        and neutral_resolution_count == len(all_response_records)
        and final_digest == initial_digest
    )
    return {
        "schema": SCHEMA,
        "status": (
            "PASS_RECURRENT_ASSEMBLY_STACK_GATE_V1"
            if passed
            else "FAIL_RECURRENT_ASSEMBLY_STACK_GATE_V1"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
        "cohort_novel_at_artifact_spend": False,
        "developmental_mechanism_family_visible": True,
        "fresh_combinations_unseen_by_model_training": True,
        "evaluation_status": EVALUATION_STATUS,
        "protected_evidence": False,
        "fixture_sha256": RECURRENT_ASSEMBLY_STACK_FIXTURE_SHA256,
        "decision_freeze": decision_freeze,
        "decisions_frozen_before_scoring": True,
        "neutral_resolution_during_scoring": True,
        "initial_model_digest": initial_digest,
        "final_model_digest": final_digest,
        "inference_digest_unchanged": final_digest == initial_digest,
        "scores": {
            "treatment_exact_count": treatment_exact_count,
            "treatment_expected": 2,
            "treatment_nested_physical_count": treatment_structure_count,
            "treatment_nested_physical_expected": 2,
            "stack_off_nested_exact_count": stack_off_nested_exact_count,
            "stack_off_nested_exact_expected": 0,
            "direct_treatment_exact_count": int(direct_treatment_exact),
            "direct_stack_off_exact_count": int(direct_stack_off_exact),
            "unknown_physical_stop_count": int(unknown["passed"] is True),
            "physical_trace_count": physical_trace_count,
            "physical_trace_expected": len(all_response_records),
            "inference_digest_match_count": inference_digest_match_count,
            "neutral_resolution_count": neutral_resolution_count,
        },
        "sandbox_executions": 0,
        "integrated_task_inputs": 0,
        "persistent_learner_inputs": 0,
        "target_teacher_inputs": 0,
        "private_or_protected_inputs": 0,
        "fresh_nested_cases": nested,
        "fresh_atom_control": direct,
        "unknown_cue_probe": unknown,
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


def _run_and_write() -> int:
    if ARTIFACT_PATH.exists():
        raise FileExistsError(f"gate artifact already exists: {ARTIFACT_PATH}")
    authenticate_fixture()
    _authenticate_token_encoding()
    source_custody = _source_custody()
    model, config, types, observation_count = _grow_model()
    reservation = _reservation_record(
        model,
        config,
        source_custody,
        observation_count,
        status="RUNNING",
    )
    _write_exclusive(ARTIFACT_PATH, reservation)
    try:
        result = run_gate(model, types)
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


def _die(parser: argparse.ArgumentParser, message: str) -> NoReturn:
    parser.error(message)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument(
        "--preflight-schema",
        action="store_true",
        help="validate fixture/source hashes without importing the model",
    )
    action.add_argument(
        "--run",
        action="store_true",
        help="irreversibly reserve and spend the one-shot artifact",
    )
    args = parser.parse_args()
    if not args.preflight_schema and not args.run:
        _die(parser, "one explicit action is required")
    return args


def main() -> int:
    args = _parse_args()
    if args.preflight_schema:
        print(json.dumps(preflight_schema(), indent=2, sort_keys=True))
        return 0
    return _run_and_write()


if __name__ == "__main__":
    raise SystemExit(main())
