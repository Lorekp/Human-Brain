"""One-shot public causal gate for the role-relative compositor.

The embedded two-byte fixture recruits a fresh, exact 36-cell GID layout.
Importing this module is inert; run it once with
``python -m experiments.r4_integrated.role_relative_compositor_gate_v1``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

from humanbrain.role_relative_compositor import (
    CompositionAblation,
    PendingConsequenceError,
    RoleRelativeComposition,
    RoleRelativeCompositorConfig,
    RoleRelativeSequenceCompositor,
    RoleSequenceDemonstration,
    Valence,
)

SCHEMA = "r4-role-relative-compositor-gate-v1"
STATUS_PASS = "PASS_R4_ROLE_RELATIVE_COMPOSITOR_GATE_V1"
CLAIM_BOUNDARY = (
    "fresh public opaque-byte microfixture evidence for learned role-relative "
    "two-context composition, fresh query-role binding, physical recurrent and "
    "compositor necessity, fail-closed ambiguity and unknown handling, and one "
    "chain-local target-free negative revision only; no language, Python "
    "execution, persistent session, V6, SWE, protected evaluation, biological "
    "fidelity, or broad capability claim"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "benchmarks" / "r4_role_relative_compositor_gate_v1.json"
MODEL_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "role_relative_compositor.py"
FOCUSED_TEST_PATH = REPO_ROOT / "tests" / "test_role_relative_compositor.py"
CELLS_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "cells.py"
RACE_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "resting_glif_race.py"
EXPECTED_MODEL_SOURCE_SHA256 = "26aa90d8b6b56a9ffb3094bbf45b93dd926d1d6e8e2ee33610ed00883a11e87d"
EXPECTED_FOCUSED_TEST_SHA256 = "b8cee1a6b5d57294e27784006328ec049803521ae84c639ff2cadc1a8831f812"

_FIXTURE_NAMESPACE = b"humanbrain.public-role-relative-fixture-v1"
_RESULT_NAMESPACE = b"humanbrain.public-role-relative-result-v1"
_A = bytes.fromhex("8310")
_B = bytes.fromhex("8320")
_AMBIGUOUS = bytes.fromhex("8330")
_UNKNOWN = bytes.fromhex("83ff")
_DECOYS = tuple(bytes.fromhex(value) for value in ("8401", "8402", "8403"))
_FRESH_ROLES = (bytes.fromhex("d191"), bytes.fromhex("d192"))
_CONFIG = RoleRelativeCompositorConfig(
    max_demonstrations=11,
    max_observation_events=5,
    max_roles=2,
    max_chunk_emissions=3,
    max_event_bytes=2,
    max_cells=36,
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _demo(
    context: bytes,
    tag: int,
    head: str,
    ordinal: int,
    tail: str | None = None,
) -> RoleSequenceDemonstration:
    roles = (bytes((tag, 1)), bytes((tag, 2)))
    literals = (bytes.fromhex(head), roles[ordinal])
    emissions = literals if tail is None else (*literals, bytes.fromhex(tail))
    return RoleSequenceDemonstration(context, roles, emissions)


def _demonstrations() -> tuple[RoleSequenceDemonstration, ...]:
    return (
        _demo(_A, 0x41, "f501", 0, "f502"),
        _demo(_A, 0x42, "f501", 0, "f502"),
        _demo(_A, 0x43, "f501", 0, "f502"),
        _demo(_A, 0x44, "f601", 1, "f602"),
        _demo(_A, 0x45, "f601", 1, "f602"),
        _demo(_B, 0x51, "e501", 1, "e502"),
        _demo(_B, 0x52, "e501", 1, "e502"),
        _demo(_AMBIGUOUS, 0x61, "a501", 0),
        _demo(_AMBIGUOUS, 0x62, "a501", 0),
        _demo(_AMBIGUOUS, 0x63, "b501", 1),
        _demo(_AMBIGUOUS, 0x64, "b501", 1),
    )


def _treatment_observation() -> tuple[bytes, ...]:
    return (_DECOYS[0], _A, _DECOYS[1], _B, _DECOYS[2])


def _expected_treatment() -> tuple[bytes, ...]:
    return tuple(bytes.fromhex(value) for value in ("f501", "d191", "f502", "e501", "d192", "e502"))


def _fixture_record() -> dict[str, object]:
    rows = _demonstrations()
    return {
        "schema": "r4-role-relative-compositor-public-fixture-v1",
        "config": asdict(_CONFIG),
        "demonstrations": tuple(
            {
                "context_hex": row.context_event.hex(),
                "role_inventory_hex": tuple(value.hex() for value in row.role_inventory),
                "emissions_hex": tuple(value.hex() for value in row.emissions),
            }
            for row in rows
        ),
        "treatment_observation_hex": tuple(value.hex() for value in _treatment_observation()),
        "fresh_role_inventory_hex": tuple(value.hex() for value in _FRESH_ROLES),
        "expected_treatment_hex": tuple(value.hex() for value in _expected_treatment()),
        "unknown_observation_hex": _UNKNOWN.hex(),
    }


def _source_custody() -> dict[str, dict[str, str]]:
    paths = {
        "gate_fixture": Path(__file__),
        "model_source": MODEL_SOURCE_PATH,
        "focused_test": FOCUSED_TEST_PATH,
        "cells_source": CELLS_SOURCE_PATH,
        "resting_glif_race_source": RACE_SOURCE_PATH,
    }
    return {
        name: {
            "path": path.relative_to(REPO_ROOT).as_posix(),
            "sha256": _sha256_file(path),
        }
        for name, path in paths.items()
    }


def _authenticate_source_custody(custody: dict[str, dict[str, str]]) -> None:
    expected = {
        "model_source": EXPECTED_MODEL_SOURCE_SHA256,
        "focused_test": EXPECTED_FOCUSED_TEST_SHA256,
    }
    for name, expected_sha256 in expected.items():
        if custody[name]["sha256"] != expected_sha256:
            raise RuntimeError(f"captured {name} hash changed before the one-shot gate")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _expect_pending_error(action: Callable[[], object], fragment: str) -> str:
    try:
        action()
    except PendingConsequenceError as error:
        message = str(error)
        _require(fragment in message, "pending-protocol error identity changed")
        return message
    raise AssertionError("pending-protocol violation did not fail closed")


def _all_physical(result: RoleRelativeComposition) -> bool:
    return (
        bool(result.races)
        and all(race.physical_cell_steps > 0 and race.spike_step >= 0 for race in result.races)
        and sum(race.physical_cell_steps for race in result.races) == result.physical_cell_steps
    )


def _physical_stop(result: RoleRelativeComposition, population: str) -> bool:
    final = result.races[-1]
    return bool(
        result.decision == "STOP"
        and result.emissions == ()
        and result.eligibility is None
        and final.population == population
        and final.stopped
        and final.winner_cell == final.candidate_cells[-1]
        and _all_physical(result)
    )


def _decision_record(result: RoleRelativeComposition) -> dict[str, object]:
    receipt = result.eligibility
    return {
        "decision": result.decision,
        "emissions_hex": tuple(value.hex() for value in result.emissions),
        "stop_reason": result.stop_reason,
        "decision_sha256": result.sha256,
        "eligibility_sha256": None if receipt is None else receipt.sha256,
        "race_count": len(result.races),
        "race_chain_sha256": _digest(
            b"r4-role-relative-race-chain-v1", tuple(row.sha256 for row in result.races)
        ),
        "physical_cell_steps": result.physical_cell_steps,
        "all_races_physical": _all_physical(result),
    }


def run_gate() -> dict[str, object]:
    """Run the fresh public microfalsifier in memory."""

    demonstrations = _demonstrations()
    demonstrated_outputs = {row.emissions for row in demonstrations}
    training_events = {
        event
        for row in demonstrations
        for event in (row.context_event, *row.role_inventory, *row.emissions)
    }
    _require(set(_FRESH_ROLES).isdisjoint(training_events), "fresh roles leaked into training")

    model = RoleRelativeSequenceCompositor.grow(demonstrations, _CONFIG)
    base_digest = model.checkpoint_sha256
    summary = model.training_summary
    _require(
        tuple(asdict(summary).values()) == (11, 3, 5, 13, 10, 5, 46, 109, 36),
        "fresh recruited GID layout changed",
    )

    treatment = model.respond(_treatment_observation(), _FRESH_ROLES)
    _require(treatment.eligibility is not None, "treatment lost eligibility")
    pending_error = _expect_pending_error(
        lambda: model.respond((_A,), _FRESH_ROLES), "resolve the pending"
    )
    neutral = model.resolve(Valence.NEUTRAL)
    duplicate_error = _expect_pending_error(lambda: model.resolve(Valence.NEUTRAL), "no pending")
    post_neutral_digest = model.checkpoint_sha256

    compositor_off = model.respond(
        (_A, _DECOYS[1], _B),
        _FRESH_ROLES,
        ablation=CompositionAblation.COMPOSITOR_OFF,
    )
    recurrence_off = model.respond((_B,), _FRESH_ROLES, ablation=CompositionAblation.RECURRENCE_OFF)
    ambiguous = model.respond((_AMBIGUOUS,), _FRESH_ROLES)
    unknown = model.respond((_UNKNOWN,), _FRESH_ROLES)

    first_revision = model.respond((_A,), _FRESH_ROLES)
    _require(first_revision.eligibility is not None, "revision lost eligibility")
    selected_edge = first_revision.eligibility.transitions[0]
    negative = model.resolve(Valence.NEGATIVE)
    revised = model.respond((_A,), _FRESH_ROLES)
    model.resolve(Valence.NEUTRAL)

    nogo_model = RoleRelativeSequenceCompositor.grow(demonstrations, _CONFIG)
    nogo_digest = nogo_model.checkpoint_sha256
    nogo_first = nogo_model.respond((_A,), _FRESH_ROLES, ablation=CompositionAblation.NOGO_OFF)
    nogo_negative = nogo_model.resolve(Valence.NEGATIVE)
    nogo_repeat = nogo_model.respond((_A,), _FRESH_ROLES, ablation=CompositionAblation.NOGO_OFF)
    nogo_model.resolve(Valence.NEUTRAL)

    transitions = treatment.eligibility.transitions
    checks = {
        "novel_two_context_composition": (
            treatment.decision == "EMIT"
            and treatment.emissions == _expected_treatment()
            and treatment.emissions not in demonstrated_outputs
        ),
        "fresh_role_binding": (
            set(_FRESH_ROLES).issubset(treatment.emissions)
            and set(_FRESH_ROLES).isdisjoint(training_events)
            and [race.winner_cell for race in treatment.races if race.population == "role_binding"]
            == [31, 32]
        ),
        "physical_treatment_chain": (
            len(treatment.races) == len(transitions) == 22
            and treatment.physical_cell_steps == 27_000
            and tuple(edge.target_cell for edge in transitions)
            == tuple(race.winner_cell for race in treatment.races)
            and _all_physical(treatment)
        ),
        "compositor_off_stop": (
            compositor_off.stop_reason == "context_chunk:tonic_stop"
            and compositor_off.races[-1].candidate_scores == (0, 1)
            and _physical_stop(compositor_off, "context_chunk")
        ),
        "recurrence_off_stop": (
            recurrence_off.stop_reason == "state_next:tonic_stop"
            and recurrence_off.races[-1].candidate_scores == (0, 1)
            and _physical_stop(recurrence_off, "state_next")
        ),
        "ambiguous_stop": (
            ambiguous.stop_reason == "context_chunk:collision"
            and ambiguous.races[-1].candidate_scores == (2, 2, 1)
            and ambiguous.races[-1].simultaneous_spike_candidates == 2
            and ambiguous.races[-1].stop_relay_spike_step is not None
            and _physical_stop(ambiguous, "context_chunk")
        ),
        "unknown_stop": (
            unknown.stop_reason == "context_chunk:tonic_stop"
            and unknown.races[-1].candidate_scores == (1,)
            and _physical_stop(unknown, "context_chunk")
        ),
        "pending_resolution_exactly_once": bool(
            pending_error
            and duplicate_error
            and neutral.receipt_sha256 == treatment.eligibility.sha256
        ),
        "neutral_digest_identity": (
            neutral.changed_transitions == ()
            and neutral.prior_training_sha256 == neutral.training_sha256 == base_digest
            and post_neutral_digest == base_digest
        ),
        "target_free_negative_changed_chain": (
            first_revision.emissions
            == (bytes.fromhex("f501"), _FRESH_ROLES[0], bytes.fromhex("f502"))
            and first_revision.races[0].candidate_scores == (3, 2, 1)
            and negative.changed_transitions == (selected_edge,)
            and selected_edge.bank == "context_chunk"
            and selected_edge.target_cell == first_revision.races[0].winner_cell == 6
            and negative.training_sha256 != base_digest
        ),
        "negative_changed_next_winner": (
            revised.emissions == (bytes.fromhex("f601"), _FRESH_ROLES[1], bytes.fromhex("f602"))
            and revised.races[0].candidate_scores == (1, 2, 1)
            and revised.races[0].winner_cell == 7
        ),
        "nogo_off_repeats": (
            nogo_first == nogo_repeat
            and nogo_negative.changed_transitions == ()
            and nogo_negative.prior_training_sha256
            == nogo_negative.training_sha256
            == nogo_model.checkpoint_sha256
            == nogo_digest
        ),
    }
    _require(all(checks.values()), "one or more causal checks failed")

    decisions = {
        "treatment": _decision_record(treatment),
        "compositor_off": _decision_record(compositor_off),
        "recurrence_off": _decision_record(recurrence_off),
        "ambiguous": _decision_record(ambiguous),
        "unknown": _decision_record(unknown),
        "negative_before": _decision_record(first_revision),
        "negative_after": _decision_record(revised),
        "nogo_off_first": _decision_record(nogo_first),
        "nogo_off_repeat": _decision_record(nogo_repeat),
    }
    return {
        "schema": SCHEMA,
        "status": STATUS_PASS,
        "claim_boundary": CLAIM_BOUNDARY,
        "fixture_sha256": _digest(_FIXTURE_NAMESPACE, _fixture_record()),
        "custody": {
            "fixture_visibility": "public",
            "gate_execution_count": 1,
            "model_growth_count": 2,
            "respond_count": len(decisions),
            "effective_negative_resolution_count": 1,
            "nogo_off_negative_control_count": 1,
            "target_supplied_to_negative_resolution": False,
            "target_teacher_inputs": 0,
            "private_or_protected_inputs": 0,
            "v4_or_v6_task_inputs": 0,
            "sandbox_executions": 0,
        },
        "metrics": {
            "checks": checks,
            "training_summary": asdict(summary),
            "demonstration_count": len(demonstrations),
            "demonstrated_whole_output_count": len(demonstrated_outputs),
            "fresh_role_count": len(_FRESH_ROLES),
            "treatment_context_count": 2,
            "treatment_emission_count": len(treatment.emissions),
            "base_digest": base_digest,
            "post_neutral_digest": post_neutral_digest,
            "post_negative_digest": negative.training_sha256,
            "negative_changed_edge_sha256": negative.changed_transitions[0].sha256,
            "negative_before_winner_gid": first_revision.races[0].winner_cell,
            "negative_after_winner_gid": revised.races[0].winner_cell,
            "pending_response_error": pending_error,
            "duplicate_resolution_error": duplicate_error,
        },
        "decisions": decisions,
    }


def _write(path: Path, record: dict[str, object], *, exclusive: bool) -> None:
    mode = "x" if exclusive else "w"
    with path.open(mode, encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    if ARTIFACT_PATH.exists():
        raise FileExistsError(f"one-shot artifact already exists: {ARTIFACT_PATH}")
    source_custody = _source_custody()
    _authenticate_source_custody(source_custody)
    fixture_sha256 = _digest(_FIXTURE_NAMESPACE, _fixture_record())
    reservation = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "claim_boundary": CLAIM_BOUNDARY,
        "fixture_sha256": fixture_sha256,
        "source_custody": source_custody,
        "source_custody_sha256": _digest(b"r4-role-relative-source-custody-v1", source_custody),
    }
    _write(ARTIFACT_PATH, reservation, exclusive=True)
    try:
        result = run_gate()
        _require(result["fixture_sha256"] == fixture_sha256, "fixture custody changed")
    except BaseException as error:
        failure = {
            **reservation,
            "status": "FAIL_R4_ROLE_RELATIVE_COMPOSITOR_GATE_V1",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        failure["result_sha256"] = _digest(_RESULT_NAMESPACE, failure)
        _write(ARTIFACT_PATH, failure, exclusive=False)
        raise

    completed = {
        **result,
        "source_custody": source_custody,
        "source_custody_sha256": reservation["source_custody_sha256"],
        "reservation_status": "COMPLETED",
    }
    completed["result_sha256"] = _digest(_RESULT_NAMESPACE, completed)
    _write(ARTIFACT_PATH, completed, exclusive=False)
    print(f"{STATUS_PASS}: wrote {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
