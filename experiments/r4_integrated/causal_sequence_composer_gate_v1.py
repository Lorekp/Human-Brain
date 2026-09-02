"""Prepared one-shot gate for opaque causal-sequence composition.

The frozen public fixture was authored independently of the learner.  This
runner adapts its bytes to reversible opaque text tokens, grows one neuronal
composer, and checks novel composition, physical temporal custody, abstention,
exactly-once consequences, and context-local NoGo learning.

Importing or calling :func:`preflight` is inert.  The gate can execute only
after :func:`main` has exclusively created its ``RUNNING`` artifact.  A process
interruption therefore leaves a spent ``RUNNING`` reservation rather than an
apparently reusable cohort.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated import causal_sequence_composer_fixture_v1 as fixture
from humanbrain.causal_sequence_pointer import (
    CausalSequenceAblation,
    CausalSequenceCompletionTrace,
    CausalSequenceConfig,
    CausalSequenceLearningUpdate,
    CausalSequencePointerModel,
    CausalSequenceTrainingExample,
)
from humanbrain.pointer_binding import SymbolInventory, materialize

SCHEMA = "r4-causal-sequence-composer-gate-v1"
STATUS_PASS = "PASS_R4_CAUSAL_SEQUENCE_COMPOSER_GATE_V1"
STATUS_FAIL = "FAIL_R4_CAUSAL_SEQUENCE_COMPOSER_GATE_V1"
CLAIM_BOUNDARY = (
    "fresh public opaque two-factor sequence-composition microfixture only; "
    "no language understanding, Python semantics, repository coding, persistent "
    "session, SWE, protected evaluation, biological fidelity, or broad capability"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO_ROOT / "benchmarks" / "r4_causal_sequence_composer_gate_v1.json"
FIXTURE_SOURCE_PATH = (
    REPO_ROOT / "experiments" / "r4_integrated" / "causal_sequence_composer_fixture_v1.py"
)
MODEL_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "causal_sequence_pointer.py"
POINTER_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "pointer_binding.py"
TOKENIZER_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "tokenizer.py"
CORTEX_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "cortex.py"
CELLS_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "cells.py"
CELL_IDENTITY_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "human_cell_identity.py"
RACE_BIAS_SOURCE_PATH = REPO_ROOT / "src" / "humanbrain" / "race_bias.py"

EXPECTED_FIXTURE_SOURCE_SHA256 = (
    "135e522aa08bd568237a3414da3993ddc637b397c0e7c654fb8251faa15cabe8"
)
EXPECTED_MODEL_SOURCE_SHA256 = (
    "0c7eb90d253943b13f8670696224ef9747859c0fc44daf374e2685a8cd6a5116"
)
EXPECTED_POINTER_SOURCE_SHA256 = (
    "0cf13bd13be981c27d1eb00c1feb3fa08039c543654b17b02fa1b0e07e38171d"
)
EXPECTED_TOKENIZER_SOURCE_SHA256 = (
    "08ef114a758006868944ad3091ca66619d3c082086ea2d44d93211d82a9791ce"
)
EXPECTED_CORTEX_SOURCE_SHA256 = (
    "12438277d72b80c9a5ccab3a056ea2b967392a3b027ef770704213fa42294411"
)
EXPECTED_CELLS_SOURCE_SHA256 = (
    "5ca05a64a89f930db3ea351e3e236e3242bf21652fff127747a36d9ca488d2e6"
)
EXPECTED_CELL_IDENTITY_SOURCE_SHA256 = (
    "0050670ce4976c6163bc652eb33846d7118d6e2ca047eeeda1cf334efb8cc6f9"
)
EXPECTED_RACE_BIAS_SOURCE_SHA256 = (
    "8d9dff4b65a28447840429d688d551d6b2bb8f963b97dcd95e0420c8c33a0100"
)
EXPECTED_FIXTURE_PAYLOAD_SHA256 = (
    "28a0801a7e9b50e6235c6a793c9481c186e056ba0e8b48a3c332e31539c18e56"
)
EXPECTED_FIXTURE_CUSTODY_SHA256 = (
    "c98abb519bbd6a4cc81b291cb1c45d39f21b991752353cdd2f980f4a5af40439"
)

_SOURCE_CUSTODY_NAMESPACE = b"r4-causal-sequence-source-custody-v1"
_RESULT_NAMESPACE = b"r4-causal-sequence-result-v1"
_RESERVATION_NAMESPACE = b"r4-causal-sequence-reservation-v1"
_TRACE_NAMESPACE = b"r4-causal-sequence-trace-v1"
_NEGATIVE_RECEIPT = "receipt-n0"
_CONFIG = CausalSequenceConfig(random_seed=43)


@dataclass(frozen=True, slots=True)
class _GateCase:
    prompt: str
    cue_text: str
    inventory: SymbolInventory
    expected_output: str | None


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _role_symbol(role: bytes) -> str:
    return f"r{role.hex()}"


def _cue_text(events: tuple[bytes, ...]) -> str:
    return " ".join(f"q{event.hex()}" for event in events)


def _render_emissions(emissions: fixture.Emission) -> str:
    atoms: list[str] = []
    for chunk, role in emissions:
        atoms.extend((f"c{chunk.hex()}", _role_symbol(role)))
    return " ".join(atoms)


def _inventory(roles: tuple[bytes, bytes]) -> SymbolInventory:
    return SymbolInventory(tuple(_role_symbol(role) for role in roles))


def _training_examples() -> tuple[CausalSequenceTrainingExample, ...]:
    return tuple(
        CausalSequenceTrainingExample(
            prompt=_cue_text(lesson.cue_events),
            output=_render_emissions(lesson.demonstrated_emissions),
            inventory=_inventory(lesson.role_inventory),
            cue_text=_cue_text(lesson.cue_events),
            provenance_id=lesson.record_sha256,
        )
        for lesson in fixture.TRAINING_LESSONS
    )


def _case(observation: fixture.CausalSequenceObservationV1) -> _GateCase:
    expected = fixture.scoring_emissions(observation)
    text = _cue_text(observation.cue_events)
    return _GateCase(
        prompt=text,
        cue_text=text,
        inventory=_inventory(observation.role_inventory),
        expected_output=None if expected is None else _render_emissions(expected),
    )


def _source_custody() -> dict[str, dict[str, str]]:
    paths = {
        "gate_runner": Path(__file__),
        "fixture_source": FIXTURE_SOURCE_PATH,
        "model_source": MODEL_SOURCE_PATH,
        "pointer_inventory_source": POINTER_SOURCE_PATH,
        "tokenizer_source": TOKENIZER_SOURCE_PATH,
        "cortex_source": CORTEX_SOURCE_PATH,
        "cells_source": CELLS_SOURCE_PATH,
        "cell_identity_source": CELL_IDENTITY_SOURCE_PATH,
        "race_bias_source": RACE_BIAS_SOURCE_PATH,
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
        "fixture_source": EXPECTED_FIXTURE_SOURCE_SHA256,
        "model_source": EXPECTED_MODEL_SOURCE_SHA256,
        "pointer_inventory_source": EXPECTED_POINTER_SOURCE_SHA256,
        "tokenizer_source": EXPECTED_TOKENIZER_SOURCE_SHA256,
        "cortex_source": EXPECTED_CORTEX_SOURCE_SHA256,
        "cells_source": EXPECTED_CELLS_SOURCE_SHA256,
        "cell_identity_source": EXPECTED_CELL_IDENTITY_SOURCE_SHA256,
        "race_bias_source": EXPECTED_RACE_BIAS_SOURCE_SHA256,
    }
    for name, expected_sha256 in expected.items():
        _require(
            custody[name]["sha256"] == expected_sha256,
            f"captured {name} hash changed before the one-shot gate",
        )


def _authenticate_fixture() -> None:
    fixture.authenticate_fixture()
    _require(
        fixture.CAUSAL_SEQUENCE_PAYLOAD_SHA256 == EXPECTED_FIXTURE_PAYLOAD_SHA256,
        "fixture payload root changed",
    )
    _require(
        fixture.CAUSAL_SEQUENCE_CUSTODY_SHA256 == EXPECTED_FIXTURE_CUSTODY_SHA256,
        "fixture scoring custody root changed",
    )


def _fixture_inputs() -> tuple[
    tuple[CausalSequenceTrainingExample, ...],
    _GateCase,
    tuple[_GateCase, ...],
]:
    examples = _training_examples()
    held_out = _case(fixture.HELD_OUT_CASE)
    probes = tuple(_case(probe) for probe in fixture.PROBES)
    _require(len(examples) == 3, "training grid size changed")
    _require(held_out.expected_output is not None, "held-out scoring sequence vanished")
    _require(
        all(probe.expected_output is None for probe in probes),
        "unresolved probe unexpectedly acquired a scoring sequence",
    )

    demonstrated_outputs = {example.output for example in examples}
    training_roles = {
        symbol for example in examples for symbol in example.inventory.symbols
    }
    _require(
        held_out.expected_output not in demonstrated_outputs,
        "held-out whole sequence leaked into training",
    )
    _require(
        set(held_out.inventory.symbols).isdisjoint(training_roles),
        "held-out role inventory is not fresh",
    )
    return examples, held_out, probes


def preflight() -> dict[str, object]:
    """Authenticate the unspent cohort without growing or invoking a model."""

    if ARTIFACT_PATH.exists():
        raise FileExistsError(f"one-shot artifact already exists: {ARTIFACT_PATH}")
    _authenticate_fixture()
    examples, held_out, probes = _fixture_inputs()
    source_custody = _source_custody()
    _authenticate_source_custody(source_custody)
    return {
        "schema": SCHEMA,
        "artifact_absent": True,
        "fixture_payload_sha256": EXPECTED_FIXTURE_PAYLOAD_SHA256,
        "fixture_custody_sha256": EXPECTED_FIXTURE_CUSTODY_SHA256,
        "source_custody": source_custody,
        "source_custody_sha256": _digest(
            _SOURCE_CUSTODY_NAMESPACE,
            source_custody,
        ),
        "demonstration_count": len(examples),
        "query_count": 1 + len(probes),
        "held_out_output_absent_from_training": held_out.expected_output
        not in {example.output for example in examples},
        "model_growth_count": 0,
        "model_inference_count": 0,
    }


def _expect_pending_error(action: Callable[[], object]) -> str:
    try:
        action()
    except RuntimeError as error:
        return f"{type(error).__name__}: {error}"
    raise RuntimeError("pending consequence was not enforced exactly once")


def _take_trace(model: CausalSequencePointerModel) -> CausalSequenceCompletionTrace:
    trace = model.last_trace
    if trace is None:
        raise RuntimeError("model response did not expose a public completion trace")
    return trace


def _complete(
    model: CausalSequencePointerModel,
    case: _GateCase,
) -> tuple[str, CausalSequenceCompletionTrace]:
    raw_output = model.complete(
        case.prompt,
        case.inventory,
        cue_text=case.cue_text,
        max_new_tokens=32,
    )
    return materialize(raw_output, case.inventory), _take_trace(model)


def _raw_complete(
    model: CausalSequencePointerModel,
    case: _GateCase,
) -> tuple[str, CausalSequenceCompletionTrace]:
    output = model.raw_complete(
        case.prompt,
        case.inventory,
        cue_text=case.cue_text,
        max_new_tokens=32,
    )
    trace = _take_trace(model)
    _require(output == trace.raw_output, "raw output and trace custody diverged")
    return output, trace


def _resolve_neutral(
    model: CausalSequencePointerModel,
    case: _GateCase,
) -> CausalSequenceLearningUpdate:
    return model.resolve_pending_neutral(
        case.prompt,
        case.inventory,
        case.cue_text,
    )


def _physical_backward_trace(trace: CausalSequenceCompletionTrace) -> bool:
    if not trace.steps or trace.selected_stop == trace.selected_eos:
        return False
    for expected_index, step in enumerate(trace.steps):
        candidate_cells = tuple(cell for cell, _current in step.candidate_currents_pa)
        currents = tuple(current for _cell, current in step.candidate_currents_pa)
        dependencies = step.depends_on_steps
        if not (
            step.index == expected_index
            and step.winner_cell in candidate_cells
            and bool(step.winner_operation)
            and math.isfinite(step.spike_time_ms)
            and step.spike_time_ms >= 0.0
            and candidate_cells
            and len(candidate_cells) == len(set(candidate_cells))
            and all(math.isfinite(current) for current in currents)
            and dependencies == tuple(sorted(set(dependencies)))
            and all(0 <= dependency < step.index for dependency in dependencies)
        ):
            return False
    terminal = trace.steps[-1].winner_operation
    return terminal == ("STOP" if trace.selected_stop else "EOS")


def _physical_stop(trace: CausalSequenceCompletionTrace) -> bool:
    return bool(
        trace.selected_stop
        and not trace.selected_eos
        and trace.steps
        and trace.steps[-1].winner_operation == "STOP"
        and _physical_backward_trace(trace)
    )


def _physical_success(trace: CausalSequenceCompletionTrace) -> bool:
    return bool(
        trace.selected_eos
        and not trace.selected_stop
        and trace.steps
        and trace.steps[-1].winner_operation == "EOS"
        and _physical_backward_trace(trace)
    )


def _trace_record(trace: CausalSequenceCompletionTrace) -> dict[str, object]:
    steps = tuple(
        {
            "index": step.index,
            "phase": step.phase,
            "winner_cell": step.winner_cell,
            "winner_operation": step.winner_operation,
            "spike_time_ms": step.spike_time_ms,
            "candidate_currents_pa": step.candidate_currents_pa,
            "depends_on_steps": step.depends_on_steps,
        }
        for step in trace.steps
    )
    body = {
        "query_sha256": trace.query_sha256,
        "prior_training_digest": trace.prior_training_digest,
        "ablation": trace.ablation.value,
        "observation_state": trace.observation_state,
        "emitted_token_ids": trace.emitted_token_ids,
        "raw_output": trace.raw_output,
        "selected_stop": trace.selected_stop,
        "selected_eos": trace.selected_eos,
        "steps": steps,
        "physical_backward_only": _physical_backward_trace(trace),
    }
    return {**body, "trace_sha256": _digest(_TRACE_NAMESPACE, body)}


def _update_record(update: CausalSequenceLearningUpdate) -> dict[str, object]:
    return {
        "polarity": update.polarity,
        "evidence_sha256": update.evidence_sha256,
        "changed_synapses": update.changed_synapses,
        "observed_success_sequence": update.observed_success_sequence,
        "prior_training_digest": update.prior_training_digest,
        "training_digest": update.training_digest,
    }


def _negative_step_index(update: CausalSequenceLearningUpdate) -> int:
    nogo_edges = tuple(
        edge for edge in update.changed_synapses if edge.startswith("nogo:")
    )
    consequence_edges = tuple(
        edge for edge in update.changed_synapses if edge.startswith("consequence:")
    )
    _require(
        len(nogo_edges) == 1
        and len(consequence_edges) == 1
        and len(update.changed_synapses) == 2,
        "negative receipt did not change exactly one NoGo branch and one receipt state",
    )
    fields = nogo_edges[0].split(":", 4)
    _require(
        len(fields) == 5 and fields[0] == "nogo",
        "negative receipt did not identify one public NoGo branch",
    )
    try:
        return int(fields[2])
    except ValueError as error:
        raise RuntimeError("NoGo branch carried an invalid step index") from error


def _authenticate_running_reservation(expected_sha256: str) -> None:
    record = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    supplied_sha256 = record.pop("reservation_sha256", None)
    _require(record.get("status") == "RUNNING", "gate has no live RUNNING reservation")
    actual_sha256 = _digest(_RESERVATION_NAMESPACE, record)
    _require(
        supplied_sha256 == expected_sha256 == actual_sha256,
        "RUNNING reservation custody changed",
    )


def _run_reserved_gate(reservation_sha256: str) -> dict[str, object]:
    """Spend the cohort only behind an authenticated exclusive reservation."""

    _authenticate_running_reservation(reservation_sha256)
    _authenticate_fixture()
    examples, held_out, probes = _fixture_inputs()
    _require(len(probes) == 2, "ambiguous/unknown query count changed")

    demonstrated_outputs = {example.output for example in examples}
    expected_held_out = held_out.expected_output
    _require(expected_held_out is not None, "held-out output unavailable to scorer")

    model = CausalSequencePointerModel.grow(examples, _CONFIG)
    base_digest = model.training_digest
    treatment_output, treatment_trace = _complete(model, held_out)
    pending_error = _expect_pending_error(lambda: _complete(model, held_out))
    neutral = _resolve_neutral(model, held_out)
    duplicate_resolution_error = _expect_pending_error(
        lambda: _resolve_neutral(model, held_out)
    )
    post_neutral_digest = model.training_digest

    _ambiguous_raw, ambiguous_trace = _raw_complete(model, probes[0])
    ambiguous_no_pending = _expect_pending_error(lambda: _resolve_neutral(model, probes[0]))
    _unknown_raw, unknown_trace = _raw_complete(model, probes[1])
    unknown_no_pending = _expect_pending_error(lambda: _resolve_neutral(model, probes[1]))

    recurrence_model = CausalSequencePointerModel.grow(
        examples,
        replace(_CONFIG, ablation=CausalSequenceAblation.RECURRENCE_OFF),
    )
    _recurrence_raw, recurrence_trace = _raw_complete(recurrence_model, held_out)
    recurrence_no_pending = _expect_pending_error(
        lambda: _resolve_neutral(recurrence_model, held_out)
    )

    revision_model = CausalSequencePointerModel.grow(examples, _CONFIG)
    revision_base_digest = revision_model.training_digest
    before_output, before_trace = _complete(revision_model, held_out)
    negative = revision_model.observe_failure(
        held_out.prompt,
        held_out.inventory,
        cue_text=held_out.cue_text,
        feedback_text=_NEGATIVE_RECEIPT,
    )
    negative_digest = revision_model.training_digest
    after_output, after_trace = _complete(revision_model, held_out)
    _resolve_neutral(revision_model, held_out)

    retained_outputs: list[str] = []
    retained_traces: list[CausalSequenceCompletionTrace] = []
    retained_neutral_updates: list[CausalSequenceLearningUpdate] = []
    for example in examples:
        retained_case = _GateCase(
            prompt=example.prompt,
            cue_text=example.cue_text,
            inventory=example.inventory,
            expected_output=example.output,
        )
        output, trace = _complete(revision_model, retained_case)
        retained_outputs.append(output)
        retained_traces.append(trace)
        retained_neutral_updates.append(_resolve_neutral(revision_model, retained_case))
    retained_digest = revision_model.training_digest

    changed_step = _negative_step_index(negative)
    _require(
        changed_step < len(before_trace.steps) and changed_step < len(after_trace.steps),
        "changed branch step fell outside a completion trace",
    )
    branch_before = before_trace.steps[changed_step]
    branch_after = after_trace.steps[changed_step]

    nogo_model = CausalSequencePointerModel.grow(
        examples,
        replace(_CONFIG, ablation=CausalSequenceAblation.NOGO_OFF),
    )
    nogo_digest = nogo_model.training_digest
    nogo_first_output, nogo_first_trace = _complete(nogo_model, held_out)
    nogo_negative = nogo_model.observe_failure(
        held_out.prompt,
        held_out.inventory,
        cue_text=held_out.cue_text,
        feedback_text=_NEGATIVE_RECEIPT,
    )
    nogo_repeat_output, nogo_repeat_trace = _complete(nogo_model, held_out)
    nogo_neutral = _resolve_neutral(nogo_model, held_out)

    named_traces = {
        "treatment": treatment_trace,
        "recurrence_off": recurrence_trace,
        "ambiguous": ambiguous_trace,
        "unknown": unknown_trace,
        "negative_before": before_trace,
        "negative_after": after_trace,
        **{
            f"retained_{index}": trace
            for index, trace in enumerate(retained_traces)
        },
        "nogo_off_first": nogo_first_trace,
        "nogo_off_repeat": nogo_repeat_trace,
    }
    checks = {
        "held_out_whole_sequence_novel_and_exact": (
            treatment_output == expected_held_out
            and treatment_output not in demonstrated_outputs
            and _physical_success(treatment_trace)
        ),
        "held_out_roles_fresh_and_exact": (
            all(symbol in treatment_output for symbol in held_out.inventory.symbols)
            and set(held_out.inventory.symbols).isdisjoint(
                symbol for example in examples for symbol in example.inventory.symbols
            )
        ),
        "all_choices_physical_backward_only": all(
            _physical_backward_trace(trace) for trace in named_traces.values()
        ),
        "recurrence_sequence_off_stop": (
            recurrence_trace.ablation is CausalSequenceAblation.RECURRENCE_OFF
            and _physical_stop(recurrence_trace)
        ),
        "control_bootstrap_digest_identity": (
            recurrence_trace.prior_training_digest
            == recurrence_model.training_digest
            == revision_base_digest
            == nogo_digest
            == base_digest
        ),
        "ambiguous_stop": _physical_stop(ambiguous_trace),
        "unknown_stop": _physical_stop(unknown_trace),
        "physical_stop_leaves_no_pending_action": bool(
            recurrence_no_pending and ambiguous_no_pending and unknown_no_pending
        ),
        "pending_resolution_exactly_once": bool(
            pending_error
            and duplicate_resolution_error
            and neutral.polarity == "neutral"
        ),
        "neutral_digest_identity": (
            neutral.changed_synapses == ()
            and neutral.prior_training_digest
            == neutral.training_digest
            == base_digest
            == post_neutral_digest
        ),
        "target_free_negative_changed_selected_branch": (
            before_output == expected_held_out
            and after_output != before_output
            and negative.polarity == "failure"
            and len(negative.changed_synapses) == 2
            and negative.prior_training_digest == revision_base_digest
            and negative.training_digest == negative_digest
            and negative_digest != revision_base_digest
            and branch_before.winner_cell != branch_after.winner_cell
            and branch_before.winner_operation != branch_after.winner_operation
        ),
        "unrelated_trained_contexts_retain": (
            tuple(retained_outputs) == tuple(example.output for example in examples)
            and all(_physical_success(trace) for trace in retained_traces)
            and all(
                update.changed_synapses == ()
                and update.prior_training_digest
                == update.training_digest
                == negative_digest
                for update in retained_neutral_updates
            )
            and retained_digest == negative_digest
        ),
        "nogo_off_repeats": (
            nogo_first_output == nogo_repeat_output == expected_held_out
            and nogo_first_trace.ablation is CausalSequenceAblation.NOGO_OFF
            and nogo_repeat_trace.ablation is CausalSequenceAblation.NOGO_OFF
            and nogo_negative.polarity == "failure"
            and len(nogo_negative.changed_synapses) == 1
            and nogo_negative.changed_synapses[0].startswith("consequence:")
            and not any(
                edge.startswith("nogo:") for edge in nogo_negative.changed_synapses
            )
            and nogo_negative.prior_training_digest == nogo_digest
            and nogo_negative.training_digest != nogo_digest
            and nogo_negative.training_digest
            == nogo_model.training_digest
            and nogo_neutral.changed_synapses == ()
            and nogo_neutral.prior_training_digest
            == nogo_neutral.training_digest
            == nogo_negative.training_digest
            and tuple(
                (step.winner_cell, step.winner_operation)
                for step in nogo_first_trace.steps
            )
            == tuple(
                (step.winner_cell, step.winner_operation)
                for step in nogo_repeat_trace.steps
            )
        ),
    }
    _require(all(checks.values()), "one or more causal-sequence checks failed")

    return {
        "schema": SCHEMA,
        "status": STATUS_PASS,
        "claim_boundary": CLAIM_BOUNDARY,
        "fixture_payload_sha256": EXPECTED_FIXTURE_PAYLOAD_SHA256,
        "fixture_custody_sha256": EXPECTED_FIXTURE_CUSTODY_SHA256,
        "custody": {
            "fixture_visibility": "public",
            "gate_execution_count": 1,
            "model_growth_count": 4,
            "negative_resolution_count": 1,
            "nogo_off_negative_control_count": 1,
            "target_supplied_to_negative_resolution": False,
            "target_bytes_supplied_to_negative_resolution": 0,
            "private_or_protected_inputs": 0,
            "sandbox_executions": 0,
        },
        "metrics": {
            "checks": checks,
            "demonstration_count": len(examples),
            "held_out_role_count": len(held_out.inventory.symbols),
            "base_training_digest": base_digest,
            "post_neutral_digest": post_neutral_digest,
            "revision_base_digest": revision_base_digest,
            "post_negative_digest": negative_digest,
            "changed_branch_step": changed_step,
            "changed_branch_before_cell": branch_before.winner_cell,
            "changed_branch_after_cell": branch_after.winner_cell,
            "retained_context_count": len(retained_outputs),
            "pending_response_error": pending_error,
            "duplicate_resolution_error": duplicate_resolution_error,
            "recurrence_stop_resolution_error": recurrence_no_pending,
            "ambiguous_stop_resolution_error": ambiguous_no_pending,
            "unknown_stop_resolution_error": unknown_no_pending,
        },
        "updates": {
            "neutral": _update_record(neutral),
            "negative": _update_record(negative),
            "nogo_off_negative": _update_record(nogo_negative),
        },
        "decisions": {
            name: _trace_record(trace) for name, trace in named_traces.items()
        },
    }


def _write_exclusive(path: Path, record: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _write_terminal(path: Path, record: dict[str, object]) -> None:
    current = json.loads(path.read_text(encoding="utf-8"))
    _require(current.get("status") == "RUNNING", "artifact is not in RUNNING state")
    with path.open("w", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main() -> int:
    prepared = preflight()
    source_custody = prepared["source_custody"]
    source_custody_sha256 = prepared["source_custody_sha256"]
    reservation = {
        "schema": SCHEMA,
        "status": "RUNNING",
        "claim_boundary": CLAIM_BOUNDARY,
        "fixture_payload_sha256": EXPECTED_FIXTURE_PAYLOAD_SHA256,
        "fixture_custody_sha256": EXPECTED_FIXTURE_CUSTODY_SHA256,
        "source_custody": source_custody,
        "source_custody_sha256": source_custody_sha256,
    }
    reservation_sha256 = _digest(_RESERVATION_NAMESPACE, reservation)
    _write_exclusive(
        ARTIFACT_PATH,
        {**reservation, "reservation_sha256": reservation_sha256},
    )
    try:
        _require(
            _source_custody() == source_custody,
            "source custody changed after reservation",
        )
        result = _run_reserved_gate(reservation_sha256)
    except BaseException as error:
        failure = {
            **reservation,
            "status": STATUS_FAIL,
            "reservation_sha256": reservation_sha256,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        failure["result_sha256"] = _digest(_RESULT_NAMESPACE, failure)
        _write_terminal(ARTIFACT_PATH, failure)
        raise

    completed = {
        **result,
        "reservation_sha256": reservation_sha256,
        "reservation_status": "COMPLETED",
        "source_custody": source_custody,
        "source_custody_sha256": source_custody_sha256,
    }
    completed["result_sha256"] = _digest(_RESULT_NAMESPACE, completed)
    _write_terminal(ARTIFACT_PATH, completed)
    print(f"{STATUS_PASS}: wrote {ARTIFACT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
