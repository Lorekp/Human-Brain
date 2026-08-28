"""PROTOTYPE: task-refreshed typed-hole decoding with local cell races.

Question: can one fitted recurrent-cell circuit recombine program productions
learned from ordinary teacher programs when every production is a fresh,
prompt-last query?  This is a freely rerunnable architecture diagnostic, not a
capability result and not part of the frozen M0/C1 evidence path.

The host supplies only the existing typed grammar mask, a generic description
of the current hole, and structural compilation.  Every emitted production is
the winner of a simulated fitted-cell race.  There are no factor labels,
fragment tables, fixed target skeletons, or offline classifiers.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.recurrent import (  # noqa: E402
    RecurrentCellCircuit,
    RecurrentCircuitConfig,
)
from humanbrain.typed_program_types import (  # noqa: E402
    ProgramParameter,
    ProgramType,
    TypedSignature,
)

_MAX_PRODUCTIONS = 64
_CONFIG = RecurrentCircuitConfig(
    reservoir_cells=96,
    input_fanout=16,
    recurrent_fanout=8,
    readout_learning_rule="three_factor",
    local_learning_rate_pa=64.0,
    local_competitive_margin_pa=64.0,
    random_seed=17,
)


@dataclass(frozen=True, slots=True)
class _HoleCursor:
    path: tuple[int, ...]
    parent: str
    child_index: int

    @property
    def depth(self) -> int:
        return len(self.path)


@dataclass(frozen=True, slots=True)
class DiagnosticCheckpoint:
    fitted: tp._FrozenCircuitState
    control: tp._FrozenCircuitState
    curriculum_prompts: tuple[str, ...]
    training_episodes: int


@dataclass(frozen=True, slots=True)
class DecisionAudit:
    position: int
    accepted_token: str
    winner_cell_id: int
    allowed_tokens: tuple[str, ...]
    path: tuple[int, ...]
    parent: str
    descriptor_sha256: str
    frame_sha256: str
    prompt_last: bool


@dataclass(frozen=True, slots=True)
class DecodeResult:
    arm: str
    prompt: str
    expected_tokens: tuple[str, ...]
    tokens: tuple[str, ...]
    exact: bool
    first_mismatch: int | None
    source: str | None
    failure: str | None
    persistent_before: str
    persistent_after: str
    decisions: tuple[DecisionAudit, ...]


@dataclass(frozen=True, slots=True)
class DiagnosticResult:
    verdict: str
    reason: str
    fitted_checkpoint_sha256: str
    control_checkpoint_sha256: str
    training_episodes: int
    curriculum_prompts: tuple[str, ...]
    target_prompts: tuple[str, ...]
    fitted: tuple[DecodeResult, ...]
    control: tuple[DecodeResult, ...]


@dataclass(frozen=True, slots=True)
class _FreshTarget:
    prompt: str
    signature: TypedSignature
    expected_tokens: tuple[str, ...]


def _signature(parameter_type: ProgramType, result: ProgramType) -> TypedSignature:
    return TypedSignature(
        entry_point="solve",
        parameters=(ProgramParameter("arg0", parameter_type),),
        result=result,
    )


def curriculum_examples() -> tuple[tp.Example, ...]:
    """Return only ordinary prompt/signature/teacher-program examples."""
    integer = ProgramType.Int()
    integer_list = ProgramType.List(integer)
    return (
        tp.Example(
            prompt="Transform an integer into its absolute value.",
            signature=_signature(integer, integer),
            teacher_tokens=("absolute", "variable-0"),
        ),
        tp.Example(
            prompt="Subtract each list integer from itself.",
            signature=_signature(integer_list, integer_list),
            teacher_tokens=(
                "map",
                "bind",
                "subtract",
                "variable-0",
                "variable-0",
                "variable-0",
            ),
        ),
        tp.Example(
            prompt="Add all integers in the list.",
            signature=_signature(integer_list, integer),
            teacher_tokens=("sum", "variable-0"),
        ),
        tp.Example(
            prompt="Return the supplied integer unchanged.",
            signature=_signature(integer, integer),
            teacher_tokens=("variable-0",),
        ),
    )


def _fresh_targets() -> tuple[_FreshTarget, ...]:
    """Instantiate unseen composition requests only after fitting is complete."""
    integer = ProgramType.Int()
    integer_list = ProgramType.List(integer)
    return (
        _FreshTarget(
            prompt="Transform each list integer into its absolute value.",
            signature=_signature(integer_list, integer_list),
            expected_tokens=("map", "bind", "absolute", "variable-0", "variable-0"),
        ),
        _FreshTarget(
            prompt="Add the absolute values of all integers in the list.",
            signature=_signature(integer_list, integer),
            expected_tokens=(
                "sum",
                "map",
                "bind",
                "absolute",
                "variable-0",
                "variable-0",
            ),
        ),
    )


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _descriptor_bytes(state: tp._TypedHoleState, cursor: _HoleCursor) -> bytes:
    literal = state._literal
    if literal is None:
        hole = state._holes[-1]
        substitutions = state._substitutions
        expected = tp._type_term_record(tp._resolve(hole.expected, substitutions))
        environment = [
            tp._type_term_record(tp._resolve(term, substitutions))
            for term in hole.environment
        ]
        literal_phase: object = None
    else:
        expected = {"literal": "integer"}
        environment = []
        literal_phase = {
            "digits": len(literal.digits),
            "sign_selected": literal.sign is not None,
        }
    return _canonical_json(
        {
            "child_index": cursor.child_index,
            "depth": cursor.depth,
            "environment": environment,
            "expected": expected,
            "literal_phase": literal_phase,
            "parent": cursor.parent,
            "path": list(cursor.path),
        }
    )


def _decision_frame(
    prompt: str,
    signature: TypedSignature,
    state: tp._TypedHoleState,
    cursor: _HoleCursor,
    prefix: tuple[str, ...],
) -> tuple[tuple[int, ...], bytes]:
    """Build signature/descriptor/prefix then prompt-last decision framing."""
    descriptor = _descriptor_bytes(state, cursor)
    frame = (
        *tp._signature_bytes(signature),
        *b"\n<hole>",
        *descriptor,
        *b"\n<prefix>",
        *tp._program_token_ids(prefix),
        tp._CONTEXT_DELIMITER_ID,
        *prompt.encode("utf-8"),
        tp._INPUT_DELIMITER_ID,
    )
    return frame, descriptor


def _advance_cursor(
    cursors: list[_HoleCursor],
    spec: tp._PrimitiveSpec,
) -> None:
    current = cursors[-1]
    if spec.role == "operation":
        cursors.pop()
        for child_index in reversed(range(len(spec.arguments))):
            cursors.append(
                _HoleCursor(
                    path=(*current.path, child_index),
                    parent=spec.token,
                    child_index=child_index,
                )
            )
    elif spec.role == "binder":
        cursors.pop()
        cursors.append(
            _HoleCursor(
                path=(*current.path, 0),
                parent=spec.token,
                child_index=0,
            )
        )
    elif spec.role == "literal":
        cursors.pop()
        cursors.append(
            _HoleCursor(
                path=(*current.path, 0),
                parent=spec.token,
                child_index=0,
            )
        )
    elif spec.role in {"variable", "literal-end"}:
        cursors.pop()
    elif spec.role not in {"sign", "digit"}:
        raise AssertionError(f"unsupported grammar role {spec.role!r}")


def _replay(circuit: RecurrentCellCircuit, frame: tuple[int, ...]) -> None:
    circuit.reset_sequence()
    for token_id in frame:
        circuit.observe(token_id)


def _validate_teacher(example: tp.Example) -> None:
    tp.compile(tp.Program(example.teacher_tokens, example.signature))


def fit_checkpoint(*, epochs: int = 12) -> DiagnosticCheckpoint:
    """Fit local synapses on independent typed-hole query episodes."""
    if epochs < 1:
        raise ValueError("epochs must be positive")
    examples = curriculum_examples()
    for example in examples:
        _validate_teacher(example)

    control_circuit = RecurrentCellCircuit(tp._TOKEN_CELL_COUNT, config=_CONFIG)
    # The production capture seam requires an immutable lifecycle state, while
    # its public freeze method intentionally rejects zero-example circuits.
    # This flag freezes the constructor-initialized (therefore untrained)
    # synapses; it does not write or fit any readout value.
    control_circuit._learning_frozen = True
    control = tp._freeze_circuit(control_circuit)
    fitted_circuit = RecurrentCellCircuit(tp._TOKEN_CELL_COUNT, config=_CONFIG)
    episodes = 0
    for _ in range(epochs):
        for example in examples:
            state = tp._TypedHoleState(example.signature)
            cursors = [_HoleCursor(path=(), parent="root", child_index=-1)]
            prefix: list[str] = []
            for teacher_token in example.teacher_tokens:
                if not cursors:
                    raise AssertionError("teacher program exhausted its typed holes early")
                allowed = state.allowed()
                spec = tp._SPEC_BY_TOKEN[teacher_token]
                if spec not in allowed:
                    raise AssertionError(
                        f"teacher token {teacher_token!r} is invalid after {prefix!r}"
                    )
                frame, _ = _decision_frame(
                    example.prompt,
                    example.signature,
                    state,
                    cursors[-1],
                    tuple(prefix),
                )
                _replay(fitted_circuit, frame)
                fitted_circuit.strengthen(tp._TOKEN_TO_ID[teacher_token])
                state.consume(spec)
                _advance_cursor(cursors, spec)
                prefix.append(teacher_token)
                episodes += 1
            if not state.complete or cursors:
                raise AssertionError("teacher program did not close every typed hole")
    fitted_circuit.freeze_learning()
    fitted = tp._freeze_circuit(fitted_circuit)
    return DiagnosticCheckpoint(
        fitted=fitted,
        control=control,
        curriculum_prompts=tuple(example.prompt for example in examples),
        training_episodes=episodes,
    )


def _first_mismatch(actual: tuple[str, ...], expected: tuple[str, ...]) -> int | None:
    for index, (actual_token, expected_token) in enumerate(zip(actual, expected)):
        if actual_token != expected_token:
            return index
    if len(actual) != len(expected):
        return min(len(actual), len(expected))
    return None


def _decode(
    frozen: tp._FrozenCircuitState,
    arm: str,
    target: _FreshTarget,
) -> DecodeResult:
    circuit = tp._restore_circuit(frozen)
    before = tp._freeze_circuit(circuit).sha256
    state = tp._TypedHoleState(target.signature)
    cursors = [_HoleCursor(path=(), parent="root", child_index=-1)]
    prefix: list[str] = []
    decisions: list[DecisionAudit] = []
    failure: str | None = None
    while not state.complete and len(prefix) < _MAX_PRODUCTIONS:
        if not cursors:
            failure = "cursor stack ended before typed state"
            break
        allowed = state.allowed()
        if not allowed:
            failure = "typed grammar exposed no production"
            break
        frame, descriptor = _decision_frame(
            target.prompt,
            target.signature,
            state,
            cursors[-1],
            tuple(prefix),
        )
        _replay(circuit, frame)
        allowed_ids = tuple(tp._TOKEN_TO_ID[spec.token] for spec in allowed)
        race = circuit.race(allowed_token_ids=allowed_ids)
        accepted = tp._ID_TO_SPEC[race.winner]
        prompt_suffix = (
            tp._CONTEXT_DELIMITER_ID,
            *target.prompt.encode("utf-8"),
            tp._INPUT_DELIMITER_ID,
        )
        cursor = cursors[-1]
        decisions.append(
            DecisionAudit(
                position=len(prefix),
                accepted_token=accepted.token,
                winner_cell_id=race.winner,
                allowed_tokens=tuple(spec.token for spec in allowed),
                path=cursor.path,
                parent=cursor.parent,
                descriptor_sha256=sha256(descriptor).hexdigest(),
                frame_sha256=sha256(tp._token_stream_bytes(frame)).hexdigest(),
                prompt_last=frame[-len(prompt_suffix) :] == prompt_suffix,
            )
        )
        state.consume(accepted)
        _advance_cursor(cursors, accepted)
        prefix.append(accepted.token)
    if not state.complete and failure is None:
        failure = f"decoder exceeded {_MAX_PRODUCTIONS} productions"
    tokens = tuple(prefix)
    source: str | None = None
    if state.complete:
        if cursors:
            failure = "typed state completed before cursor stack"
        else:
            try:
                source = tp.compile(tp.Program(tokens, target.signature)).source
            except Exception as error:  # diagnostic localization, not recovery
                failure = f"compiler rejected cell-selected tree: {error}"
    after = tp._freeze_circuit(circuit).sha256
    if after != before:
        failure = "inference changed frozen cell/synapse state"
    mismatch = _first_mismatch(tokens, target.expected_tokens)
    return DecodeResult(
        arm=arm,
        prompt=target.prompt,
        expected_tokens=target.expected_tokens,
        tokens=tokens,
        exact=mismatch is None and failure is None,
        first_mismatch=mismatch,
        source=source,
        failure=failure,
        persistent_before=before,
        persistent_after=after,
        decisions=tuple(decisions),
    )


def run_diagnostic(*, epochs: int = 12) -> DiagnosticResult:
    """Fit first, then construct and decode the two unseen compositions."""
    checkpoint = fit_checkpoint(epochs=epochs)
    targets = _fresh_targets()
    target_prompts = tuple(target.prompt for target in targets)
    if set(target_prompts) & set(checkpoint.curriculum_prompts):
        raise AssertionError("fresh target prompt duplicated a curriculum prompt")
    fitted = tuple(_decode(checkpoint.fitted, "fitted", target) for target in targets)
    control = tuple(_decode(checkpoint.control, "untrained", target) for target in targets)
    fitted_exact = sum(result.exact for result in fitted)
    control_exact = sum(result.exact for result in control)
    if fitted_exact == len(targets) and control_exact == 0:
        verdict = "PASS"
        reason = "fitted cells recombined both fresh programs; untrained cells solved neither"
    else:
        verdict = "FAIL"
        failures = ", ".join(
            f"{result.arm}[{index}]@{result.first_mismatch}"
            for result in (*fitted, *control)
            for index in range(len(targets))
            if result.prompt == targets[index].prompt and not result.exact
        )
        reason = (
            f"exact fitted={fitted_exact}/{len(targets)}, "
            f"untrained={control_exact}/{len(targets)}; mismatches: {failures}"
        )
    return DiagnosticResult(
        verdict=verdict,
        reason=reason,
        fitted_checkpoint_sha256=checkpoint.fitted.sha256,
        control_checkpoint_sha256=checkpoint.control.sha256,
        training_episodes=checkpoint.training_episodes,
        curriculum_prompts=checkpoint.curriculum_prompts,
        target_prompts=target_prompts,
        fitted=fitted,
        control=control,
    )


def main() -> int:
    result = run_diagnostic()
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
