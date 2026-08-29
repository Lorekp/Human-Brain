from __future__ import annotations

import base64
import copy
import json
from collections.abc import Sequence
from hashlib import sha256

import humanbrain.typed_program as typed_program
import pytest
from humanbrain.cells import AllenGlif5Parameters
from humanbrain.human_cell_identity import HUMAN_GLIF5_IDENTITY
from humanbrain.typed_program import (
    CompiledEditSite,
    Example,
    FactorAction,
    FrozenCheckpoint,
    Layer1Protocol,
    Program,
    ProgramEditSite,
    ProgramFormatError,
    ProgramParameter,
    ProgramType,
    TypedSignature,
    apply_factor_action,
    compile,
    compile_edit_site,
    complete,
    grow,
    program_edit_sites,
    replace_subtree,
)


def _canonical_checkpoint_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _replace_nested(record: object, path: tuple[str | int, ...], value: object) -> None:
    cursor = record
    for part in path[:-1]:
        cursor = cursor[part]  # type: ignore[index]
    cursor[path[-1]] = value  # type: ignore[index]


def _rehash_checkpoint_circuit(record: dict[str, object]) -> None:
    circuit = record["circuit"]
    assert isinstance(circuit, dict)
    metadata = circuit["metadata"]
    arrays = circuit["arrays"]
    assert isinstance(arrays, list)
    digest = sha256(_canonical_checkpoint_bytes(metadata))
    for item in arrays:
        assert isinstance(item, dict)
        digest.update(str(item["name"]).encode("ascii"))
        digest.update(str(item["dtype"]).encode("ascii"))
        digest.update(json.dumps(tuple(item["shape"])).encode("ascii"))
        digest.update(base64.b64decode(str(item["data"]), validate=True))
    circuit["sha256"] = digest.hexdigest()


def test_frozen_checkpoint_can_only_be_constructed_from_raw_bytes() -> None:
    with pytest.raises(TypeError, match="from_bytes"):
        FrozenCheckpoint(
            sha256="0" * 64,
            training_digest="0" * 64,
            training_input_sha256=("0" * 64,),
            training_sequence_token_counts=(2,),
            training_teacher_token_counts=(1,),
            cell_synapse_sha256="0" * 64,
            protocol_sha256="0" * 64,
            _bytes=b"{}",
        )


def test_complete_rejects_checkpoint_subclasses_before_virtual_dispatch() -> None:
    class ForgedCheckpoint(FrozenCheckpoint):
        def _validated_payload(self) -> object:
            raise AssertionError("forged checkpoint method was dispatched")

    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    forged = object.__new__(ForgedCheckpoint)

    with pytest.raises(TypeError, match="exact FrozenCheckpoint"):
        complete(forged, "Return the integer.", signature)


def test_complete_rejects_forged_checkpoint_derived_fields() -> None:
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt="Return the integer.",
                signature=signature,
                teacher_tokens=("variable-0",),
            ),
        ),
    )
    forged = FrozenCheckpoint.from_bytes(checkpoint.to_bytes())
    object.__setattr__(forged, "training_digest", "0" * 64)

    with pytest.raises(ValueError, match="derived fields do not match"):
        complete(forged, "Return the integer.", signature)

    forged_counts = FrozenCheckpoint.from_bytes(checkpoint.to_bytes())
    object.__setattr__(forged_counts, "training_sequence_token_counts", (2,))

    with pytest.raises(ValueError, match="derived fields do not match"):
        complete(forged_counts, "Return the integer.", signature)

    forged_teacher_counts = FrozenCheckpoint.from_bytes(checkpoint.to_bytes())
    object.__setattr__(forged_teacher_counts, "training_teacher_token_counts", (2,))

    with pytest.raises(ValueError, match="derived fields do not match"):
        complete(forged_teacher_counts, "Return the integer.", signature)


def test_complete_rejects_noncanonical_checkpoint_bytes() -> None:
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt="Return the integer.",
                signature=signature,
                teacher_tokens=("variable-0",),
            ),
        ),
    )
    forged = FrozenCheckpoint.from_bytes(checkpoint.to_bytes())
    noncanonical = b" " + checkpoint.to_bytes()
    object.__setattr__(forged, "_bytes", noncanonical)
    object.__setattr__(forged, "sha256", sha256(noncanonical).hexdigest())

    with pytest.raises(ValueError, match="not canonically encoded"):
        complete(forged, "Return the integer.", signature)


def test_growth_caps_materialization_without_trusting_sequence_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    protocol = Layer1Protocol()

    def forbidden_model_or_circuit_work(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("oversized corpus reached model or circuit work")

    monkeypatch.setattr(typed_program, "_model_input", forbidden_model_or_circuit_work)
    monkeypatch.setattr(
        typed_program,
        "RecurrentCellCircuit",
        forbidden_model_or_circuit_work,
    )

    class UnboundedExamples(Sequence[Example]):
        def __len__(self) -> int:
            raise AssertionError("growth trusted the sequence length")

        def __getitem__(self, index: int) -> Example:
            if index > protocol.max_training_examples:
                raise AssertionError(f"growth read past bounded sentinel at index {index}")
            return object()  # type: ignore[return-value]

    with pytest.raises(ValueError, match="at most 256"):
        grow(protocol, UnboundedExamples())


def test_checkpoint_byte_limit_is_enforced_before_json_validation() -> None:
    protocol = Layer1Protocol()
    oversized_json_string = b'"' + (b"x" * protocol.max_checkpoint_bytes) + b'"'

    assert protocol.max_checkpoint_bytes == 4_194_304
    with pytest.raises(ValueError, match="exceeds the 4194304-byte protocol limit"):
        FrozenCheckpoint.from_bytes(oversized_json_string)


def test_checkpoint_records_sequences_separately_from_cell_presentations() -> None:
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt="Return the integer.",
                signature=signature,
                teacher_tokens=("variable-0",),
            ),
        ),
    )
    record = json.loads(checkpoint.to_bytes().decode("utf-8"))

    assert record["format"] == "humanbrain-layer1-typed-program-v5"
    assert record["training_sequence_token_counts"] == [170]
    assert record["training_teacher_token_counts"] == [1]
    assert checkpoint.training_sequence_token_counts == (170,)
    assert checkpoint.training_teacher_token_counts == (1,)
    assert record["circuit"]["metadata"]["growth_examples"] == 3


def test_a0_a1_a2_freeze_only_preregistered_framing_and_target_differences() -> None:
    a0 = Layer1Protocol.for_lane("A0")
    a1 = Layer1Protocol.for_lane("A1")
    a2 = Layer1Protocol.for_lane("A2")

    assert a0.lane == "A0"
    assert a0.framing_order == "prompt/context-delimiter/canonical-signature/input-delimiter"
    assert a0.supervision_scope == "all-next-token-targets"
    assert a1.lane == "A1"
    assert a1.framing_order == a2.framing_order
    assert a1.supervision_scope == "all-next-token-targets"
    assert a2.lane == "A2"
    assert a2.supervision_scope == "program-token-targets-only"

    a0_record = json.loads(typed_program._protocol_json(a0).decode("utf-8"))
    a1_record = json.loads(typed_program._protocol_json(a1).decode("utf-8"))
    a2_record = json.loads(typed_program._protocol_json(a2).decode("utf-8"))
    for record in (a0_record, a1_record, a2_record):
        record.pop("lane")
        record.pop("framing_order")
        record.pop("supervision_scope")
    assert a0_record == a1_record == a2_record


def test_a1_and_a2_observe_identical_states_but_a2_strengthens_only_program_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = TypedSignature(
        entry_point="q_bcdfghjkl",
        parameters=(ProgramParameter("datum", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    example = Example(
        prompt="Increase an integer by 2.",
        signature=signature,
        teacher_tokens=("add", "variable-0", "integer", "+", "2", "end"),
    )
    original_observe = typed_program.RecurrentCellCircuit.observe
    original_strengthen = typed_program.RecurrentCellCircuit.strengthen
    lane_calls: dict[str, tuple[list[int], list[bytes], list[int]]] = {}

    for lane in ("A1", "A2"):
        observed: list[int] = []
        state_bytes: list[bytes] = []
        strengthened: list[int] = []

        def observe(
            circuit: typed_program.RecurrentCellCircuit,
            token_id: int,
            observed: list[int] = observed,
            state_bytes: list[bytes] = state_bytes,
        ) -> object:
            step = original_observe(circuit, token_id)
            observed.append(token_id)
            state_bytes.append(circuit.eligibility.tobytes())
            return step

        def strengthen(
            circuit: typed_program.RecurrentCellCircuit,
            target_token_id: int,
            strengthened: list[int] = strengthened,
        ) -> None:
            strengthened.append(target_token_id)
            original_strengthen(circuit, target_token_id)

        with monkeypatch.context() as patch:
            patch.setattr(typed_program.RecurrentCellCircuit, "observe", observe)
            patch.setattr(typed_program.RecurrentCellCircuit, "strengthen", strengthen)
            checkpoint = grow(Layer1Protocol.for_lane(lane), (example,))
        lane_calls[lane] = (observed, state_bytes, strengthened)
        assert checkpoint.lane == lane

    a1_observed, a1_states, a1_strengthened = lane_calls["A1"]
    a2_observed, a2_states, a2_strengthened = lane_calls["A2"]
    assert a1_observed == a2_observed
    assert a1_states == a2_states
    assert len(a1_strengthened) == len(a1_observed)
    assert len(a2_strengthened) == 3 * len(example.teacher_tokens)
    assert a2_strengthened == list(typed_program._program_token_ids(example.teacher_tokens)) * 3


def test_checkpoint_v5_requires_exact_sequence_and_teacher_count_fields() -> None:
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt="Return the integer.",
                signature=signature,
                teacher_tokens=("variable-0",),
            ),
        ),
    )
    valid_record = json.loads(checkpoint.to_bytes().decode("utf-8"))
    missing = copy.deepcopy(valid_record)
    del missing["training_sequence_token_counts"]
    missing_teacher = copy.deepcopy(valid_record)
    del missing_teacher["training_teacher_token_counts"]
    extra = copy.deepcopy(valid_record)
    extra["training_sequence_lengths"] = [170]
    legacy = copy.deepcopy(valid_record)
    legacy["format"] = "humanbrain-layer1-typed-program-v3"

    for malformed in (missing, missing_teacher, extra):
        with pytest.raises(ValueError, match="must contain exactly"):
            FrozenCheckpoint.from_bytes(_canonical_checkpoint_bytes(malformed))
    with pytest.raises(ValueError, match="unsupported frozen checkpoint format"):
        FrozenCheckpoint.from_bytes(_canonical_checkpoint_bytes(legacy))


def test_checkpoint_rejects_hash_consistent_growth_count_forgery() -> None:
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt="Return the integer.",
                signature=signature,
                teacher_tokens=("variable-0",),
            ),
        ),
    )
    forged = json.loads(checkpoint.to_bytes().decode("utf-8"))
    forged["circuit"]["metadata"]["growth_examples"] = 1_000_000
    _rehash_checkpoint_circuit(forged)

    with pytest.raises(
        ValueError,
        match="growth count differs from the lane's supervised transition count",
    ):
        FrozenCheckpoint.from_bytes(_canonical_checkpoint_bytes(forged))


def test_v5_checkpoint_persists_weights_but_never_ridge_statistics() -> None:
    """v5 diet (protocol: stats never migrate): weights + config persist only."""
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt="Return the integer.",
                signature=signature,
                teacher_tokens=("variable-0",),
            ),
        ),
    )
    record = json.loads(checkpoint.to_bytes().decode("utf-8"))
    array_names = {item["name"] for item in record["circuit"]["arrays"]}
    assert "readout_synapses_pa" in array_names
    assert "normal_matrix" not in array_names
    assert "target_moments" not in array_names
    # Metadata stays on the canonical frozen schema; era-guard siblings belong
    # to statistics-carrying envelopes only, and the tau bank lives nested in
    # circuit_config.
    assert set(record["circuit"]["metadata"]) == set(
        typed_program._FROZEN_CIRCUIT_METADATA_KEYS
    )
    assert "eligibility_taus_ms" not in record["circuit"]["metadata"]

    payload = typed_program._decode_checkpoint_bytes(checkpoint.to_bytes())
    circuit = typed_program._restore_circuit(payload.circuit)
    assert circuit._normal_matrix is None
    assert circuit._target_moments is None


def test_protocol_commits_growth_and_complete_pinned_human_cell_identity() -> None:
    protocol = Layer1Protocol()
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    checkpoint = grow(
        protocol,
        (
            Example(
                prompt="Return the integer.",
                signature=signature,
                teacher_tokens=("variable-0",),
            ),
        ),
    )
    record = json.loads(checkpoint.to_bytes().decode("utf-8"))
    expected_parameters = {
        "after_spike_amplitudes_pa": [-41.71849286427266, -63.68622662400284],
        "after_spike_taus_ms": [100.0, 33.33333333333334],
        "capacitance_pf": 263.16486906079916,
        "dt_ms": 0.05,
        "input_resistance_mohm": 298.63483359781054,
        "model_id": 523000114,
        "rest_mv": -70.5407829284668,
        "specimen_id": 508378649,
        "spike_cut_steps": 113,
        "threshold_inf_relative_mv": 23.806910928976624,
        "threshold_spike_decay_per_s": 92.14596727562302,
        "threshold_spike_increment_mv": 2.5455720316895298,
        "threshold_voltage_coupling_per_s": 6.503496294985007,
        "threshold_voltage_decay_per_s": 19.773561643976418,
        "voltage_reset_offset_mv": -39.60700777695478,
        "voltage_reset_scale": 1.669558016548051,
    }

    assert protocol.protocol_version == "humanbrain-layer1-typed-program-v5"
    assert protocol.growth_epochs == 3
    assert protocol.allen_glif5_config_sha256 == HUMAN_GLIF5_IDENTITY.config_sha256
    assert protocol.allen_glif5_parameters == AllenGlif5Parameters()
    assert record["format"] == protocol.protocol_version
    assert record["protocol"]["growth_epochs"] == 3
    assert record["protocol"]["allen_glif5_config_sha256"] == (protocol.allen_glif5_config_sha256)
    assert record["protocol"]["allen_glif5_parameters"] == expected_parameters
    assert record["protocol"]["max_training_examples"] == 256
    assert record["protocol"]["max_checkpoint_bytes"] == 4_194_304


def test_completion_restores_exactly_one_recurrent_circuit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt="Return the integer.",
                signature=signature,
                teacher_tokens=("variable-0",),
            ),
        ),
    )
    # v5 routes Layer-1 completion through the dieted weights-only restore
    # (`_restore_circuit`); the generic `restore_cell_synapse_state` remains
    # the strict entry point for statistics-carrying recurrent artifacts. The
    # invariant under test is unchanged: exactly ONE circuit restored per
    # completion.
    restore = typed_program._restore_circuit
    construction_count = 0

    def counted_restore(
        frozen: object,
    ) -> typed_program.RecurrentCellCircuit:
        nonlocal construction_count
        construction_count += 1
        return restore(frozen)

    monkeypatch.setattr(typed_program, "_restore_circuit", counted_restore)

    completion = complete(checkpoint, "Return the integer.", signature)

    assert completion.program.tokens == ("variable-0",)
    assert construction_count == 1


def test_layer1_int_program_grows_completes_and_compiles_through_cell_races() -> None:
    signature = TypedSignature(
        entry_point="increment",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    teacher = ("add", "variable-0", "integer", "+", "1", "end")

    checkpoint = grow(
        Layer1Protocol(),
        (Example(prompt="Increment the integer.", signature=signature, teacher_tokens=teacher),),
    )
    completion = complete(checkpoint, "Increment the integer.", signature)

    assert completion.program.tokens == teacher
    assert completion.race_count == len(teacher)
    assert all(len(race.eligibility_bytes) == 128 * 8 for race in completion.races)
    assert all(
        len(race.allowed_output_currents_pa) == len(race.allowed_cell_ids)
        for race in completion.races
    )
    assert completion.prompt_sha256 == sha256(b"Increment the integer.").hexdigest()
    assert compile(completion.program).source == ("def increment(value):\n    return value + 1\n")


def test_generic_map_composes_a_typed_binder_and_list_argument() -> None:
    signature = TypedSignature(
        entry_point="increment_all",
        parameters=(ProgramParameter("items", ProgramType.List(ProgramType.Int())),),
        result=ProgramType.List(ProgramType.Int()),
    )
    teacher = (
        "map",
        "bind",
        "add",
        "variable-0",
        "integer",
        "+",
        "1",
        "end",
        "variable-0",
    )

    checkpoint = grow(
        Layer1Protocol(),
        (Example(prompt="Increment every integer.", signature=signature, teacher_tokens=teacher),),
    )
    completion = complete(checkpoint, "Increment every integer.", signature)

    assert completion.program.tokens == teacher
    assert completion.race_count == len(teacher)
    assert compile(completion.program).source == (
        "def increment_all(items):\n    return [item_0 + 1 for item_0 in items]\n"
    )


def test_absolute_is_a_generic_unary_int_primitive_inside_map() -> None:
    signature = TypedSignature(
        entry_point="absolute_all",
        parameters=(ProgramParameter("items", ProgramType.List(ProgramType.Int())),),
        result=ProgramType.List(ProgramType.Int()),
    )
    program = Program(
        ("map", "bind", "absolute", "variable-0", "variable-0"),
        signature,
    )

    assert compile(program).source == (
        "def absolute_all(items):\n    return [abs(item_0) for item_0 in items]\n"
    )


def test_sum_is_a_generic_list_int_reduction() -> None:
    signature = TypedSignature(
        entry_point="total",
        parameters=(ProgramParameter("items", ProgramType.List(ProgramType.Int())),),
        result=ProgramType.Int(),
    )
    program = Program(("sum", "variable-0"), signature)

    assert compile(program).source == "def total(items):\n    return sum(items)\n"


def test_nested_typed_if_lowers_an_exact_two_sided_scalar_clamp() -> None:
    signature = TypedSignature(
        entry_point="clamp",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    program = Program(
        (
            "if",
            "less-than",
            "variable-0",
            "integer",
            "-",
            "5",
            "end",
            "integer",
            "-",
            "5",
            "end",
            "if",
            "less-than",
            "integer",
            "+",
            "5",
            "end",
            "variable-0",
            "integer",
            "+",
            "5",
            "end",
            "variable-0",
        ),
        signature,
    )

    assert compile(program).source == (
        "def clamp(value):\n"
        "    return -5 if value < -5 else 5 if 5 < value else value\n"
    )


def test_typed_if_rejects_non_bool_condition_and_mismatched_branches() -> None:
    signature = TypedSignature(
        entry_point="invalid_branch",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    malformed = (
        ("if", "variable-0", "variable-0", "variable-0"),
        (
            "if",
            "less-than",
            "variable-0",
            "integer",
            "+",
            "0",
            "end",
            "variable-0",
            "less-than",
            "variable-0",
            "integer",
            "+",
            "0",
            "end",
        ),
    )

    for tokens in malformed:
        with pytest.raises(ProgramFormatError):
            compile(Program(tokens=tokens, signature=signature))


def test_subtract_is_a_general_binary_int_primitive_with_literal_operand_order() -> None:
    signature = TypedSignature(
        entry_point="difference",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )

    input_first = Program(
        ("subtract", "variable-0", "integer", "+", "7", "end"),
        signature,
    )
    literal_first = Program(
        ("subtract", "integer", "+", "7", "end", "variable-0"),
        signature,
    )

    assert compile(input_first).source == "def difference(value):\n    return value - 7\n"
    assert compile(literal_first).source == "def difference(value):\n    return 7 - value\n"


def test_program_edit_seam_exposes_paths_and_replaces_one_typed_subtree() -> None:
    signature = TypedSignature(
        entry_point="offset",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    program = Program(
        ("add", "variable-0", "integer", "+", "3", "end"),
        signature,
    )

    assert program_edit_sites(program) == (
        ProgramEditSite(
            (),
            program.tokens,
            "Int",
            "add",
            ("Int", "unbound"),
            False,
        ),
        ProgramEditSite(
            (0,),
            ("variable-0",),
            "Int",
            "variable-0",
            ("Int", "unbound"),
            False,
        ),
        ProgramEditSite(
            (1,),
            ("integer", "+", "3", "end"),
            "Int",
            "integer",
            ("Int", "unbound"),
            False,
        ),
    )

    edited = replace_subtree(
        program,
        (0,),
        ("absolute", "variable-0"),
    )
    assert compile(edited).source == "def offset(value):\n    return abs(value) + 3\n"


def test_program_edit_seam_rejects_unknown_paths_and_ill_typed_replacements() -> None:
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    program = Program(("variable-0",), signature)

    with pytest.raises(ValueError, match="no edit site"):
        replace_subtree(program, (0,), ("variable-0",))
    with pytest.raises(ProgramFormatError):
        replace_subtree(program, (), ("less-than", "variable-0", "variable-0"))

    binary = Program(("add", "variable-0", "variable-0"), signature)
    with pytest.raises(ProgramFormatError, match="exactly one expression"):
        replace_subtree(binary, (0,), ("variable-0", "absolute"))

    list_integer = ProgramType.List(ProgramType.Int())
    map_signature = TypedSignature(
        entry_point="mapped",
        parameters=(ProgramParameter("values", list_integer),),
        result=list_integer,
    )
    mapped = Program(("map", "bind", "variable-0", "variable-0"), map_signature)
    function_valued_if = (
        "if",
        "less-than",
        "integer",
        "+",
        "0",
        "end",
        "integer",
        "+",
        "1",
        "end",
        "bind",
        "variable-0",
        "bind",
        "variable-0",
    )
    with pytest.raises(ProgramFormatError, match="map requires one explicit binder"):
        replace_subtree(mapped, (0,), function_valued_if)


def test_compile_edit_site_lowers_scalar_root_and_child_with_generic_scope() -> None:
    signature = TypedSignature(
        entry_point="offset",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    program = Program(
        ("add", "absolute", "variable-0", "integer", "+", "2", "end"),
        signature,
    )

    root = compile_edit_site(program, ())
    child = compile_edit_site(program, (0,))

    assert isinstance(root, CompiledEditSite)
    assert root.path == ()
    assert root.result_type_label == "Int"
    assert root.scope_type_labels == ("Int",)
    assert root.source == (
        "def edit_site(scope_0):\n    return abs(scope_0) + 2\n"
    )
    assert child.path == (0,)
    assert child.result_type_label == "Int"
    assert child.scope_type_labels == ("Int",)
    assert child.source == "def edit_site(scope_0):\n    return abs(scope_0)\n"


def test_compile_edit_site_preserves_two_parameter_de_bruijn_order() -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="difference",
        parameters=(
            ProgramParameter("a", integer),
            ProgramParameter("b", integer),
        ),
        result=integer,
    )

    a_minus_b = compile_edit_site(
        Program(("subtract", "variable-1", "variable-0"), signature),
        (),
    )
    b_minus_a = compile_edit_site(
        Program(("subtract", "variable-0", "variable-1"), signature),
        (),
    )

    assert a_minus_b.scope_type_labels == ("Int", "Int")
    assert a_minus_b.source == (
        "def edit_site(scope_0, scope_1):\n    return scope_1 - scope_0\n"
    )
    assert b_minus_a.source == (
        "def edit_site(scope_0, scope_1):\n    return scope_0 - scope_1\n"
    )


def test_compile_edit_site_supports_map_body_and_collection_value_sites() -> None:
    list_integer = ProgramType.List(ProgramType.Int())
    signature = TypedSignature(
        entry_point="map_values",
        parameters=(ProgramParameter("values", list_integer),),
        result=list_integer,
    )
    absolute_map = Program(
        ("map", "bind", "absolute", "variable-0", "variable-0"),
        signature,
    )
    add_two_map = Program(
        (
            "map",
            "bind",
            "add",
            "variable-0",
            "integer",
            "+",
            "2",
            "end",
            "variable-0",
        ),
        signature,
    )

    absolute_body = compile_edit_site(absolute_map, (0, 0))
    add_two_body = compile_edit_site(add_two_map, (0, 0))
    collection = compile_edit_site(absolute_map, (1,))

    assert absolute_body.result_type_label == "Int"
    assert absolute_body.scope_type_labels == ("Int", "List[Int]")
    assert absolute_body.source == (
        "def edit_site(scope_0, scope_1):\n    return abs(scope_0)\n"
    )
    assert add_two_body.scope_type_labels == ("Int", "List[Int]")
    assert add_two_body.source == (
        "def edit_site(scope_0, scope_1):\n    return scope_0 + 2\n"
    )
    assert collection.result_type_label == "List[Int]"
    assert collection.scope_type_labels == ("List[Int]",)
    assert collection.source == "def edit_site(scope_0):\n    return scope_0\n"


def test_compile_edit_site_hash_is_deterministic_and_name_independent() -> None:
    integer = ProgramType.Int()
    first = Program(
        ("absolute", "variable-0"),
        TypedSignature(
            entry_point="first_name",
            parameters=(ProgramParameter("first_parameter", integer),),
            result=integer,
        ),
    )
    second = Program(
        ("absolute", "variable-0"),
        TypedSignature(
            entry_point="second_name",
            parameters=(ProgramParameter("second_parameter", integer),),
            result=integer,
        ),
    )

    first_site = compile_edit_site(first, ())
    repeated = compile_edit_site(first, ())
    renamed = compile_edit_site(second, ())

    assert first_site == repeated == renamed
    assert first_site.sha256 == sha256(first_site.source.encode("utf-8")).hexdigest()


def test_compile_edit_site_rejects_bad_paths_and_non_value_binders() -> None:
    list_integer = ProgramType.List(ProgramType.Int())
    signature = TypedSignature(
        entry_point="mapped",
        parameters=(ProgramParameter("values", list_integer),),
        result=list_integer,
    )
    program = Program(("map", "bind", "variable-0", "variable-0"), signature)

    with pytest.raises(ValueError, match="no edit site"):
        compile_edit_site(program, (9,))
    with pytest.raises(ProgramFormatError, match="non-value"):
        compile_edit_site(program, (0,))


def test_factor_action_preserves_binary_binding_orientation() -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="difference",
        parameters=(
            ProgramParameter("first", integer),
            ProgramParameter("second", integer),
        ),
        result=integer,
    )
    incumbent = Program(("variable-1",), signature)

    first_minus_second = apply_factor_action(
        incumbent,
        FactorAction((), "APPLY2", "subtract", "SCOPE1", "SCOPE0", None),
    )
    second_minus_first = apply_factor_action(
        incumbent,
        FactorAction((), "APPLY2", "subtract", "SCOPE0", "SCOPE1", None),
    )

    assert compile(first_minus_second).source == (
        "def difference(first, second):\n    return first - second\n"
    )
    assert compile(second_minus_first).source == (
        "def difference(first, second):\n    return second - first\n"
    )


def test_factor_action_preserves_comparison_operand_orientation() -> None:
    integer = ProgramType.Int()
    signature = TypedSignature(
        entry_point="predicate",
        parameters=(ProgramParameter("value", integer),),
        result=ProgramType.Bool(),
    )
    incumbent = Program(
        ("less-than", "variable-0", "integer", "+", "0", "end"),
        signature,
    )

    value_less_than_two = apply_factor_action(
        incumbent,
        FactorAction((), "APPLY2", "less-than", "SCOPE0", "LITERAL", 2),
    )
    two_less_than_value = apply_factor_action(
        incumbent,
        FactorAction((), "APPLY2", "less-than", "LITERAL", "SCOPE0", 2),
    )

    assert compile(value_less_than_two).source == (
        "def predicate(value):\n    return value < 2\n"
    )
    assert compile(two_less_than_value).source == (
        "def predicate(value):\n    return 2 < value\n"
    )


def test_factor_action_builds_map_then_edits_its_typed_binder_body() -> None:
    list_integer = ProgramType.List(ProgramType.Int())
    signature = TypedSignature(
        entry_point="map_values",
        parameters=(ProgramParameter("values", list_integer),),
        result=list_integer,
    )
    incumbent = Program(("variable-0",), signature)
    identity_map = apply_factor_action(
        incumbent,
        FactorAction((), "MAP_BIND", "map", "BOUND0", "SELECTED", None),
    )
    sites = {site.path: site for site in program_edit_sites(identity_map)}

    assert sites[()].result_type_label == "List[Int]"
    assert sites[()].head_production == "map"
    assert sites[(0,)].result_type_label == "internal-function"
    assert sites[(0,)].is_binder
    assert sites[(0, 0)].result_type_label == "Int"
    assert sites[(0, 0)].scope_binding_type_labels == ("Int", "List[Int]")
    assert sites[(1,)].scope_binding_type_labels == ("List[Int]", "unbound")

    absolute_map = apply_factor_action(
        identity_map,
        FactorAction((0, 0), "APPLY1", "absolute", "BOUND0", "NONE", None),
    )
    add_two_map = apply_factor_action(
        identity_map,
        FactorAction((0, 0), "APPLY2", "add", "BOUND0", "LITERAL", 2),
    )

    assert compile(absolute_map).source == (
        "def map_values(values):\n    return [abs(item_0) for item_0 in values]\n"
    )
    assert compile(add_two_map).source == (
        "def map_values(values):\n    return [item_0 + 2 for item_0 in values]\n"
    )


def test_factor_action_done_is_identity_and_invalid_tuples_do_not_actuate() -> None:
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    program = Program(("variable-0",), signature)

    assert apply_factor_action(
        program,
        FactorAction((), "DONE", "NONE", "NONE", "NONE", None),
    ) is program

    with pytest.raises(ValueError, match="APPLY1 supports only"):
        apply_factor_action(
            program,
            FactorAction((), "APPLY1", "add", "SELECTED", "NONE", None),
        )
    with pytest.raises(ProgramFormatError):
        apply_factor_action(
            program,
            FactorAction((), "APPLY2", "less-than", "SCOPE0", "LITERAL", 2),
        )


def test_variable_one_selects_the_outer_parameter_and_is_masked_without_it() -> None:
    integer = ProgramType.Int()
    two_inputs = TypedSignature(
        entry_point="difference",
        parameters=(
            ProgramParameter("first", integer),
            ProgramParameter("second", integer),
        ),
        result=integer,
    )

    program = Program(
        ("subtract", "variable-1", "variable-0"),
        two_inputs,
    )
    assert compile(program).source == (
        "def difference(first, second):\n    return first - second\n"
    )

    one_input = TypedSignature(
        entry_point="invalid",
        parameters=(ProgramParameter("only", integer),),
        result=integer,
    )
    state = typed_program._TypedHoleState(one_input)
    assert "variable-1" not in {spec.token for spec in state.allowed()}
    with pytest.raises(ProgramFormatError, match="invalid at this typed hole"):
        compile(Program(("variable-1",), one_input))


def test_checkpoint_corruption_is_rejected_before_array_or_circuit_restore(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signature = TypedSignature(
        entry_point="increment",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    teacher = ("add", "variable-0", "integer", "+", "1", "end")
    checkpoint = grow(
        Layer1Protocol(),
        (Example(prompt="Increment.", signature=signature, teacher_tokens=teacher),),
    )
    valid_record = json.loads(checkpoint.to_bytes().decode("utf-8"))

    array_name = valid_record["circuit"]["arrays"][0]["name"]
    array_data = valid_record["circuit"]["arrays"][0]["data"]
    changed_array_data = ("A" if array_data[0] != "A" else "B") + array_data[1:]
    input_digest = valid_record["training_input_sha256"][0]
    sequence_token_count = valid_record["training_sequence_token_counts"][0]
    teacher_token_count = valid_record["training_teacher_token_counts"][0]
    mutations = (
        (
            "object dtype",
            ("circuit", "arrays", 0, "dtype"),
            "|O",
            "dtype does not match the protocol",
        ),
        (
            "oversized shape",
            ("circuit", "arrays", 0, "shape"),
            [1_000_000_000],
            "shape does not match the protocol",
        ),
        (
            "negative shape",
            ("circuit", "arrays", 0, "shape"),
            [-1],
            "shape does not match the protocol",
        ),
        (
            "wrong byte length",
            ("circuit", "arrays", 0, "data"),
            "",
            "byte length does not match the protocol",
        ),
        (
            "same-length array corruption",
            ("circuit", "arrays", 0, "data"),
            changed_array_data,
            "cell/synapse state digest mismatch",
        ),
        (
            "wrong circuit config",
            ("circuit", "metadata", "circuit_config", "reservoir_cells"),
            129,
            "metadata does not match the protocol",
        ),
        (
            "wrong token count",
            ("circuit", "metadata", "token_cells"),
            valid_record["circuit"]["metadata"]["token_cells"] + 1,
            "metadata does not match the protocol",
        ),
        (
            "wrong Allen identity",
            ("circuit", "metadata", "allen_glif5_config_sha256"),
            "0" * 64,
            "metadata does not match the protocol",
        ),
        (
            "wrong fitted cell parameters",
            ("circuit", "metadata", "cell_parameters", "rest_mv"),
            0.0,
            "metadata does not match the protocol",
        ),
        (
            "extra circuit metadata",
            ("circuit", "metadata", "task_id"),
            "forbidden",
            "must contain exactly",
        ),
        (
            "wrong array set",
            ("circuit", "arrays", 0, "name"),
            f"unexpected_{array_name}",
            "array names do not match the protocol",
        ),
        (
            "empty training provenance",
            ("training_input_sha256",),
            [],
            "at least one training input digest",
        ),
        (
            "too much training provenance",
            ("training_input_sha256",),
            [input_digest] * 257,
            "at most 256 training input digests",
        ),
        (
            "empty training sequence counts",
            ("training_sequence_token_counts",),
            [],
            "at least one training sequence token count",
        ),
        (
            "too many training sequence counts",
            ("training_sequence_token_counts",),
            [sequence_token_count] * 257,
            "at most 256 training sequence token counts",
        ),
        (
            "noninteger training sequence count",
            ("training_sequence_token_counts",),
            [True],
            "must be integers",
        ),
        (
            "one-token training sequence",
            ("training_sequence_token_counts",),
            [1],
            "at least two tokens",
        ),
        (
            "sequence count missing its input digest",
            ("training_sequence_token_counts",),
            [sequence_token_count, sequence_token_count],
            "one count per training input digest",
        ),
        (
            "empty training teacher counts",
            ("training_teacher_token_counts",),
            [],
            "at least one training teacher token count",
        ),
        (
            "too many training teacher counts",
            ("training_teacher_token_counts",),
            [teacher_token_count] * 257,
            "at most 256 training teacher token counts",
        ),
        (
            "noninteger training teacher count",
            ("training_teacher_token_counts",),
            [True],
            "must be integers",
        ),
        (
            "empty teacher token stream",
            ("training_teacher_token_counts",),
            [0],
            "at least one token",
        ),
        (
            "teacher count missing its input digest",
            ("training_teacher_token_counts",),
            [teacher_token_count, teacher_token_count],
            "one teacher count per training input digest",
        ),
        (
            "teacher consumes the entire sequence",
            ("training_teacher_token_counts",),
            [sequence_token_count],
            "smaller than sequence count",
        ),
    )

    def forbidden_restore(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("corrupt checkpoint reached array or circuit restoration")

    monkeypatch.setattr(typed_program.np, "frombuffer", forbidden_restore)
    monkeypatch.setattr(
        typed_program.RecurrentCellCircuit,
        "restore_cell_synapse_state",
        forbidden_restore,
    )

    for _label, path, replacement, message in mutations:
        corrupted = copy.deepcopy(valid_record)
        _replace_nested(corrupted, path, replacement)
        with pytest.raises(ValueError, match=message):
            FrozenCheckpoint.from_bytes(_canonical_checkpoint_bytes(corrupted))


def test_checkpoint_round_trip_preserves_framed_prompt_signature_audit() -> None:
    signature = TypedSignature(
        entry_point="identity",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    signature_text = _canonical_checkpoint_bytes(signature.to_record()).decode("utf-8")
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt=signature_text,
                signature=signature,
                teacher_tokens=("variable-0",),
            ),
        ),
    )
    restored = FrozenCheckpoint.from_bytes(checkpoint.to_bytes())
    record = json.loads(restored.to_bytes().decode("utf-8"))
    token_order = tuple(item["token"] for item in record["protocol"]["primitive_table"])
    context_delimiter = 256 + token_order.index("context-delimiter")
    input_delimiter = 256 + token_order.index("input-delimiter")
    signature_bytes = _canonical_checkpoint_bytes(signature.to_record())
    framed_ids = (
        *signature_bytes,
        context_delimiter,
        *signature_text.encode("utf-8"),
        input_delimiter,
    )
    framed_bytes = b"".join(token_id.to_bytes(2, "big") for token_id in framed_ids)
    unframed_ids = (
        *signature_text.encode("utf-8"),
        *signature_bytes,
        input_delimiter,
    )
    unframed_bytes = b"".join(token_id.to_bytes(2, "big") for token_id in unframed_ids)

    assert restored == checkpoint
    assert restored.to_bytes() == checkpoint.to_bytes()
    assert restored.training_input_sha256 == (sha256(framed_bytes).hexdigest(),)
    assert restored.training_sequence_token_counts == (299,)
    assert restored.training_teacher_token_counts == (1,)
    assert record["training_sequence_token_counts"] == [299]
    assert record["training_teacher_token_counts"] == [1]
    assert restored.training_input_sha256 != (sha256(unframed_bytes).hexdigest(),)
    completion = complete(restored, signature_text, signature)
    assert completion.model_input_sha256 == restored.training_input_sha256[0]


def test_signed_literals_are_bounded_canonical_and_type_checked() -> None:
    signature = TypedSignature(
        entry_point="subtract_twelve",
        parameters=(ProgramParameter("value", ProgramType.Int()),),
        result=ProgramType.Int(),
    )
    teacher = ("add", "variable-0", "integer", "-", "1", "2", "end")
    checkpoint = grow(
        Layer1Protocol(),
        (
            Example(
                prompt="Subtract twelve.",
                signature=signature,
                teacher_tokens=teacher,
            ),
        ),
    )

    completion = complete(checkpoint, "Subtract twelve.", signature)

    assert completion.program.tokens == teacher
    assert completion.race_count == len(teacher)
    assert compile(completion.program).source == (
        "def subtract_twelve(value):\n    return value + -12\n"
    )

    constant_signature = TypedSignature(
        entry_point="constant",
        parameters=(),
        result=ProgramType.Int(),
    )
    invalid_literals = (
        ("integer", "-", "0", "end"),
        ("integer", "+", "0", "1", "end"),
        ("integer", "+", "1", "2", "3", "4", "5", "end"),
    )
    for tokens in invalid_literals:
        with pytest.raises(ProgramFormatError):
            compile(Program(tokens=tokens, signature=constant_signature))

    bool_signature = TypedSignature(
        entry_point="not_an_integer",
        parameters=(),
        result=ProgramType.Bool(),
    )
    with pytest.raises(ProgramFormatError):
        compile(
            Program(
                tokens=("integer", "+", "1", "end"),
                signature=bool_signature,
            )
        )
