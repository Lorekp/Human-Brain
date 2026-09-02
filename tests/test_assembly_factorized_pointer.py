from __future__ import annotations

import json
from hashlib import sha256

import pytest
from experiments.r4_integrated.public_coding_prior_v1 import (
    public_coding_prior_v1,
)
from humanbrain.assembly_factorized_pointer import (
    AssemblyFactorizedPointerConfig,
    AssemblyFactorizedPointerError,
    AssemblyFactorizedPointerModel,
    AssemblyPointerTrainingExample,
)
from humanbrain.persistent_coding import (
    ObservationBlock,
    ObservationFrame,
    PersistentCodingLearner,
    StopDecision,
    ToolAffordance,
    _prompt,
)
from humanbrain.pointer_binding import (
    SymbolInventory,
    materialize,
    pointerize,
    python_interface_inventory,
)
from humanbrain.recurrent_assembly_stack import (
    AssemblyAblation,
    AssemblyStackConfig,
    Valence,
)
from humanbrain.tokenizer import TokenizerConfig

_INVENTORY = SymbolInventory(("entry", "argument"))
_NESTED_ROWS = (
    ("ACE12", "qu1y2ws"),
    ("ADF34", "qv4z3xs"),
    ("BCF56", "ru6z5wt"),
    ("BDE78", "rv7y8xt"),
)
_FRESH_NESTED_CUE = "ACF90"
_FRESH_NESTED_OUTPUT = "qu0z9ws"
_DIRECT_TRAIN_CUE = "K!"
_DIRECT_FRESH_CUE = "K?"
_FRESH_INVENTORY = SymbolInventory(("fresh_entry", "fresh_argument"))
_VARIABLE_ROWS = (
    (
        "alpha cold lane entry argument",
        "HEAD[entry]::AAAA::MID::CCCC::TAIL[argument]",
    ),
    (
        "argument entry lane cold alpha",
        "HEAD[entry]::AAAA::MID::DDDDDDDD::TAIL[argument]",
    ),
    (
        "beta cold lane expanded entry argument",
        "HEAD[entry]::BBBBBBBB::MID::CCCC::TAIL[argument]",
    ),
    (
        "alpha lane cold with extra public context entry argument",
        "HEAD[entry]::AAAA::MID::DDDDDDDD::TAIL[argument]",
    ),
)
_WRITE_AFFORDANCE = ToolAffordance.seal(
    "write_source",
    {
        "additionalProperties": False,
        "properties": {"source": {"type": "string"}},
        "required": ["source"],
        "type": "object",
    },
)


class _MalformedPendingModel:
    def __init__(self) -> None:
        self._digest = "malformed-pending-before"
        self.pending = False
        self.failure_calls = 0
        self.changed_edges: tuple[str, ...] = ()

    @property
    def training_digest(self) -> str:
        return self._digest

    def raw_complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 32,
        stop_text: str | tuple[str, ...] | None = None,
    ) -> str:
        del prompt, inventory, cue_text, max_new_tokens, stop_text
        self.pending = True
        return "opaque bytes outside the action protocol"

    def observe_failure(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        feedback_text: str,
    ) -> object:
        del prompt, inventory
        assert self.pending
        assert cue_text == "exercise the public failure loop"
        assert "opaque bytes outside the action protocol" in feedback_text
        self.pending = False
        self.failure_calls += 1
        self.changed_edges = ("projected-span:newest",)
        self._digest = "malformed-pending-after"
        return self.changed_edges

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> object:
        del prompt, output, inventory, cue_text, strength
        raise AssertionError("malformed output must not enter positive learning")


class _NeverExecution:
    def __init__(self) -> None:
        self.actions: list[object] = []

    def execute(self, action: object) -> object:
        self.actions.append(action)
        raise AssertionError("malformed output must not reach execution")


def _examples() -> tuple[AssemblyPointerTrainingExample, ...]:
    rows = (*_NESTED_ROWS, (_DIRECT_TRAIN_CUE, "!"))
    return tuple(
        AssemblyPointerTrainingExample(
            prompt=f"opaque public observation {index}",
            output=output,
            inventory=_INVENTORY,
            cue_text=cue,
            provenance_id=f"opaque-row-{index}",
        )
        for index, (cue, output) in enumerate(rows)
    )


def _config(
    *,
    plasticity: bool = True,
    ablation: AssemblyAblation = AssemblyAblation.NONE,
) -> AssemblyFactorizedPointerConfig:
    return AssemblyFactorizedPointerConfig(
        stack=AssemblyStackConfig(
            max_cells=2_048,
            max_examples=32,
            max_programs=256,
            max_cue_tokens=8,
            max_sequence_tokens=16,
            max_depth=3,
            max_steps=64,
            max_output_tokens=16,
            random_seed=23,
        ),
        tokenizer=TokenizerConfig(max_phrases=0, min_frequency=2),
        plasticity=plasticity,
        ablation=ablation,
    )


def _grow(
    *,
    plasticity: bool = True,
    ablation: AssemblyAblation = AssemblyAblation.NONE,
) -> AssemblyFactorizedPointerModel:
    return AssemblyFactorizedPointerModel.grow(
        _examples(),
        _config(plasticity=plasticity, ablation=ablation),
    )


def _variable_examples() -> tuple[AssemblyPointerTrainingExample, ...]:
    variable = tuple(
        AssemblyPointerTrainingExample(
            prompt=prompt,
            output=output,
            inventory=_INVENTORY,
            cue_text="assemble opaque fragments",
            provenance_id=f"variable-row-{index}",
        )
        for index, (prompt, output) in enumerate(_VARIABLE_ROWS)
    )
    direct = AssemblyPointerTrainingExample(
        prompt="atom K!",
        output="!",
        inventory=_INVENTORY,
        cue_text="route atom",
        provenance_id="direct-atom",
    )
    return (*variable, direct)


def _grow_variable(
    *,
    plasticity: bool = True,
    ablation: AssemblyAblation = AssemblyAblation.NONE,
) -> AssemblyFactorizedPointerModel:
    config = AssemblyFactorizedPointerConfig(
        stack=AssemblyStackConfig(
            max_cells=8_192,
            max_examples=32,
            max_programs=512,
            max_cue_tokens=64,
            max_sequence_tokens=256,
            max_depth=3,
            max_steps=4_096,
            max_output_tokens=256,
            random_seed=29,
        ),
        tokenizer=TokenizerConfig(max_phrases=128, min_frequency=2),
        plasticity=plasticity,
        ablation=ablation,
    )
    return AssemblyFactorizedPointerModel.grow(_variable_examples(), config)


def _public_prior_examples() -> tuple[
    tuple[AssemblyPointerTrainingExample, ...],
    tuple[tuple[str, SymbolInventory, str, str], ...],
]:
    examples: list[AssemblyPointerTrainingExample] = []
    queries: list[tuple[str, SymbolInventory, str, str]] = []
    for index, case in enumerate(public_coding_prior_v1()):
        frame = ObservationFrame.seal(
            blocks=(
                ObservationBlock.seal("starter-source", case.starter_source),
                ObservationBlock.seal("current-source", case.starter_source),
                ObservationBlock.seal(
                    "typed-signature",
                    json.dumps(
                        {"entry_point": case.entrypoint},
                        separators=(",", ":"),
                        sort_keys=True,
                    ),
                ),
                ObservationBlock.seal("public-task", case.public_prompt),
            ),
            affordances=(_WRITE_AFFORDANCE,),
        )
        prompt = _prompt(frame)
        inventory = python_interface_inventory(
            case.starter_source,
            case.entrypoint,
        )
        action = json.dumps(
            {
                "arguments": {"source": case.successful_source},
                "name": "write_source",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        output = f"<tool>{action}</tool>"
        examples.append(
            AssemblyPointerTrainingExample(
                prompt=prompt,
                output=output,
                inventory=inventory,
                cue_text=case.public_prompt,
                provenance_id=f"public-coding-prior-v1:{index}",
            )
        )
        queries.append((prompt, inventory, case.public_prompt, output))
    return tuple(examples), tuple(queries)


def test_nested_output_is_selected_by_assemblies_then_learns_its_consequence() -> None:
    model = _grow()
    prior = model.training_digest

    output = model.raw_complete(
        "fresh opaque query",
        _INVENTORY,
        cue_text=_FRESH_NESTED_CUE,
        max_new_tokens=16,
    )

    assert output == _FRESH_NESTED_OUTPUT
    assert model.training_digest == prior
    assert model.last_trace is not None
    assert model.last_trace.selected_stop is False
    assert model.last_trace.stack_trace.max_depth == 2
    operations = tuple(
        step.operation
        for step in model.last_trace.stack_trace.steps
        if step.phase == "control"
    )
    assert operations.count("push") == 2
    assert operations.count("pop") == 2
    with pytest.raises(AssemblyFactorizedPointerError, match="pending"):
        model.learn(
            "different query",
            "ignored consequence receipt",
            _INVENTORY,
            cue_text=_FRESH_NESTED_CUE,
        )

    receipt = "authenticated passing receipt, not an output teacher"
    update = model.learn(
        "fresh opaque query",
        receipt,
        _INVENTORY,
        cue_text=_FRESH_NESTED_CUE,
    )

    assert update.valence is Valence.POSITIVE
    assert update.observed_success_sequence is True
    assert update.evidence_sha256 == sha256(receipt.encode()).hexdigest()
    assert update.training_digest != prior
    assert model.training_digest == update.training_digest


def test_no_update_and_stack_off_start_from_identical_learned_state() -> None:
    treatment = _grow()
    no_update = _grow(plasticity=False)
    stack_off = _grow(plasticity=False, ablation=AssemblyAblation.STACK_OFF)
    initial = treatment.training_digest

    assert no_update.training_digest == initial
    assert stack_off.training_digest == initial

    no_update_output = no_update.raw_complete(
        "no-update query",
        _INVENTORY,
        cue_text=_FRESH_NESTED_CUE,
    )
    assert no_update_output == _FRESH_NESTED_OUTPUT
    no_update_result = no_update.observe_failure(
        "no-update query",
        _INVENTORY,
        cue_text=_FRESH_NESTED_CUE,
        feedback_text="scalar failure receipt",
    )
    assert no_update_result.valence is Valence.NEUTRAL
    assert no_update_result.changed_edges == ()
    assert no_update_result.observed_success_sequence is False
    assert no_update.training_digest == initial

    nested_stop = stack_off.raw_complete(
        "stack-off nested query",
        _INVENTORY,
        cue_text=_FRESH_NESTED_CUE,
    )
    assert nested_stop == "<stop>insufficient recurrent assembly evidence</stop>"
    assert stack_off.last_trace is not None
    assert stack_off.last_trace.selected_stop is True
    assert stack_off.training_digest == initial

    direct = stack_off.raw_complete(
        "stack-off direct query",
        _INVENTORY,
        cue_text=_DIRECT_FRESH_CUE,
    )
    assert direct == "?"
    direct_update = stack_off.learn(
        "stack-off direct query",
        "passing receipt",
        _INVENTORY,
        cue_text=_DIRECT_FRESH_CUE,
    )
    assert direct_update.valence is Valence.NEUTRAL
    assert stack_off.training_digest == initial


def test_failure_is_scalar_negative_consequence_not_a_replacement_target() -> None:
    model = _grow()
    prior = model.training_digest

    assert (
        model.raw_complete(
            "direct query",
            _INVENTORY,
            cue_text=_DIRECT_FRESH_CUE,
        )
        == "?"
    )
    receipt = "execution failed without any corrected source"
    update = model.observe_failure(
        "direct query",
        _INVENTORY,
        cue_text=_DIRECT_FRESH_CUE,
        feedback_text=receipt,
    )

    assert update.valence is Valence.NEGATIVE
    assert update.evidence_sha256 == sha256(receipt.encode()).hexdigest()
    assert update.changed_edges
    assert update.observed_success_sequence is False
    assert update.training_digest != prior


def test_public_prior_factors_replay_and_neutral_close_one_checkpoint() -> None:
    examples, queries = _public_prior_examples()
    model = AssemblyFactorizedPointerModel.grow(
        examples,
        AssemblyFactorizedPointerConfig(),
    )
    initial = model.training_digest
    memory = model._fragment_memory
    assert memory is not None
    program = next(item for item in memory._programs if not item.direct)
    slots = tuple(slot for slot in program.slots if slot is not None)
    factor_slot = next(
        slot
        for slot in slots
        if sorted(len(item.support_rows) for item in slot.candidates) == [1, 7]
    )
    matching_slot = next(
        slot
        for slot in slots
        if slot is not factor_slot
        and memory._factor_by_slot[slot.slot_cell]
        == memory._factor_by_slot[factor_slot.slot_cell]
    )
    singleton_slot = next(
        slot
        for slot in slots
        if len(slot.candidates) == 1
        and slot.candidates[0].support_rows == tuple(range(8))
    )
    assert memory._factor_by_slot[factor_slot.slot_cell] == (
        memory._factor_by_slot[matching_slot.slot_cell]
    )
    assert sorted(len(item.support_rows) for item in factor_slot.candidates) == [1, 7]
    assert singleton_slot.candidates[0].support_rows == tuple(range(8))

    saw_singleton_physical_win = False
    for prompt, inventory, cue_text, expected in queries:
        output = model.raw_complete(
            prompt,
            inventory,
            cue_text=cue_text,
            max_new_tokens=512,
        )
        assert output == pointerize(expected, inventory)
        assert model.last_trace is not None
        assert model.last_trace.selected_stop is False
        saw_singleton_physical_win |= any(
            step.phase == "projected-slot"
            and step.operation == "push"
            and len(step.candidate_scores) == 2
            and step.candidate_scores[0] > step.candidate_scores[1]
            for step in model.last_trace.stack_trace.steps
        )
        update = model.resolve_pending_neutral(prompt, inventory, cue_text)
        assert update.polarity == "neutral"
        assert update.valence is Valence.NEUTRAL
        assert update.changed_edges == ()
        assert update.observed_success_sequence is False
        assert update.prior_training_digest == initial
        assert update.training_digest == initial
        assert model.training_digest == initial

    assert saw_singleton_physical_win


def test_inventory_end_cursor_emits_a_novel_changed_cardinality_sequence() -> None:
    training_inventory = SymbolInventory(("solve", "left", "middle", "right"))
    fresh_inventory = SymbolInventory(("fresh_solve", "x", "y"))
    outputs = "PACK[solve](left, middle, right)"
    prompts = (
        "alpha solve left middle right",
        "beta expanded solve left middle right",
        "gamma with extra context solve left middle right",
    )
    examples = tuple(
        AssemblyPointerTrainingExample(
            prompt=prompt,
            output=outputs,
            inventory=training_inventory,
            cue_text="emit ordered roles",
            provenance_id=f"role-loop-{index}",
        )
        for index, prompt in enumerate(prompts)
    )
    config = AssemblyFactorizedPointerConfig(
        stack=AssemblyStackConfig(
            max_cells=4_096,
            max_examples=16,
            max_programs=128,
            max_cue_tokens=256,
            max_sequence_tokens=128,
            max_depth=3,
            max_steps=2_048,
            max_output_tokens=128,
            random_seed=31,
        ),
        tokenizer=TokenizerConfig(max_phrases=0, min_frequency=2),
    )
    model = AssemblyFactorizedPointerModel.grow(examples, config)
    prompt = "delta fresh_solve x y"
    result = model.raw_complete(
        prompt,
        fresh_inventory,
        cue_text="emit ordered roles",
        max_new_tokens=128,
    )
    expected = "PACK[<@0>](<@1>, <@2>)"

    assert result == expected
    assert result not in {
        pointerize(row.output, row.inventory) for row in examples
    }
    assert model.last_trace is not None
    phases = tuple(step.phase for step in model.last_trace.stack_trace.steps)
    assert phases.count("role-binding") >= 2
    assert "role-boundary" in phases
    model.resolve_pending_neutral(prompt, fresh_inventory, "emit ordered roles")

    stack_off = AssemblyFactorizedPointerModel.grow(
        examples,
        AssemblyFactorizedPointerConfig(
            stack=config.stack,
            tokenizer=config.tokenizer,
            plasticity=False,
            ablation=AssemblyAblation.STACK_OFF,
        ),
    )
    assert (
        stack_off.raw_complete(
            prompt,
            fresh_inventory,
            cue_text="emit ordered roles",
            max_new_tokens=128,
        )
        == "<stop>insufficient recurrent assembly evidence</stop>"
    )


def test_variable_fragment_projection_recombines_and_remains_causal() -> None:
    treatment = _grow_variable()
    ordering = _grow_variable(plasticity=False)
    failure = _grow_variable()
    stack_off = _grow_variable(
        plasticity=False,
        ablation=AssemblyAblation.STACK_OFF,
    )
    initial = treatment.training_digest
    assert ordering.training_digest == initial
    assert failure.training_digest == initial
    assert stack_off.training_digest == initial

    forward = ordering.raw_complete(
        "alpha cold lane fresh_entry fresh_argument",
        _FRESH_INVENTORY,
        cue_text="assemble opaque fragments",
        max_new_tokens=256,
    )
    assert "::AAAA::MID::CCCC::" in forward
    ordering.learn(
        "alpha cold lane fresh_entry fresh_argument",
        "neutral close",
        _FRESH_INVENTORY,
        cue_text="assemble opaque fragments",
    )
    reverse = ordering.raw_complete(
        "fresh_argument fresh_entry lane cold alpha",
        _FRESH_INVENTORY,
        cue_text="assemble opaque fragments",
        max_new_tokens=256,
    )
    assert "::AAAA::MID::DDDDDDDD::" in reverse
    ordering.learn(
        "fresh_argument fresh_entry lane cold alpha",
        "neutral close",
        _FRESH_INVENTORY,
        cue_text="assemble opaque fragments",
    )
    assert forward != reverse
    assert ordering.training_digest == initial

    fresh_prompt = "beta lane cold fresh_entry fresh_argument"
    expected_materialized = (
        "HEAD[fresh_entry]::BBBBBBBB::MID::DDDDDDDD::TAIL[fresh_argument]"
    )
    expected_pointer = pointerize(expected_materialized, _FRESH_INVENTORY)
    output = treatment.raw_complete(
        fresh_prompt,
        _FRESH_INVENTORY,
        cue_text="assemble opaque fragments",
        max_new_tokens=256,
    )
    assert output == expected_pointer
    assert materialize(output, _FRESH_INVENTORY) == expected_materialized
    assert treatment.last_trace is not None
    trace = treatment.last_trace.stack_trace
    assert trace.selected_stop is False
    assert trace.max_depth >= 1
    assert len(trace.selected_structural_edges) >= 2
    assert all(
        edge.startswith("projected-span:")
        for edge in trace.selected_structural_edges
    )
    selected_edges = trace.selected_structural_edges
    learned = treatment.learn(
        fresh_prompt,
        "authenticated success receipt only",
        _FRESH_INVENTORY,
        cue_text="assemble opaque fragments",
    )
    assert learned.valence is Valence.POSITIVE
    assert learned.changed_edges == tuple(dict.fromkeys(selected_edges))
    assert learned.observed_success_sequence is True
    assert treatment._fragment_memory is not None
    assert treatment._fragment_memory.prior_contribution_scan_count == 0
    assert treatment._fragment_memory.last_coactivity_write_count > 0
    assert (
        treatment.raw_complete(
            "ΩЖ",
            _INVENTORY,
            cue_text="未知",
            max_new_tokens=256,
        )
        == "<stop>insufficient recurrent assembly evidence</stop>"
    )
    assert treatment.last_trace is not None
    assert treatment.last_trace.selected_stop is True

    assert (
        failure.raw_complete(
            fresh_prompt,
            _FRESH_INVENTORY,
            cue_text="assemble opaque fragments",
            max_new_tokens=256,
        )
        == expected_pointer
    )
    assert failure.last_trace is not None
    failed_choices = failure.last_trace.stack_trace.selected_choice_edges
    assert failed_choices
    failed = failure.observe_failure(
        fresh_prompt,
        _FRESH_INVENTORY,
        cue_text="assemble opaque fragments",
        feedback_text="scalar rejection without replacement output",
    )
    assert failed.valence is Valence.NEGATIVE
    assert failed.changed_edges == (failed_choices[-1],)
    revised = failure.raw_complete(
        fresh_prompt,
        _FRESH_INVENTORY,
        cue_text="assemble opaque fragments",
        max_new_tokens=256,
    )
    assert revised != expected_pointer
    if revised != "<stop>insufficient recurrent assembly evidence</stop>":
        failure.resolve_pending_neutral(
            fresh_prompt,
            _FRESH_INVENTORY,
            "assemble opaque fragments",
        )

    stopped = stack_off.raw_complete(
        fresh_prompt,
        _FRESH_INVENTORY,
        cue_text="assemble opaque fragments",
        max_new_tokens=256,
    )
    assert stopped == "<stop>insufficient recurrent assembly evidence</stop>"
    assert stack_off.last_trace is not None
    assert stack_off.last_trace.selected_stop is True
    assert stack_off.training_digest == initial

    assert (
        stack_off.raw_complete(
            "atom K?",
            _INVENTORY,
            cue_text="route atom",
        )
        == "?"
    )
    stack_off.learn(
        "atom K?",
        "neutral close",
        _INVENTORY,
        cue_text="route atom",
    )
    assert stack_off.training_digest == initial
    assert (
        stack_off.raw_complete(
            "ΩЖ",
            _INVENTORY,
            cue_text="未知",
        )
        == "<stop>insufficient recurrent assembly evidence</stop>"
    )


def test_host_rejected_output_resolves_pending_failure_exactly_once() -> None:
    model = _MalformedPendingModel()
    execution = _NeverExecution()
    frame = ObservationFrame.seal(
        blocks=(
            ObservationBlock.seal(
                "starter-source",
                "def solve(value):\n    raise NotImplementedError\n",
            ),
            ObservationBlock.seal(
                "typed-signature",
                '{"entry_point":"solve"}',
            ),
            ObservationBlock.seal(
                "public-task",
                "exercise the public failure loop",
            ),
        ),
        affordances=(),
    )

    result = PersistentCodingLearner(model, execution).episode(frame)

    assert isinstance(result.decision, StopDecision)
    assert result.decision.reason == "malformed_model_output"
    assert result.learning_applied
    assert result.learning_outcome == "failure"
    assert result.prior_model_digest == "malformed-pending-before"
    assert result.model_digest == "malformed-pending-after"
    assert model.changed_edges == ("projected-span:newest",)
    assert model.failure_calls == 1
    assert model.pending is False
    assert execution.actions == []
    assert result.is_authentic()
