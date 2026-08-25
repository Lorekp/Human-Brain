from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest
from humanbrain.cells import AdaptiveGlifPopulation
from humanbrain.program import (
    LAYER0_END_TOKEN,
    LAYER0_EXPRESSION_TOKENS,
    LAYER0_PROGRAM_TOKEN_ORDER,
    LAYER0_PROMPT_DELIMITER_ID,
    Layer0CellProgramModel,
    Layer0ProgramCompiler,
    Layer0ProgramExample,
    Layer0ProgramFormatError,
    Layer0ProgramGrammar,
    Layer0ProgramModelConfig,
    Layer0ProgramSignature,
)
from humanbrain.recurrent import RecurrentCellCircuit, RecurrentCircuitConfig


def test_scalar_program_grammar_exposes_only_prefixes_that_can_finish() -> None:
    grammar = Layer0ProgramGrammar(max_nodes=3)

    assert grammar.allowed(()) == LAYER0_EXPRESSION_TOKENS
    assert grammar.allowed(("ADD",)) == frozenset({"ARG0", "NEG_ONE", "ZERO", "ONE", "TWO"})
    assert grammar.allowed(("ADD", "ARG0")) == frozenset({"ARG0", "NEG_ONE", "ZERO", "ONE", "TWO"})
    assert grammar.allowed(("ADD", "ARG0", "ONE")) == frozenset({LAYER0_END_TOKEN})
    assert grammar.allowed(("ADD", "ARG0", "ONE", LAYER0_END_TOKEN)) == frozenset()

    with pytest.raises(Layer0ProgramFormatError, match="invalid program prefix"):
        grammar.allowed((LAYER0_END_TOKEN,))


def test_program_compiler_literally_translates_one_complete_cell_program() -> None:
    compiler = Layer0ProgramCompiler(Layer0ProgramGrammar(max_nodes=5))

    source = compiler.compile(
        ("MAX", "ZERO", "ADD", "ARG0", "ONE", LAYER0_END_TOKEN),
        Layer0ProgramSignature(entry_point="bump_nonnegative", parameter_name="value"),
    )

    assert source == ("def bump_nonnegative(value):\n    return max(0, (value + 1))\n")
    with pytest.raises(Layer0ProgramFormatError, match="invalid program prefix"):
        compiler.compile(
            ("ADD", "ARG0", LAYER0_END_TOKEN),
            Layer0ProgramSignature(entry_point="broken", parameter_name="value"),
        )
    with pytest.raises(ValueError, match="valid Python identifier"):
        Layer0ProgramSignature(entry_point="bad-name", parameter_name="value")
    with pytest.raises(Layer0ProgramFormatError, match="invalid program prefix"):
        Layer0ProgramCompiler().compile(
            (
                "ADD",
                "ARG0",
                "ADD",
                "ARG0",
                "ADD",
                "ARG0",
                "ADD",
                "ARG0",
                "ONE",
                LAYER0_END_TOKEN,
            ),
            Layer0ProgramSignature(entry_point="too_large", parameter_name="value"),
        )


def _scalar_first_spike_step(circuit: RecurrentCellCircuit, current_pa: float) -> int:
    cell = AdaptiveGlifPopulation(1, circuit.cell_parameters)
    steps = round(circuit.config.output_race_ms / circuit.cell_parameters.dt_ms)
    for step in range(steps):
        if cell.step(np.asarray([current_pa], dtype=np.float64))[0]:
            return step
    raise AssertionError("test current did not make its fitted output cell spike")


def test_syntax_mask_keeps_the_exact_earliest_allowed_glif5_spike() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=4,
        config=RecurrentCircuitConfig(
            reservoir_cells=16,
            input_fanout=4,
            recurrent_fanout=3,
            random_seed=31,
        ),
    )
    desired_current = np.asarray([1_600.0, 1_250.0, 1_450.0, 1_250.0])

    first_steps = tuple(
        _scalar_first_spike_step(circuit, float(current)) for current in desired_current
    )
    exact_order = tuple(sorted(range(4), key=lambda token: (first_steps[token], token)))
    expected_winner = next(token for token in exact_order if token in {1, 3})
    expected_rejected = tuple(
        token for token in exact_order[: exact_order.index(expected_winner)] if token not in {1, 3}
    )

    before = desired_current.copy()
    race = circuit.race_output_current_pa(
        desired_current,
        allowed_token_ids=(1, 3),
    )

    np.testing.assert_array_equal(desired_current, before)
    assert race.winner == expected_winner == 1
    assert race.simulated_steps == first_steps[expected_winner] + 1
    assert race.excluded_token_ids == expected_rejected == (0, 2)
    assert race.active_cells == 2


def test_all_candidate_fast_race_matches_explicit_candidate_race() -> None:
    circuit = RecurrentCellCircuit(
        token_cells=6,
        config=RecurrentCircuitConfig(
            reservoir_cells=16,
            input_fanout=4,
            recurrent_fanout=3,
            random_seed=37,
        ),
    )
    currents = (
        np.asarray([450.0, 1_250.0, 1_250.0, 800.0, 1_600.0, 1_450.0]),
        np.asarray([1_300.0, 1_300.0, 1_100.0, 900.0, 700.0, 500.0]),
    )

    for current in currents:
        fast = circuit.race_output_current_pa(current)
        explicit = circuit.race_output_current_pa(
            current,
            allowed_token_ids=tuple(range(circuit.token_cells)),
        )

        assert fast.winner == explicit.winner
        assert fast.spike_time_ms == explicit.spike_time_ms
        assert fast.peak_voltage_mv == explicit.peak_voltage_mv
        assert fast.simulated_steps == explicit.simulated_steps
        assert fast.simultaneous_spike_candidates == (explicit.simultaneous_spike_candidates)
        assert fast.discarded_simultaneous_candidates == (
            explicit.discarded_simultaneous_candidates
        )


def test_grown_cell_program_compiles_and_runs_one_public_training_fixture() -> None:
    example = Layer0ProgramExample(
        prompt="Return the input integer plus one.",
        tokens=("ADD", "ARG0", "ONE", LAYER0_END_TOKEN),
    )
    config = Layer0ProgramModelConfig(
        grammar=Layer0ProgramGrammar(max_nodes=5),
        circuit=RecurrentCircuitConfig(
            reservoir_cells=128,
            input_fanout=16,
            recurrent_fanout=8,
            random_seed=17,
        ),
        growth_epochs=3,
    )
    model = Layer0CellProgramModel.grow((example,), config=config)
    checkpoint_before = model.checkpoint_cell_synapse_sha256

    completion = model.complete(example.prompt)
    source = Layer0ProgramCompiler(config.grammar).compile(
        completion.tokens,
        Layer0ProgramSignature(entry_point="increment", parameter_name="value"),
    )
    namespace: dict[str, object] = {}
    exec(source, namespace)

    assert completion.tokens == example.tokens
    assert namespace["increment"](41) == 42
    assert completion.learning_rule == "ridge"
    assert completion.checkpoint_cell_synapse_before == checkpoint_before
    assert completion.checkpoint_cell_synapse_after == checkpoint_before
    assert tuple(step.accepted_token for step in completion.races) == example.tokens
    assert completion.races[0].prefix_sha256 == (
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945"
    )
    assert completion.races[0].allowed_cell_ids == tuple(range(257, 268))
    assert len(completion.races[0].output_current_sha256) == 64


@pytest.mark.parametrize(
    "learning_rule",
    ("ridge", "three_factor", "sparse_three_factor"),
)
def test_program_checkpoint_round_trip_preserves_exact_cell_continuation_for_readout(
    tmp_path: Path,
    learning_rule: str,
) -> None:
    example = Layer0ProgramExample(
        prompt="Return the input integer unchanged.",
        tokens=("ARG0", LAYER0_END_TOKEN),
    )
    config = Layer0ProgramModelConfig(
        grammar=Layer0ProgramGrammar(max_nodes=3),
        circuit=RecurrentCircuitConfig(
            reservoir_cells=64,
            input_fanout=8,
            recurrent_fanout=4,
            readout_learning_rule=learning_rule,
            readout_fan_in=16,
            random_seed=23,
        ),
        growth_epochs=2,
    )
    model = Layer0CellProgramModel.grow((example,), config=config)
    expected = model.complete(example.prompt)
    path = tmp_path / "cell-program.npz"

    artifact_sha256 = model.save(path)
    restored = Layer0CellProgramModel.load(path)
    actual = restored.complete(example.prompt)

    assert len(artifact_sha256) == 64
    assert restored.checkpoint_cell_synapse_sha256 == model.checkpoint_cell_synapse_sha256
    assert actual == expected
    assert actual.learning_rule == learning_rule


def test_sparse_three_factor_remains_a_usable_cell_program_learning_path() -> None:
    example = Layer0ProgramExample(
        prompt="Return the absolute value of the input integer.",
        tokens=("ABS", "ARG0", LAYER0_END_TOKEN),
    )
    config = Layer0ProgramModelConfig(
        grammar=Layer0ProgramGrammar(max_nodes=3),
        circuit=RecurrentCircuitConfig(
            reservoir_cells=64,
            input_fanout=8,
            recurrent_fanout=4,
            readout_learning_rule="sparse_three_factor",
            readout_fan_in=16,
            random_seed=29,
        ),
        growth_epochs=2,
    )

    model = Layer0CellProgramModel.grow((example,), config=config)
    completion = model.complete(example.prompt)
    source = Layer0ProgramCompiler(config.grammar).compile(
        completion.tokens,
        Layer0ProgramSignature(entry_point="candidate", parameter_name="value"),
    )

    assert source.startswith("def candidate(value):\n")
    assert completion.learning_rule == "sparse_three_factor"
    assert completion.checkpoint_cell_synapse_before == completion.checkpoint_cell_synapse_after


def test_circuit_owns_the_public_cell_synapse_checkpoint_boundary(tmp_path: Path) -> None:
    circuit = RecurrentCellCircuit(
        token_cells=4,
        config=RecurrentCircuitConfig(
            reservoir_cells=16,
            input_fanout=4,
            recurrent_fanout=3,
            random_seed=41,
        ),
    )
    circuit.grow_sequence((0, 1, 2, 3), epochs=1)
    circuit.freeze_learning()
    checkpoint = tmp_path / "circuit-state.npz"

    before = circuit.cell_synapse_state_sha256
    artifact_sha256 = circuit.save_cell_synapse_checkpoint(
        checkpoint,
        envelope={"owner": "public-circuit-state-test"},
    )
    envelope, restored = RecurrentCellCircuit.load_cell_synapse_checkpoint(checkpoint)

    assert len(artifact_sha256) == 64
    assert envelope == {"owner": "public-circuit-state-test"}
    assert restored.cell_synapse_state_sha256 == before
    program_source = (Path(__file__).parents[1] / "src" / "humanbrain" / "program.py").read_text(
        encoding="utf-8"
    )
    assert "humanbrain.recurrent_artifacts" not in program_source


def test_layer0_public_protocol_freezes_only_the_plumbing_probe() -> None:
    path = Path(__file__).parents[1] / "benchmarks" / "cell_program_layer0_protocol_v1.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))

    assert set(protocol) == {
        "protocol_version",
        "scope",
        "claim_exclusions",
        "cell_origin_contract",
        "vocabulary",
        "grammar",
        "model",
        "fixtures",
        "dev_family_commitment",
        "anti_overfit",
    }
    assert protocol["protocol_version"] == "cell-program-layer0-public-v1"
    assert protocol["scope"] == "layer-0-cell-program-plumbing-only"
    assert protocol["claim_exclusions"] == [
        "typed-simple-swe-dsl",
        "held-out-generalization",
        "canonical-evaluation",
    ]
    assert protocol["cell_origin_contract"] == {
        "semantic_choice": ("earliest-fitted-glif5-spike-among-grammar-allowed-output-cells"),
        "syntax_mask": "prefix-and-arity-only",
        "search": False,
        "repair": False,
        "compiler_task_text_access": False,
        "compiler_test_access": False,
    }
    assert protocol["vocabulary"] == {
        "prompt_ids": "utf8-bytes-0-through-255",
        "prompt_delimiter_id": LAYER0_PROMPT_DELIMITER_ID,
        "program_token_order": list(LAYER0_PROGRAM_TOKEN_ORDER),
    }
    assert protocol["grammar"] == {
        "value_type": "int",
        "parameter_count": 1,
        "max_nodes": 7,
        "literal_encoding": "named-bounded-layer0-literals",
        "note": (
            "Layer 1 must replace this with signed-digit literals and a "
            "PrimitiveSpec-driven typed hole stack."
        ),
    }
    assert Layer0ProgramGrammar().max_nodes == protocol["grammar"]["max_nodes"]
    assert Layer0ProgramModelConfig().grammar.max_nodes == protocol["grammar"]["max_nodes"]
    default_circuit = RecurrentCircuitConfig()
    # R0 multi-trace schema (protocol §1.2): the deprecated scalar
    # eligibility_tau_ms freezes as null and the canonical tau bank
    # eligibility_taus_ms (geomspace 80 ms -> 4 s, K = 8) rides in its own
    # field; the JSON dump carries the bank as a plain float list.
    expected_circuit = json.loads(json.dumps(asdict(default_circuit)))
    expected_circuit["active_reservoir_cells"] = (
        default_circuit.active_reservoir_cells or default_circuit.reservoir_cells
    )
    assert protocol["model"] == {
        "growth_epochs": 3,
        "circuit": expected_circuit,
    }
    assert Layer0ProgramModelConfig().growth_epochs == protocol["model"]["growth_epochs"]
    assert set(protocol["fixtures"]) == {"public_train", "public_dev_probe"}
    fixtures_payload = json.dumps(
        protocol["fixtures"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert sha256(fixtures_payload).hexdigest() == (
        "073c2c59598b7e6409b06abe953d34546885247edb7a22d4e8813f66438101da"
    )
    assert protocol["dev_family_commitment"] == "unseen-generic-primitive-composition-v1"
    assert protocol["anti_overfit"] == {
        "protocol_and_implementation_commit_before_dev": True,
        "public_dev_run_budget": 1,
        "public_dev_runs_completed": 0,
        "configuration_changes_after_dev": False,
        "all_cases_reported": True,
        "syntax_validity_and_semantic_pass_reported_separately": True,
        "canonical_seed_reserved": False,
        "private_or_reveal_artifacts_allowed": False,
        "external_model_data_allowed": False,
    }
