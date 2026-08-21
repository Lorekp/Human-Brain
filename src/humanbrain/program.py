"""Layer-0 scalar Int-to-Int plumbing for simulated-cell program races.

This deliberately is not the later typed Simple-SWE DSL. Layer 1 must replace
the scalar arity stack and named literals before any generalization claim.
"""

from __future__ import annotations

import json
import keyword
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path

from humanbrain.recurrent import (
    RecurrentCellCircuit,
    RecurrentCircuitConfig,
    RecurrentRace,
)
from humanbrain.recurrent_artifacts import RecurrentCellSynapseState

LAYER0_END_TOKEN = "END"
LAYER0_TERMINAL_TOKENS = frozenset({"ARG0", "NEG_ONE", "ZERO", "ONE", "TWO"})
LAYER0_UNARY_TOKENS = frozenset({"ABS"})
LAYER0_BINARY_TOKENS = frozenset({"ADD", "SUB", "MUL", "MIN", "MAX"})
LAYER0_EXPRESSION_TOKENS = (
    LAYER0_TERMINAL_TOKENS | LAYER0_UNARY_TOKENS | LAYER0_BINARY_TOKENS
)
LAYER0_PROGRAM_TOKEN_ORDER = (
    "ARG0",
    "NEG_ONE",
    "ZERO",
    "ONE",
    "TWO",
    "ABS",
    "ADD",
    "SUB",
    "MUL",
    "MIN",
    "MAX",
    LAYER0_END_TOKEN,
)
LAYER0_PROMPT_DELIMITER_ID = 256
_PROGRAM_TOKEN_TO_ID = {
    token: LAYER0_PROMPT_DELIMITER_ID + 1 + index
    for index, token in enumerate(LAYER0_PROGRAM_TOKEN_ORDER)
}
_PROGRAM_ID_TO_TOKEN = {token_id: token for token, token_id in _PROGRAM_TOKEN_TO_ID.items()}
LAYER0_PROGRAM_CELL_COUNT = LAYER0_PROMPT_DELIMITER_ID + 1 + len(LAYER0_PROGRAM_TOKEN_ORDER)
LAYER0_PROGRAM_CHECKPOINT_FORMAT = "humanbrain-cell-program-layer0-checkpoint-v1"

_ARITY = {
    **{token: 0 for token in LAYER0_TERMINAL_TOKENS},
    **{token: 1 for token in LAYER0_UNARY_TOKENS},
    **{token: 2 for token in LAYER0_BINARY_TOKENS},
}


class Layer0ProgramFormatError(ValueError):
    """The emitted token prefix cannot form one frozen scalar program."""


@dataclass(frozen=True)
class Layer0ProgramGrammar:
    """A bounded, scalar-int prefix grammar used only for syntax masking."""

    max_nodes: int = 9

    def __post_init__(self) -> None:
        if self.max_nodes < 1:
            raise ValueError("max_nodes must be positive")

    def allowed(self, prefix: Sequence[str]) -> frozenset[str]:
        """Return tokens that keep ``prefix`` completable within ``max_nodes``."""
        pending = 1
        nodes = 0
        ended = False
        for token in prefix:
            if ended or token not in LAYER0_EXPRESSION_TOKENS | {LAYER0_END_TOKEN}:
                raise Layer0ProgramFormatError("invalid program prefix")
            if token == LAYER0_END_TOKEN:
                if pending != 0:
                    raise Layer0ProgramFormatError("invalid program prefix")
                ended = True
                continue
            if pending == 0 or nodes >= self.max_nodes:
                raise Layer0ProgramFormatError("invalid program prefix")
            nodes += 1
            pending = pending - 1 + _ARITY[token]
            if nodes + pending > self.max_nodes:
                raise Layer0ProgramFormatError("invalid program prefix")

        if ended:
            return frozenset()
        if pending == 0:
            return frozenset({LAYER0_END_TOKEN})

        allowed = {
            token
            for token, arity in _ARITY.items()
            if nodes + 1 + pending - 1 + arity <= self.max_nodes
        }
        if not allowed:
            raise Layer0ProgramFormatError("invalid program prefix")
        return frozenset(allowed)


@dataclass(frozen=True)
class Layer0ProgramSignature:
    """The frozen scalar-int callable shape supplied to the literal compiler."""

    entry_point: str
    parameter_name: str

    def __post_init__(self) -> None:
        for value in (self.entry_point, self.parameter_name):
            if not value.isidentifier() or keyword.iskeyword(value):
                raise ValueError("program names must be valid Python identifiers")


class Layer0ProgramCompiler:
    """Translate one Layer-0 AST literally; never search or repair it."""

    def __init__(self, grammar: Layer0ProgramGrammar | None = None) -> None:
        self._grammar = grammar or Layer0ProgramGrammar()

    def compile(self, tokens: Sequence[str], signature: Layer0ProgramSignature) -> str:
        emitted = tuple(str(token) for token in tokens)
        if self._grammar.allowed(emitted):
            raise Layer0ProgramFormatError("program is incomplete")
        expression, next_index = self._parse_expression(
            emitted, 0, signature.parameter_name
        )
        if next_index != len(emitted) - 1 or emitted[next_index] != LAYER0_END_TOKEN:
            raise Layer0ProgramFormatError("invalid program prefix")
        return (
            f"def {signature.entry_point}({signature.parameter_name}):\n"
            f"    return {expression}\n"
        )

    def _parse_expression(
        self,
        tokens: tuple[str, ...],
        index: int,
        parameter_name: str,
    ) -> tuple[str, int]:
        token = tokens[index]
        if token == "ARG0":
            return parameter_name, index + 1
        literals = {"NEG_ONE": "-1", "ZERO": "0", "ONE": "1", "TWO": "2"}
        if token in literals:
            return literals[token], index + 1
        if token == "ABS":
            operand, next_index = self._parse_expression(
                tokens, index + 1, parameter_name
            )
            return f"abs({operand})", next_index
        left, next_index = self._parse_expression(tokens, index + 1, parameter_name)
        right, next_index = self._parse_expression(tokens, next_index, parameter_name)
        if token in {"MIN", "MAX"}:
            return f"{token.lower()}({left}, {right})", next_index
        operators = {"ADD": "+", "SUB": "-", "MUL": "*"}
        return f"({left} {operators[token]} {right})", next_index


@dataclass(frozen=True)
class Layer0ProgramExample:
    prompt: str
    tokens: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.prompt:
            raise ValueError("program growth prompt cannot be empty")
        if not self.tokens:
            raise ValueError("program growth tokens cannot be empty")


@dataclass(frozen=True)
class Layer0ProgramModelConfig:
    grammar: Layer0ProgramGrammar = field(default_factory=Layer0ProgramGrammar)
    circuit: RecurrentCircuitConfig = field(default_factory=RecurrentCircuitConfig)
    growth_epochs: int = 3

    def __post_init__(self) -> None:
        if self.growth_epochs < 1:
            raise ValueError("growth_epochs must be positive")


@dataclass(frozen=True)
class Layer0ProgramRaceAudit:
    accepted_token: str
    winner_cell_id: int
    spike_time_ms: float
    simulated_steps: int
    prefix_sha256: str
    allowed_cell_ids: tuple[int, ...]
    excluded_cell_ids: tuple[int, ...]
    output_current_sha256: str


@dataclass(frozen=True)
class Layer0ProgramCompletion:
    tokens: tuple[str, ...]
    races: tuple[Layer0ProgramRaceAudit, ...]
    learning_rule: str
    checkpoint_cell_synapse_before: str
    checkpoint_cell_synapse_after: str

    @property
    def excluded_candidate_count(self) -> int:
        return sum(len(race.excluded_cell_ids) for race in self.races)


def _encode_prompt(prompt: str) -> tuple[int, ...]:
    raw = prompt.encode("utf-8")
    if not raw:
        raise ValueError("program completion prompt cannot be empty")
    return (*raw, LAYER0_PROMPT_DELIMITER_ID)


def _program_token_ids(tokens: Sequence[str]) -> tuple[int, ...]:
    try:
        return tuple(_PROGRAM_TOKEN_TO_ID[token] for token in tokens)
    except KeyError as error:
        raise Layer0ProgramFormatError("unknown program token") from error


class Layer0CellProgramModel:
    """Grow and emit one scalar plumbing program through fitted-cell races."""

    def __init__(
        self,
        *,
        grammar: Layer0ProgramGrammar,
        circuit: RecurrentCellCircuit,
        training_digest: str,
    ) -> None:
        if circuit.token_cells != LAYER0_PROGRAM_CELL_COUNT:
            raise ValueError("program vocabulary and output-cell population must align")
        if not circuit.learning_frozen:
            raise ValueError("cell-program model requires frozen learning")
        self._grammar = grammar
        self._circuit = circuit
        self._training_digest = training_digest

    @classmethod
    def grow(
        cls,
        examples: Sequence[Layer0ProgramExample],
        config: Layer0ProgramModelConfig | None = None,
    ) -> Layer0CellProgramModel:
        cfg = config or Layer0ProgramModelConfig()
        corpus = tuple(examples)
        if not corpus:
            raise ValueError("at least one public program growth example is required")
        circuit = RecurrentCellCircuit(token_cells=LAYER0_PROGRAM_CELL_COUNT, config=cfg.circuit)
        digest = sha256()
        for example in corpus:
            if cfg.grammar.allowed(example.tokens):
                raise Layer0ProgramFormatError("growth program is incomplete")
            sequence = _encode_prompt(example.prompt) + _program_token_ids(example.tokens)
            circuit.grow_sequence(sequence, epochs=cfg.growth_epochs)
            prompt_bytes = example.prompt.encode("utf-8")
            digest.update(len(prompt_bytes).to_bytes(8, "big"))
            digest.update(prompt_bytes)
            digest.update(json.dumps(example.tokens).encode("ascii"))
        circuit.freeze_learning()
        return cls(grammar=cfg.grammar, circuit=circuit, training_digest=digest.hexdigest())

    @property
    def training_digest(self) -> str:
        return self._training_digest

    @property
    def checkpoint_cell_synapse_sha256(self) -> str:
        return RecurrentCellSynapseState.capture(self._circuit).sha256

    def save(self, path: str | Path) -> str:
        state = RecurrentCellSynapseState.capture(self._circuit)
        return state.save_enveloped_checkpoint(
            Path(path),
            {
                "artifact_format": LAYER0_PROGRAM_CHECKPOINT_FORMAT,
                "training_digest": self._training_digest,
                "grammar_max_nodes": self._grammar.max_nodes,
                "program_token_order": LAYER0_PROGRAM_TOKEN_ORDER,
                "prompt_delimiter_id": LAYER0_PROMPT_DELIMITER_ID,
            },
        )

    @classmethod
    def load(cls, path: str | Path) -> Layer0CellProgramModel:
        envelope, state = RecurrentCellSynapseState.load_enveloped_checkpoint(Path(path))
        if envelope.get("artifact_format") != LAYER0_PROGRAM_CHECKPOINT_FORMAT:
            raise ValueError("unsupported cell-program checkpoint format")
        if tuple(envelope.get("program_token_order", ())) != LAYER0_PROGRAM_TOKEN_ORDER:
            raise ValueError("cell-program checkpoint vocabulary mismatch")
        if int(envelope.get("prompt_delimiter_id", -1)) != LAYER0_PROMPT_DELIMITER_ID:
            raise ValueError("cell-program checkpoint delimiter mismatch")
        grammar = Layer0ProgramGrammar(max_nodes=int(envelope["grammar_max_nodes"]))
        circuit = state.restore()
        return cls(
            grammar=grammar,
            circuit=circuit,
            training_digest=str(envelope["training_digest"]),
        )

    def complete(self, prompt: str) -> Layer0ProgramCompletion:
        before = self.checkpoint_cell_synapse_sha256
        self._circuit.reset_sequence()
        for token_id in _encode_prompt(prompt):
            self._circuit.observe(token_id)

        tokens: list[str] = []
        audits: list[Layer0ProgramRaceAudit] = []
        for _ in range(self._grammar.max_nodes + 1):
            allowed_tokens = self._grammar.allowed(tokens)
            if not allowed_tokens:
                break
            prefix_sha256 = sha256(
                json.dumps(tokens, separators=(",", ":")).encode("ascii")
            ).hexdigest()
            allowed_cell_ids = tuple(sorted(_program_token_ids(tuple(allowed_tokens))))
            race: RecurrentRace = self._circuit.race(
                allowed_token_ids=allowed_cell_ids
            )
            accepted = _PROGRAM_ID_TO_TOKEN[race.winner]
            tokens.append(accepted)
            audits.append(
                Layer0ProgramRaceAudit(
                    accepted_token=accepted,
                    winner_cell_id=race.winner,
                    spike_time_ms=race.spike_time_ms,
                    simulated_steps=race.simulated_steps,
                    prefix_sha256=prefix_sha256,
                    allowed_cell_ids=allowed_cell_ids,
                    excluded_cell_ids=race.excluded_token_ids,
                    output_current_sha256=race.output_current_sha256,
                )
            )
            self._circuit.observe(race.winner)
            if accepted == LAYER0_END_TOKEN:
                after = self.checkpoint_cell_synapse_sha256
                return Layer0ProgramCompletion(
                    tokens=tuple(tokens),
                    races=tuple(audits),
                    learning_rule=self._circuit.config.readout_learning_rule,
                    checkpoint_cell_synapse_before=before,
                    checkpoint_cell_synapse_after=after,
                )
        raise RuntimeError("cell program did not terminate within the frozen grammar bound")


__all__ = [
    "LAYER0_BINARY_TOKENS",
    "LAYER0_END_TOKEN",
    "LAYER0_EXPRESSION_TOKENS",
    "LAYER0_PROGRAM_CELL_COUNT",
    "LAYER0_PROGRAM_CHECKPOINT_FORMAT",
    "LAYER0_PROGRAM_TOKEN_ORDER",
    "LAYER0_PROMPT_DELIMITER_ID",
    "LAYER0_TERMINAL_TOKENS",
    "LAYER0_UNARY_TOKENS",
    "Layer0CellProgramModel",
    "Layer0ProgramCompiler",
    "Layer0ProgramCompletion",
    "Layer0ProgramExample",
    "Layer0ProgramFormatError",
    "Layer0ProgramGrammar",
    "Layer0ProgramModelConfig",
    "Layer0ProgramRaceAudit",
    "Layer0ProgramSignature",
]
