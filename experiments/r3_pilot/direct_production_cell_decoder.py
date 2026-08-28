"""Direct prompt assemblies driving locally fitted production-cell races.

This is a reusable architecture spike, not benchmark evidence.  Natural task
text, its typed signature, and the current generic typed-hole role are rebuilt
as sparse fitted-cell assemblies before every decision.  A bounded sparse
readout may change only through its native local teach/turnover operations;
the host supplies the existing type mask and compiler, never a task plan.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from humanbrain import typed_program as tp  # noqa: E402
from humanbrain.cells import AdaptiveGlifPopulation  # noqa: E402
from humanbrain.concept_tokens import ConceptConfig, ConceptTokenizer  # noqa: E402
from humanbrain.readout import SparseCellReadout  # noqa: E402
from humanbrain.sdr_union import SDRProjector  # noqa: E402
from humanbrain.tokenizer import TokenizerConfig  # noqa: E402
from humanbrain.typed_program_types import TypedSignature  # noqa: E402

_ASSEMBLY_CELLS = 1024
_ASSEMBLY_ONES = 8
_READOUT_FAN_IN = _ASSEMBLY_CELLS // SparseCellReadout.CANDIDATE_FACTOR
_INPUT_CURRENT_PA = 5_000.0
_INPUT_STEPS = 30
_OUTPUT_BASE_CURRENT_PA = 600.0
_OUTPUT_RACE_STEPS = 600
_LEARNING_RATE_PA = 32.0
_MAXIMUM_WEIGHT_PA = 1_600.0
_RANDOM_SEED = 17
_CHANNEL_STRIDE = 1_000_000
_CHANNEL_MARKER_BASE = 2_000_000_000
_PAIR_TOKEN_BASE = 16_000_000
_PAIR_TOKEN_BITS = 11
_PAIR_CHANNEL_STRIDE = 1 << (2 * _PAIR_TOKEN_BITS)
_PROMPT_CHANNEL = 1
_SIGNATURE_CHANNEL = 2
_ROLE_CHANNEL = 3

_PRODUCTION_SPECS = tuple(
    spec
    for spec in tp._PRIMITIVE_SPECS
    if spec.role != "input-delimiter"
)
_OUTPUT_BY_TOKEN = {spec.token: index for index, spec in enumerate(_PRODUCTION_SPECS)}
_SPEC_BY_OUTPUT = {index: spec for index, spec in enumerate(_PRODUCTION_SPECS)}


@dataclass(frozen=True, slots=True)
class _HoleCursor:
    path: tuple[int, ...]
    parent: str
    child_index: int


@dataclass(frozen=True, slots=True)
class _FrozenArray:
    name: str
    dtype: str
    shape: tuple[int, ...]
    data: bytes


@dataclass(frozen=True, slots=True)
class ProductionRace:
    position: int
    accepted_token: str
    winner_cell: int
    allowed_tokens: tuple[str, ...]
    spike_time_ms: float
    prompt_cells: int
    signature_cells: int
    role_cells: int
    active_assembly_cells: int


@dataclass(frozen=True, slots=True)
class DirectCompletion:
    program: tp.Program
    source: str
    races: tuple[ProductionRace, ...]
    readout_before_sha256: str
    readout_after_sha256: str


class DirectProductionCellDecoder:
    """Immutable fitted sparse readout plus a task-refreshed cell decoder."""

    def __init__(
        self,
        *,
        tokenizer: ConceptTokenizer,
        arrays: tuple[_FrozenArray, ...],
        counters: dict[str, int],
        output_heterogeneity_pa: np.ndarray,
        pacemaker_synapses_pa: np.ndarray,
        local_updates: int,
        structural_recruitments: int,
    ) -> None:
        self._tokenizer = tokenizer
        self._arrays = arrays
        self._counters = dict(counters)
        self._output_heterogeneity_pa = np.array(
            output_heterogeneity_pa, dtype=np.float64, copy=True
        )
        self._pacemaker_synapses_pa = np.array(
            pacemaker_synapses_pa, dtype=np.float64, copy=True
        )
        self.local_updates = int(local_updates)
        self.structural_recruitments = int(structural_recruitments)
        self.checkpoint_sha256 = _snapshot_sha256(
            arrays,
            self._counters,
            self._output_heterogeneity_pa,
            self._pacemaker_synapses_pa,
        )

    @classmethod
    def fit(
        cls,
        examples: Sequence[tp.Example],
        *,
        epochs: int = 8,
    ) -> DirectProductionCellDecoder:
        """Fit only local sparse synapses from ordinary teacher programs."""
        materialized = tuple(examples)
        if not materialized:
            raise ValueError("direct decoder needs at least one example")
        if epochs < 1:
            raise ValueError("epochs must be positive")
        episode_contexts = _episode_contexts(materialized)
        tokenizer = ConceptTokenizer.fit(
            (
                text
                for prompt, signature, role in episode_contexts
                for text in (prompt, signature, role)
            ),
            config=ConceptConfig(
                min_frequency=2,
                min_mutual_information_bits=0.0,
                max_concepts=128,
                min_tokens=2,
                max_tokens=4,
            ),
            base_config=TokenizerConfig(max_phrases=128, min_frequency=1),
        )
        projector = SDRProjector(
            n_bits=_ASSEMBLY_CELLS,
            w_ones=_ASSEMBLY_ONES,
            seed=_RANDOM_SEED,
        )
        eligibility = np.zeros(_ASSEMBLY_CELLS, dtype=np.float64)
        rng = np.random.default_rng(_RANDOM_SEED)
        output_count = len(_PRODUCTION_SPECS)
        output_heterogeneity_pa = rng.normal(0.0, 0.01, output_count)
        pacemaker_synapses_pa = np.zeros(output_count, dtype=np.float64)
        readout = SparseCellReadout.create(
            output_cells=output_count,
            reservoir_cells=_ASSEMBLY_CELLS,
            fan_in=_READOUT_FAN_IN,
            rng=rng,
            reservoir_eligibility=eligibility,
            output_heterogeneity_pa=output_heterogeneity_pa,
            pacemaker_synapses_pa=pacemaker_synapses_pa,
        )
        trace_decays = np.zeros(_ASSEMBLY_CELLS, dtype=np.float64)
        for _ in range(epochs):
            for example in materialized:
                state = tp._TypedHoleState(example.signature)
                cursors = [_HoleCursor((), "root", -1)]
                for teacher_token in example.teacher_tokens:
                    allowed = state.allowed()
                    teacher = tp._SPEC_BY_TOKEN[teacher_token]
                    if teacher not in allowed:
                        raise ValueError(
                            f"teacher token {teacher_token!r} violates the typed grammar"
                        )
                    role = _role_text(state, cursors[-1])
                    eligibility[:], _ = _drive_assembly(
                        tokenizer,
                        projector,
                        example.prompt,
                        _signature_text(example.signature),
                        role,
                    )
                    allowed_outputs = tuple(_OUTPUT_BY_TOKEN[spec.token] for spec in allowed)
                    winner, _ = _physical_race(
                        readout.current(_OUTPUT_BASE_CURRENT_PA), allowed_outputs
                    )
                    target = _OUTPUT_BY_TOKEN[teacher_token]
                    readout.teach(
                        target=target,
                        winner=winner,
                        learning_rate_pa=_LEARNING_RATE_PA,
                        maximum_weight_pa=_MAXIMUM_WEIGHT_PA,
                        trace_decays=trace_decays,
                        target_post_gate=1.0,
                        winner_post_gate=1.0,
                        activity_decay=0.99,
                    )
                    presentation = readout.advance_target(target)
                    readout.turnover(
                        output=target,
                        structural_event=presentation,
                        random_seed=_RANDOM_SEED,
                        minimum_age=2,
                        turnover_probability=0.0,
                        learning_rate_pa=_LEARNING_RATE_PA,
                        maximum_weight_pa=_MAXIMUM_WEIGHT_PA,
                        homeostatic_rate=0.0,
                    )
                    state.consume(teacher)
                    _advance_cursor(cursors, teacher)
                if not state.complete or cursors:
                    raise ValueError("teacher program is incomplete")
        arrays = _freeze_arrays(readout.checkpoint_arrays())
        return cls(
            tokenizer=tokenizer,
            arrays=arrays,
            counters=readout.checkpoint_counters(),
            output_heterogeneity_pa=output_heterogeneity_pa,
            pacemaker_synapses_pa=pacemaker_synapses_pa,
            local_updates=readout.applied_updates,
            structural_recruitments=readout.structural_recruitments,
        )

    def complete(
        self,
        prompt: str,
        signature: TypedSignature,
        *,
        max_productions: int = 64,
    ) -> DirectCompletion:
        """Refresh all three assemblies and run one legal cell race per hole."""
        if not prompt:
            raise ValueError("prompt must be non-empty")
        eligibility = np.zeros(_ASSEMBLY_CELLS, dtype=np.float64)
        readout = SparseCellReadout.from_checkpoint(
            output_cells=len(_PRODUCTION_SPECS),
            reservoir_cells=_ASSEMBLY_CELLS,
            fan_in=_READOUT_FAN_IN,
            reservoir_eligibility=eligibility,
            output_heterogeneity_pa=self._output_heterogeneity_pa,
            pacemaker_synapses_pa=self._pacemaker_synapses_pa,
            arrays=_thaw_arrays(self._arrays),
            counters=self._counters,
        )
        before = _readout_sha256(readout)
        projector = SDRProjector(
            n_bits=_ASSEMBLY_CELLS,
            w_ones=_ASSEMBLY_ONES,
            seed=_RANDOM_SEED,
        )
        state = tp._TypedHoleState(signature)
        cursors = [_HoleCursor((), "root", -1)]
        tokens: list[str] = []
        races: list[ProductionRace] = []
        signature_text = _signature_text(signature)
        while not state.complete and len(tokens) < max_productions:
            allowed = state.allowed()
            if not allowed or not cursors:
                raise RuntimeError("typed decoder reached an inconsistent hole state")
            role = _role_text(state, cursors[-1])
            eligibility[:], counts = _drive_assembly(
                self._tokenizer,
                projector,
                prompt,
                signature_text,
                role,
            )
            allowed_outputs = tuple(_OUTPUT_BY_TOKEN[spec.token] for spec in allowed)
            winner, step = _physical_race(
                readout.current(_OUTPUT_BASE_CURRENT_PA), allowed_outputs
            )
            accepted = _SPEC_BY_OUTPUT[winner]
            races.append(
                ProductionRace(
                    position=len(tokens),
                    accepted_token=accepted.token,
                    winner_cell=winner,
                    allowed_tokens=tuple(spec.token for spec in allowed),
                    spike_time_ms=(step + 1) * AdaptiveGlifPopulation(1).parameters.dt_ms,
                    prompt_cells=counts[0],
                    signature_cells=counts[1],
                    role_cells=counts[2],
                    active_assembly_cells=int(np.count_nonzero(eligibility)),
                )
            )
            state.consume(accepted)
            _advance_cursor(cursors, accepted)
            tokens.append(accepted.token)
        if not state.complete:
            raise RuntimeError("direct production decoder exceeded its production bound")
        if cursors:
            raise RuntimeError("typed state completed before structural cursor")
        program = tp.Program(tuple(tokens), signature)
        source = tp.compile(program).source
        after = _readout_sha256(readout)
        if after != before:
            raise RuntimeError("completion changed fitted readout state")
        return DirectCompletion(
            program=program,
            source=source,
            races=tuple(races),
            readout_before_sha256=before,
            readout_after_sha256=after,
        )


def _episode_contexts(
    examples: tuple[tp.Example, ...],
) -> tuple[tuple[str, str, str], ...]:
    contexts: list[tuple[str, str, str]] = []
    for example in examples:
        tp.compile(tp.Program(example.teacher_tokens, example.signature))
        state = tp._TypedHoleState(example.signature)
        cursors = [_HoleCursor((), "root", -1)]
        for teacher_token in example.teacher_tokens:
            contexts.append(
                (
                    example.prompt,
                    _signature_text(example.signature),
                    _role_text(state, cursors[-1]),
                )
            )
            spec = tp._SPEC_BY_TOKEN[teacher_token]
            state.consume(spec)
            _advance_cursor(cursors, spec)
    return tuple(contexts)


def _signature_text(signature: TypedSignature) -> str:
    return json.dumps(signature.to_record(), sort_keys=True, separators=(",", ":"))


def _role_text(state: tp._TypedHoleState, cursor: _HoleCursor) -> str:
    if state._literal is None:
        hole = state._holes[-1]
        expected = tp._type_term_record(
            tp._resolve(hole.expected, state._substitutions)
        )
        environment = [
            tp._type_term_record(tp._resolve(term, state._substitutions))
            for term in hole.environment
        ]
        literal: object = None
    else:
        expected = {"constructor": "Int", "arguments": []}
        environment = []
        literal = {
            "digits": len(state._literal.digits),
            "sign": state._literal.sign is not None,
        }
    return json.dumps(
        {
            "child": cursor.child_index,
            "depth": len(cursor.path),
            "environment": environment,
            "expected": expected,
            "literal": literal,
            "parent": cursor.parent,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _advance_cursor(cursors: list[_HoleCursor], spec: tp._PrimitiveSpec) -> None:
    current = cursors[-1]
    if spec.role == "operation":
        cursors.pop()
        for child in reversed(range(len(spec.arguments))):
            cursors.append(_HoleCursor((*current.path, child), spec.token, child))
    elif spec.role in {"binder", "literal"}:
        cursors.pop()
        cursors.append(_HoleCursor((*current.path, 0), spec.token, 0))
    elif spec.role in {"variable", "literal-end"}:
        cursors.pop()
    elif spec.role not in {"sign", "digit"}:
        raise AssertionError(f"unsupported grammar role {spec.role!r}")


def _channel_token(channel: int, token_id: int) -> int:
    return channel * _CHANNEL_STRIDE + int(token_id)


def _channel_pair_token(channel: int, first: int, second: int) -> int:
    limit = 1 << _PAIR_TOKEN_BITS
    if not 0 <= first < limit or not 0 <= second < limit:
        raise ValueError("concept token exceeds the ordered-pair namespace")
    pair = (int(first) << _PAIR_TOKEN_BITS) | int(second)
    return _PAIR_TOKEN_BASE + channel * _PAIR_CHANNEL_STRIDE + pair


def _project_text(
    tokenizer: ConceptTokenizer,
    projector: SDRProjector,
    text: str,
    channel: int,
) -> np.ndarray:
    active = np.zeros(_ASSEMBLY_CELLS, dtype=np.bool_)
    active[projector.indices(_CHANNEL_MARKER_BASE + channel)] = True
    token_ids = tokenizer.encode(text)
    for token_id in token_ids:
        active[projector.indices(_channel_token(channel, token_id))] = True
    for first, second in zip(token_ids, token_ids[1:]):
        active[projector.indices(_channel_pair_token(channel, first, second))] = True
    return active


def _drive_assembly(
    tokenizer: ConceptTokenizer,
    projector: SDRProjector,
    prompt: str,
    signature: str,
    role: str,
) -> tuple[np.ndarray, tuple[int, int, int]]:
    prompt_cells = _project_text(tokenizer, projector, prompt, _PROMPT_CHANNEL)
    signature_cells = _project_text(
        tokenizer, projector, signature, _SIGNATURE_CHANNEL
    )
    role_cells = _project_text(tokenizer, projector, role, _ROLE_CHANNEL)
    active = prompt_cells | signature_cells | role_cells
    population = AdaptiveGlifPopulation(_ASSEMBLY_CELLS)
    current = active.astype(np.float64) * _INPUT_CURRENT_PA
    spikes = np.zeros(_ASSEMBLY_CELLS, dtype=np.float64)
    for _ in range(_INPUT_STEPS):
        spikes += population.step(current)
    if np.any(active & (spikes == 0.0)):
        raise RuntimeError("an active prompt-assembly cell did not spike")
    return spikes, (
        int(np.count_nonzero(prompt_cells)),
        int(np.count_nonzero(signature_cells)),
        int(np.count_nonzero(role_cells)),
    )


def _physical_race(currents: np.ndarray, allowed: tuple[int, ...]) -> tuple[int, int]:
    legal = np.zeros(len(_PRODUCTION_SPECS), dtype=np.bool_)
    legal[np.asarray(allowed, dtype=np.intp)] = True
    injected = np.zeros(len(_PRODUCTION_SPECS), dtype=np.float64)
    injected[legal] = currents[legal]
    population = AdaptiveGlifPopulation(len(_PRODUCTION_SPECS))
    for step in range(_OUTPUT_RACE_STEPS):
        fired = np.flatnonzero(population.step(injected) & legal)
        if fired.size:
            return int(fired[0]), step
    raise RuntimeError("no legal production cell spiked within the race bound")


def _freeze_arrays(arrays: dict[str, np.ndarray]) -> tuple[_FrozenArray, ...]:
    return tuple(
        _FrozenArray(
            name=name,
            dtype=value.dtype.str,
            shape=tuple(int(size) for size in value.shape),
            data=np.ascontiguousarray(value).tobytes(),
        )
        for name, value in sorted(arrays.items())
    )


def _thaw_arrays(arrays: tuple[_FrozenArray, ...]) -> dict[str, np.ndarray]:
    return {
        item.name: np.frombuffer(item.data, dtype=np.dtype(item.dtype))
        .reshape(item.shape)
        .copy()
        for item in arrays
    }


def _snapshot_sha256(
    arrays: tuple[_FrozenArray, ...],
    counters: dict[str, int],
    heterogeneity: np.ndarray,
    pacemaker: np.ndarray,
) -> str:
    digest = sha256(json.dumps(counters, sort_keys=True).encode("ascii"))
    digest.update(np.ascontiguousarray(heterogeneity).tobytes())
    digest.update(np.ascontiguousarray(pacemaker).tobytes())
    for item in arrays:
        digest.update(item.name.encode("ascii"))
        digest.update(item.data)
    return digest.hexdigest()


def _readout_sha256(readout: SparseCellReadout) -> str:
    digest = sha256(
        json.dumps(readout.checkpoint_counters(), sort_keys=True).encode("ascii")
    )
    for item in _freeze_arrays(readout.checkpoint_arrays()):
        digest.update(item.name.encode("ascii"))
        digest.update(item.data)
    return digest.hexdigest()
