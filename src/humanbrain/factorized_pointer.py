"""Factorized cue-to-fragment construction through simulated cell races.

The model learns one shared recurrent output graph.  Public cue words project
onto reusable output fragments; recurrent fragment cells supply structure and
ordering.  Query-specific symbols are bound only after a fragment cell wins,
so a fresh cue combination can construct an output that was never stored as an
episode.

The surrounding host performs lossless tokenization, authentication, and
symbol materialization.  Every fragment and STOP decision is made by a physical
first-spike race over :class:`~humanbrain.cortex.DendriticCortex` cells.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from itertools import pairwise

import numpy as np

from humanbrain.cortex import CellRace, DendriticCortex
from humanbrain.language import ModelConfig
from humanbrain.pointer_binding import SymbolInventory
from humanbrain.tokenizer import BytePhraseTokenizer, TokenizerConfig

__all__ = [
    "BINDING_OFF",
    "FRAGMENT_PROJECTION_OFF",
    "INVENTORY_LOOP_OFF",
    "STOP_OFF",
    "WHOLE_EPISODE_ONLY",
    "FactorizedCompletionTrace",
    "FactorizedLearningUpdate",
    "FactorizedPointerCellModel",
    "FactorizedPointerTrainingExample",
    "FragmentRaceStep",
    "FreshBindingRace",
    "InventoryLoopRace",
]

WHOLE_EPISODE_ONLY = "whole_episode_only"
BINDING_OFF = "binding_off"
FRAGMENT_PROJECTION_OFF = "fragment_projection_off"
INVENTORY_LOOP_OFF = "fixed_arity_loop_off"
STOP_OFF = "stop_off"
_ABLATIONS = frozenset(
    {
        WHOLE_EPISODE_ONLY,
        BINDING_OFF,
        FRAGMENT_PROJECTION_OFF,
        INVENTORY_LOOP_OFF,
        STOP_OFF,
    }
)

_MODEL_SCHEMA = "humanbrain-factorized-pointer-cell-model-v3"
_CUE = re.compile(r"[A-Za-z0-9]+")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_FRAGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\s+|.", re.DOTALL)
_STOP_TONIC = 0.20
_FRESH_SYMBOL_COUNT = 64
_EPSILON = 1e-12

_Context = tuple[int, ...]
_ProjectionEdge = tuple[_Context, int]
_CueProjectionKey = tuple[str, _Context, int]


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )


def _cue_words(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or not text:
        raise ValueError("cue_text must be one non-empty string")
    words = tuple(sorted({match.group(0).lower() for match in _CUE.finditer(text)}))
    if not words:
        raise ValueError("cue_text contains no ASCII-alphanumeric cue")
    return words


def _lex(text: str) -> tuple[str, ...]:
    if not isinstance(text, str) or not text:
        raise ValueError("output must be one non-empty string")
    pieces = tuple(match.group(0) for match in _FRAGMENT.finditer(text))
    if not pieces or "".join(pieces) != text:
        raise ValueError("output could not be losslessly fragmented")
    return pieces


def _inventory_symbols(inventory: SymbolInventory) -> tuple[str, ...]:
    if not isinstance(inventory, SymbolInventory):
        raise TypeError("inventory must be a SymbolInventory")
    # Reconstructing the value re-runs the inventory's authenticity checks.
    return SymbolInventory(inventory.symbols).symbols


def _query_sha256(
    prompt: str,
    inventory: SymbolInventory,
    cue_text: str,
) -> str:
    return sha256(
        _canonical_json(
            {
                "cue_text": cue_text,
                "inventory_sha256": inventory.sha256,
                "prompt": prompt,
            }
        ).encode("ascii")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class FactorizedPointerTrainingExample:
    """One public success used to grow shared cue and fragment synapses."""

    prompt: str
    output: str
    inventory: SymbolInventory
    cue_text: str
    provenance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, str) or not self.prompt:
            raise ValueError("prompt must be one non-empty string")
        _lex(self.output)
        _inventory_symbols(self.inventory)
        _cue_words(self.cue_text)
        if not isinstance(self.provenance_id, str) or not self.provenance_id:
            raise ValueError("provenance_id must be one non-empty string")


@dataclass(frozen=True, slots=True, order=True)
class _FragmentKey:
    kind: str
    index: int
    text: str


_InventoryLoopKey = tuple[int, int, tuple[_FragmentKey, ...]]


_BEGIN = _FragmentKey("begin", -1, "")
_INVENTORY_CONTINUE = _FragmentKey("inventory-continue", -1, "")
_INVENTORY_EXIT = _FragmentKey("inventory-exit", -1, "")
_PROJECTED_CONTEXT = _FragmentKey("projected-context", -1, "")
_STOP = _FragmentKey("stop", -1, "")


def _input(index: int) -> _FragmentKey:
    return _FragmentKey("input", index, "")


def _local(index: int) -> _FragmentKey:
    return _FragmentKey("local", index, "")


def _literal(text: str) -> _FragmentKey:
    return _FragmentKey("literal", -1, text)


def _longest_common_subsequence(
    left: Sequence[_FragmentKey],
    right: Sequence[_FragmentKey],
) -> tuple[_FragmentKey, ...]:
    """Return one deterministic generic fragment alignment backbone."""

    lengths = [[0] * (len(right) + 1) for _ in range(len(left) + 1)]
    for left_index, left_value in enumerate(left, start=1):
        for right_index, right_value in enumerate(right, start=1):
            if left_value == right_value:
                lengths[left_index][right_index] = (
                    lengths[left_index - 1][right_index - 1] + 1
                )
            else:
                lengths[left_index][right_index] = max(
                    lengths[left_index - 1][right_index],
                    lengths[left_index][right_index - 1],
                )
    aligned: list[_FragmentKey] = []
    left_index = len(left)
    right_index = len(right)
    while left_index and right_index:
        if left[left_index - 1] == right[right_index - 1]:
            aligned.append(left[left_index - 1])
            left_index -= 1
            right_index -= 1
        elif lengths[left_index - 1][right_index] >= lengths[left_index][
            right_index - 1
        ]:
            left_index -= 1
        else:
            right_index -= 1
    aligned.reverse()
    return tuple(aligned)


def _aligned_positions(
    row: Sequence[_FragmentKey],
    backbone: Sequence[_FragmentKey],
) -> frozenset[int]:
    """Locate a learned backbone; unselected positions are optional branches."""

    positions: set[int] = set()
    cursor = 0
    for fragment in backbone:
        while cursor < len(row) and row[cursor] != fragment:
            cursor += 1
        if cursor >= len(row):
            raise RuntimeError("learned alignment backbone is absent from one row")
        positions.add(cursor)
        cursor += 1
    return frozenset(positions)


@dataclass(frozen=True, slots=True)
class _InventoryLoopOccurrence:
    start_index: int
    item_count: int
    start_position: int
    end_position: int
    separator: tuple[_FragmentKey, ...]


def _inventory_loop_occurrences(
    row: Sequence[_FragmentKey],
    inventory_size: int,
) -> tuple[_InventoryLoopOccurrence, ...]:
    """Discover maximal repeated traversals over the tail of an inventory."""

    inputs = tuple(
        (position, fragment.index)
        for position, fragment in enumerate(row)
        if fragment.kind == "input"
    )
    occurrences: list[_InventoryLoopOccurrence] = []
    cursor = 0
    while cursor + 2 < len(inputs):
        first_position, first_index = inputs[cursor]
        second_position, second_index = inputs[cursor + 1]
        separator = tuple(row[first_position + 1 : second_position])
        if (
            second_index != first_index + 1
            or not separator
            or any(fragment.kind != "literal" for fragment in separator)
        ):
            cursor += 1
            continue

        end = cursor + 1
        while end + 1 < len(inputs):
            left_position, left_index = inputs[end]
            right_position, right_index = inputs[end + 1]
            if (
                right_index != left_index + 1
                or tuple(row[left_position + 1 : right_position]) != separator
            ):
                break
            end += 1

        end_position, end_index = inputs[end]
        item_count = end - cursor + 1
        if item_count >= 3 and end_index == inventory_size - 1:
            occurrences.append(
                _InventoryLoopOccurrence(
                    start_index=first_index,
                    item_count=item_count,
                    start_position=first_position,
                    end_position=end_position,
                    separator=separator,
                )
            )
            cursor = end + 1
        else:
            cursor += 1
    return tuple(occurrences)


@dataclass(frozen=True, slots=True)
class FragmentRaceStep:
    """One auditable physical fragment-cell competition."""

    step: int
    candidate_cells: tuple[int, ...]
    candidate_scores: tuple[float, ...]
    winner_cell: int
    winner_kind: str
    winner_text: str
    spike_time_ms: float
    simulated_steps: int


@dataclass(frozen=True, slots=True)
class FreshBindingRace:
    """Physical allocation of one output-local identifier assembly."""

    slot: int
    winner_cell: int
    symbol: str
    candidate_count: int
    spike_time_ms: float
    simulated_steps: int


@dataclass(frozen=True, slots=True)
class InventoryLoopRace:
    """One physical continue-or-exit decision after a bound inventory item."""

    item_index: int
    candidate_cells: tuple[int, int]
    candidate_scores: tuple[float, float]
    winner_cell: int
    decision: str
    spike_time_ms: float
    simulated_steps: int


@dataclass(frozen=True, slots=True)
class FactorizedCompletionTrace:
    """Physical custody for one factorized completion."""

    query_sha256: str
    model_digest: str
    ablation: str | None
    cue_words: tuple[str, ...]
    fragment_steps: tuple[FragmentRaceStep, ...]
    binding_races: tuple[FreshBindingRace, ...]
    inventory_loop_races: tuple[InventoryLoopRace, ...]
    emitted_fragment_cells: tuple[int, ...]
    emitted_token_ids: tuple[int, ...]
    selected_stop: bool
    terminal_stop: bool
    output: str


@dataclass(frozen=True, slots=True)
class FactorizedLearningUpdate:
    """Digest-changing signed plasticity applied to selected synapses."""

    prior_training_digest: str
    training_digest: str
    polarity: str
    synapses_changed: int


@dataclass(frozen=True, slots=True)
class _LastCompletion:
    query_sha256: str
    cue_words: tuple[str, ...]
    projected_edges: tuple[_ProjectionEdge, ...]
    output: str


@dataclass(frozen=True, slots=True)
class _InventoryLoop:
    start_index: int
    demonstrated_count: int
    item_cell: int
    separator_cells: tuple[int, ...]
    start_edges: frozenset[_ProjectionEdge]
    continue_occupied_weight: float
    continue_exhausted_weight: float
    exit_occupied_weight: float
    exit_exhausted_weight: float


class FactorizedPointerCellModel:
    """Construct outputs from shared recurrent fragments and cue projections."""

    def __init__(
        self,
        *,
        tokenizer: BytePhraseTokenizer,
        cortex: DendriticCortex,
        fragments: tuple[_FragmentKey, ...],
        transitions: dict[_Context, dict[int, float]],
        cue_weights: dict[_CueProjectionKey, float],
        projection_edges: frozenset[_ProjectionEdge],
        canonical_edges: frozenset[_ProjectionEdge],
        projection_context_cell: int,
        inventory_loops: tuple[_InventoryLoop, ...],
        inventory_loop_cue_weights: dict[str, float],
        episode_roots: tuple[str, ...],
        reserved_identifiers: frozenset[str],
        fresh_cells: tuple[int, ...],
        fresh_symbols: tuple[str, ...],
        max_order: int,
    ) -> None:
        if len(fragments) + len(fresh_cells) > cortex.cell_count:
            raise ValueError("physical cortex is smaller than the learned cell bank")
        if max_order < 1:
            raise ValueError("max_order must be positive")
        self._tokenizer = tokenizer
        self._cortex = cortex
        self._fragments = fragments
        self._fragment_to_cell = {
            fragment: cell for cell, fragment in enumerate(fragments)
        }
        self._transitions = transitions
        self._cue_weights = cue_weights
        self._plasticity_ticks: dict[_CueProjectionKey, float] = {}
        self._projection_edges = projection_edges
        self._canonical_edges = canonical_edges
        self._projection_context_cell = projection_context_cell
        self._inventory_loops = inventory_loops
        self._inventory_loop_by_edge = {
            edge: loop
            for loop in inventory_loops
            for edge in loop.start_edges
        }
        self._inventory_loop_cue_weights = inventory_loop_cue_weights
        self._episode_roots = episode_roots
        self._reserved_identifiers = reserved_identifiers
        self._fresh_cells = fresh_cells
        self._fresh_symbols = fresh_symbols
        self._max_order = max_order
        self._plasticity_root = sha256(
            b"humanbrain-factorized-pointer-plasticity-initial-v1"
        ).hexdigest()
        self._training_digest_cache: str | None = None
        self._last_trace: FactorizedCompletionTrace | None = None
        self._last_completion: _LastCompletion | None = None

    @classmethod
    def grow(
        cls,
        examples: Iterable[FactorizedPointerTrainingExample],
        config: ModelConfig | None = None,
    ) -> FactorizedPointerCellModel:
        example_tuple = tuple(examples)
        if len(example_tuple) < 2:
            raise ValueError("factorized growth needs at least two public successes")
        if any(
            not isinstance(example, FactorizedPointerTrainingExample)
            for example in example_tuple
        ):
            raise TypeError(
                "examples must contain FactorizedPointerTrainingExample values"
            )
        if len({example.provenance_id for example in example_tuple}) != len(
            example_tuple
        ):
            raise ValueError("training provenance ids must be unique")

        cfg = config or ModelConfig()
        if cfg.max_order < 1:
            raise ValueError("factorized growth requires max_order >= 1")
        token_rows = tuple(_lex(example.output) for example in example_tuple)
        node_rows: list[list[_FragmentKey | None]] = [
            [None] * len(tokens) for tokens in token_rows
        ]

        for row_index, (example, tokens) in enumerate(
            zip(example_tuple, token_rows, strict=True)
        ):
            symbol_positions = {
                symbol: index for index, symbol in enumerate(example.inventory.symbols)
            }
            for position, token in enumerate(tokens):
                input_index = symbol_positions.get(token)
                if input_index is not None:
                    node_rows[row_index][position] = _input(input_index)

        # Repetition discovers lexical equivalence without a language parser.
        # Roles are aligned by first occurrence, so differently named locals in
        # differently sized rows can share one generative/reusable symbol slot.
        token_row_sets = tuple(frozenset(row) for row in token_rows)
        for row_index, (example, tokens) in enumerate(
            zip(example_tuple, token_rows, strict=True)
        ):
            counts = Counter(tokens)
            input_symbols = frozenset(example.inventory.symbols)
            repeated_locals = tuple(
                token
                for token in dict.fromkeys(tokens)
                if counts[token] >= 2
                and _IDENTIFIER.fullmatch(token) is not None
                and token not in input_symbols
                and len(
                    {
                        tokens[position + 1]
                        if position + 1 < len(tokens)
                        else ""
                        for position, value in enumerate(tokens)
                        if value == token
                    }
                )
                > 1
                and not any(
                    token in other_tokens
                    for other_index, other_tokens in enumerate(token_row_sets)
                    if other_index != row_index
                )
            )
            for slot, token in enumerate(repeated_locals):
                marker = _local(slot)
                for position, value in enumerate(tokens):
                    if value == token and node_rows[row_index][position] is None:
                        node_rows[row_index][position] = marker

        for row_index, row in enumerate(node_rows):
            for position, node in enumerate(row):
                if node is None:
                    row[position] = _literal(token_rows[row_index][position])

        normalized_rows = tuple(
            tuple(node for node in row if node is not None) for row in node_rows
        )
        if any(
            len(row) != len(tokens)
            for row, tokens in zip(normalized_rows, token_rows, strict=True)
        ):
            raise RuntimeError("factorized output normalization left an empty fragment")

        loop_occurrence_maps = tuple(
            {
                (
                    occurrence.start_index,
                    occurrence.item_count,
                    occurrence.separator,
                ): occurrence
                for occurrence in _inventory_loop_occurrences(
                    row,
                    len(example.inventory.symbols),
                )
            }
            for example, row in zip(example_tuple, normalized_rows, strict=True)
        )
        common_loop_keys = set(loop_occurrence_maps[0])
        for occurrence_map in loop_occurrence_maps[1:]:
            common_loop_keys.intersection_update(occurrence_map)
        inventory_loop_keys = tuple(sorted(common_loop_keys))
        loop_exit_positions: list[set[int]] = [
            set() for _ in normalized_rows
        ]
        loop_start_keys: list[defaultdict[int, list[_InventoryLoopKey]]] = [
            defaultdict(list) for _ in normalized_rows
        ]
        for row_index, occurrence_map in enumerate(loop_occurrence_maps):
            for key in inventory_loop_keys:
                occurrence = occurrence_map[key]
                loop_exit_positions[row_index].add(occurrence.end_position)
                loop_start_keys[row_index][occurrence.start_position].append(key)
        loop_separator_fragments = {
            key: tuple(
                _FragmentKey("inventory-separator", position, fragment.text)
                for position, fragment in enumerate(key[2])
            )
            for key in inventory_loop_keys
        }

        alignment_order = sorted(
            range(len(normalized_rows)),
            key=lambda index: (
                len(normalized_rows[index]),
                tuple(normalized_rows[index]),
            ),
        )
        backbone = normalized_rows[alignment_order[0]]
        for row_index in alignment_order[1:]:
            backbone = _longest_common_subsequence(
                backbone,
                normalized_rows[row_index],
            )
        structural_positions = tuple(
            _aligned_positions(row, backbone) for row in normalized_rows
        )
        projection_positions = tuple(
            frozenset(range(len(row))) - structural
            for row, structural in zip(
                normalized_rows,
                structural_positions,
                strict=True,
            )
        )
        canonical_positions = tuple(
            frozenset(
                position
                for position in positions
                if position + 1 not in positions
            )
            for positions in projection_positions
        )

        fragment_set = {_BEGIN, _PROJECTED_CONTEXT, _STOP}
        for row in normalized_rows:
            fragment_set.update(row)
        if inventory_loop_keys:
            fragment_set.update({_INVENTORY_CONTINUE, _INVENTORY_EXIT})
            for separator_fragments in loop_separator_fragments.values():
                fragment_set.update(separator_fragments)
        fragments = tuple(sorted(fragment_set))
        fragment_to_cell = {
            fragment: cell for cell, fragment in enumerate(fragments)
        }
        begin_cell = fragment_to_cell[_BEGIN]
        projection_context_cell = fragment_to_cell[_PROJECTED_CONTEXT]
        stop_cell = fragment_to_cell[_STOP]
        inventory_exit_cell = (
            fragment_to_cell[_INVENTORY_EXIT] if inventory_loop_keys else -1
        )
        inventory_continue_cell = (
            fragment_to_cell[_INVENTORY_CONTINUE] if inventory_loop_keys else -1
        )

        transitions_mutable: defaultdict[_Context, Counter[int]] = defaultdict(Counter)
        canonical_edges_mutable: set[_ProjectionEdge] = set()
        loop_start_edges_mutable: dict[_InventoryLoopKey, set[_ProjectionEdge]] = {
            key: set() for key in inventory_loop_keys
        }
        row_edges: list[set[_ProjectionEdge]] = [
            set() for _ in normalized_rows
        ]
        for row_index, row in enumerate(normalized_rows):
            cells = tuple(fragment_to_cell[node] for node in row)
            row_canonical_positions = canonical_positions[row_index]
            context_cells = tuple(
                inventory_exit_cell
                if position in loop_exit_positions[row_index]
                else (
                    projection_context_cell
                    if position in row_canonical_positions
                    else cell
                )
                for position, cell in enumerate(cells)
            )
            sequence = (begin_cell, *cells, stop_cell)
            context_sequence = (begin_cell, *context_cells, stop_cell)
            for target_index in range(1, len(sequence)):
                canonical_target = target_index - 1 in row_canonical_positions
                available = min(cfg.max_order, target_index)
                for order in range(1, available + 1):
                    context = tuple(
                        context_sequence[target_index - order : target_index]
                    )
                    target = sequence[target_index]
                    transitions_mutable[context][target] += 1
                    edge = (context, target)
                    row_edges[row_index].add(edge)
                    for loop_key in loop_start_keys[row_index].get(
                        target_index - 1,
                        (),
                    ):
                        loop_start_edges_mutable[loop_key].add(edge)
                    if canonical_target:
                        canonical_edges_mutable.add(edge)
        for key in inventory_loop_keys:
            start_index, _, _ = key
            recurrent_loop_cells = (
                inventory_continue_cell,
                *(
                    fragment_to_cell[fragment]
                    for fragment in loop_separator_fragments[key]
                ),
                fragment_to_cell[_input(start_index)],
            )
            for source, target in pairwise(recurrent_loop_cells):
                transitions_mutable[(source,)][target] += len(example_tuple)
        transitions = {
            context: {
                target: count / sum(branch.values())
                for target, count in branch.items()
            }
            for context, branch in transitions_mutable.items()
        }

        projection_edges = frozenset(
            (context, target)
            for context, branch in transitions.items()
            if len(branch) > 1
            for target in branch
            if target != stop_cell
        )
        canonical_edges = frozenset(canonical_edges_mutable)
        cue_sets = tuple(
            frozenset(_cue_words(example.cue_text)) for example in example_tuple
        )
        inventory_loops = tuple(
            _InventoryLoop(
                start_index=start_index,
                demonstrated_count=item_count,
                item_cell=fragment_to_cell[_input(start_index)],
                separator_cells=tuple(
                    fragment_to_cell[fragment]
                    for fragment in loop_separator_fragments[key]
                ),
                start_edges=frozenset(loop_start_edges_mutable[key]),
                # Interior observations teach occupied -> continue; the
                # terminal boundary teaches exhausted -> exit.
                continue_occupied_weight=1.0,
                continue_exhausted_weight=0.0,
                exit_occupied_weight=0.0,
                exit_exhausted_weight=1.0,
            )
            for key in inventory_loop_keys
            for start_index, item_count, separator in (key,)
        )
        inventory_loop_cue_counts = Counter(
            cue for cue_set in cue_sets for cue in cue_set
        )
        max_inventory_loop_cue_support = max(
            inventory_loop_cue_counts.values(),
            default=1,
        )
        inventory_loop_cue_weights = (
            {
                cue: count / max_inventory_loop_cue_support
                for cue, count in inventory_loop_cue_counts.items()
            }
            if inventory_loops
            else {}
        )
        all_cues = sorted(set().union(*cue_sets))
        cue_weights: dict[_CueProjectionKey, float] = {}
        for cue in all_cues:
            positive = tuple(
                index for index, cues in enumerate(cue_sets) if cue in cues
            )
            negative = tuple(
                index for index, cues in enumerate(cue_sets) if cue not in cues
            )
            if not positive or not negative:
                continue
            for context, cell in projection_edges:
                edge = (context, cell)
                positive_count = sum(
                    edge in row_edges[row_index]
                    for row_index in positive
                )
                negative_count = sum(
                    edge in row_edges[row_index]
                    for row_index in negative
                )
                weight = positive_count / len(positive) - negative_count / len(
                    negative
                )
                if abs(weight) > _EPSILON:
                    cue_weights[(cue, context, cell)] = weight

        tokenizer = BytePhraseTokenizer.fit(
            (example.output for example in example_tuple),
            TokenizerConfig(
                max_phrases=cfg.max_phrases,
                min_frequency=cfg.min_phrase_frequency,
            ),
        )
        fresh_symbols = tuple(f"_v{index}" for index in range(_FRESH_SYMBOL_COUNT))
        fresh_cells = tuple(
            range(len(fragments), len(fragments) + len(fresh_symbols))
        )
        cortex = DendriticCortex(
            cell_count=len(fragments) + len(fresh_symbols),
            max_order=1,
            parameters=cfg.cell_parameters,
            seed=cfg.random_seed,
        )
        reserved_identifiers = frozenset(
            token
            for example, tokens in zip(example_tuple, token_rows, strict=True)
            for token in (
                *tokens,
                *example.inventory.symbols,
                *_cue_words(example.cue_text),
            )
            if _IDENTIFIER.fullmatch(token) is not None
        )
        episode_roots = tuple(
            sha256(
                _canonical_json(
                    {
                        "cue_text": example.cue_text,
                        "inventory_sha256": example.inventory.sha256,
                        "output_sha256": sha256(
                            example.output.encode("utf-8")
                        ).hexdigest(),
                        "prompt_sha256": sha256(
                            example.prompt.encode("utf-8")
                        ).hexdigest(),
                        "provenance_id": example.provenance_id,
                    }
                ).encode("ascii")
            ).hexdigest()
            for example in example_tuple
        )
        return cls(
            tokenizer=tokenizer,
            cortex=cortex,
            fragments=fragments,
            transitions=transitions,
            cue_weights=cue_weights,
            projection_edges=projection_edges,
            canonical_edges=canonical_edges,
            projection_context_cell=projection_context_cell,
            inventory_loops=inventory_loops,
            inventory_loop_cue_weights=inventory_loop_cue_weights,
            episode_roots=episode_roots,
            reserved_identifiers=reserved_identifiers,
            fresh_cells=fresh_cells,
            fresh_symbols=fresh_symbols,
            max_order=cfg.max_order,
        )

    @property
    def training_digest(self) -> str:
        if self._training_digest_cache is not None:
            return self._training_digest_cache
        transitions = [
            {
                "context": context,
                "targets": tuple(
                    (target, float(weight).hex())
                    for target, weight in sorted(branch.items())
                ),
            }
            for context, branch in sorted(self._transitions.items())
        ]
        payload = {
            "canonical_edges": tuple(sorted(self._canonical_edges)),
            "cue_weights": tuple(
                (cue, context, cell, float(weight).hex())
                for (cue, context, cell), weight in sorted(
                    self._cue_weights.items()
                )
            ),
            "episode_roots": self._episode_roots,
            "fragments": tuple(
                (fragment.kind, fragment.index, fragment.text)
                for fragment in self._fragments
            ),
            "inventory_loop_cue_weights": tuple(
                (cue, float(weight).hex())
                for cue, weight in sorted(
                    self._inventory_loop_cue_weights.items()
                )
            ),
            "inventory_loops": tuple(
                {
                    "demonstrated_count": loop.demonstrated_count,
                    "item_cell": loop.item_cell,
                    "separator_cells": loop.separator_cells,
                    "start_edges": tuple(sorted(loop.start_edges)),
                    "start_index": loop.start_index,
                    "continue_occupied_weight": float(
                        loop.continue_occupied_weight
                    ).hex(),
                    "continue_exhausted_weight": float(
                        loop.continue_exhausted_weight
                    ).hex(),
                    "exit_occupied_weight": float(
                        loop.exit_occupied_weight
                    ).hex(),
                    "exit_exhausted_weight": float(
                        loop.exit_exhausted_weight
                    ).hex(),
                }
                for loop in self._inventory_loops
            ),
            "max_order": self._max_order,
            "phrases": tuple(piece.hex() for piece in self._tokenizer.phrases),
            "plasticity_root": self._plasticity_root,
            "plasticity_ticks": tuple(
                (cue, context, cell, float(weight).hex())
                for (cue, context, cell), weight in sorted(
                    self._plasticity_ticks.items()
                )
            ),
            "projection_edges": tuple(sorted(self._projection_edges)),
            "schema": _MODEL_SCHEMA,
            "transitions": transitions,
        }
        digest = sha256(_canonical_json(payload).encode("ascii")).hexdigest()
        self._training_digest_cache = digest
        return digest

    @property
    def grown_branches(self) -> int:
        recurrent = sum(len(branch) for branch in self._transitions.values())
        cue = (
            len(self._cue_weights)
            + len(self._plasticity_ticks)
            + len(self._inventory_loop_cue_weights)
        )
        inventory = 4 * len(self._inventory_loops)
        return recurrent + cue + inventory

    @property
    def cell_count(self) -> int:
        return self._cortex.cell_count

    @property
    def vocabulary_size(self) -> int:
        return self._tokenizer.vocab_size

    @property
    def last_trace(self) -> FactorizedCompletionTrace | None:
        return self._last_trace

    def _branch(self, history: Sequence[int]) -> tuple[_Context, dict[int, float]]:
        available = min(self._max_order, len(history))
        for order in range(available, 0, -1):
            context = tuple(history[-order:])
            branch = self._transitions.get(context)
            if branch:
                return context, branch
        return (), {}

    def _cue_drive(
        self,
        cues: Sequence[str],
        context: _Context,
        cell: int,
    ) -> float:
        return sum(
            self._cue_weights.get((cue, context, cell), 0.0)
            + self._plasticity_ticks.get((cue, context, cell), 0.0)
            for cue in cues
        )

    def _race(
        self,
        cells: Sequence[int],
        scores: Sequence[float],
    ) -> CellRace:
        active = np.asarray(cells, dtype=np.int64)
        association = np.maximum(np.asarray(scores, dtype=np.float64), 0.0)
        parameters = self._cortex.parameters
        return self._cortex.race_currents(
            active,
            association
            * parameters.dendritic_probability_gain
            * parameters.branch_current_pa,
            fallback_current_pa=(
                association * parameters.dendritic_probability_gain + 6.0
            )
            * parameters.branch_current_pa,
        )

    def _fresh_binding(
        self,
        slot: int,
        forbidden: frozenset[str],
    ) -> tuple[str, FreshBindingRace]:
        available = tuple(
            (cell, symbol)
            for cell, symbol in zip(
                self._fresh_cells,
                self._fresh_symbols,
                strict=True,
            )
            if symbol not in forbidden
        )
        if not available:
            raise RuntimeError("fresh binding cell bank is exhausted")
        race = self._race(
            tuple(cell for cell, _ in available),
            (1.0,) * len(available),
        )
        symbol_by_cell = dict(available)
        symbol = symbol_by_cell[race.winner]
        return symbol, FreshBindingRace(
            slot=slot,
            winner_cell=race.winner,
            symbol=symbol,
            candidate_count=len(available),
            spike_time_ms=race.spike_time_ms,
            simulated_steps=race.simulated_steps,
        )

    @staticmethod
    def _input_placeholder(index: int) -> str:
        return "<@input>" if index == 0 else f"<@input:{index}>"

    @staticmethod
    def _inventory_placeholder(index: int) -> str:
        return f"<@inventory:{index}>"

    @staticmethod
    def _local_placeholder(index: int) -> str:
        return "<@local>" if index == 0 else f"<@local:{index}>"

    def raw_complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 64,
        stop_text: str | tuple[str, ...] | None = None,
        ablation: str | None = None,
    ) -> str:
        if not isinstance(prompt, str) or not prompt:
            raise ValueError("prompt must be one non-empty string")
        symbols = _inventory_symbols(inventory)
        cues = _cue_words(cue_text)
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if ablation is not None and ablation not in _ABLATIONS:
            raise ValueError(f"unknown factorized ablation: {ablation}")
        if stop_text is not None:
            markers = (stop_text,) if isinstance(stop_text, str) else tuple(stop_text)
            if any(not isinstance(marker, str) or not marker for marker in markers):
                raise ValueError("stop markers must be non-empty strings")

        query_sha = _query_sha256(prompt, inventory, cue_text)
        model_digest = self.training_digest
        stop_cell = self._fragment_to_cell[_STOP]
        if ablation == WHOLE_EPISODE_ONLY:
            race = self._race((stop_cell,), (1.0,))
            output = "<stop>whole episode miss</stop>"
            step = FragmentRaceStep(
                step=0,
                candidate_cells=(stop_cell,),
                candidate_scores=(1.0,),
                winner_cell=race.winner,
                winner_kind="stop",
                winner_text="",
                spike_time_ms=race.spike_time_ms,
                simulated_steps=race.simulated_steps,
            )
            self._last_trace = FactorizedCompletionTrace(
                query_sha256=query_sha,
                model_digest=model_digest,
                ablation=ablation,
                cue_words=cues,
                fragment_steps=(step,),
                binding_races=(),
                inventory_loop_races=(),
                emitted_fragment_cells=(),
                emitted_token_ids=tuple(self._tokenizer.encode(output)),
                selected_stop=True,
                terminal_stop=False,
                output=output,
            )
            self._last_completion = _LastCompletion(query_sha, cues, (), output)
            return output

        projection_enabled = ablation != FRAGMENT_PROJECTION_OFF
        binding_enabled = ablation != BINDING_OFF
        inventory_loop_enabled = ablation != INVENTORY_LOOP_OFF
        stop_enabled = ablation != STOP_OFF
        inventory_loop_cue_drive = sum(
            self._inventory_loop_cue_weights.get(cue, 0.0) for cue in cues
        )
        history = [self._fragment_to_cell[_BEGIN]]
        emitted_cells: list[int] = []
        emitted_text: list[str] = []
        projected_edges: list[_ProjectionEdge] = []
        fragment_steps: list[FragmentRaceStep] = []
        binding_races: list[FreshBindingRace] = []
        inventory_loop_races: list[InventoryLoopRace] = []
        local_bindings: dict[int, str] = {}
        forbidden = frozenset(
            {
                *self._reserved_identifiers,
                *symbols,
                *cues,
            }
        )
        selected_stop = False
        terminal_stop = False
        active_inventory_loop: _InventoryLoop | None = None
        inventory_item_index = -1

        for _ in range(max_new_tokens):
            if (
                active_inventory_loop is not None
                and history[-1] == active_inventory_loop.item_cell
            ):
                continue_cell = self._fragment_to_cell[_INVENTORY_CONTINUE]
                exit_cell = self._fragment_to_cell[_INVENTORY_EXIT]
                next_item_occupied = inventory_item_index + 1 < len(symbols)
                occupied_drive = float(next_item_occupied)
                exhausted_drive = 1.0 - occupied_drive
                decision_scores = (
                    occupied_drive
                    * active_inventory_loop.continue_occupied_weight
                    + exhausted_drive
                    * active_inventory_loop.continue_exhausted_weight,
                    occupied_drive * active_inventory_loop.exit_occupied_weight
                    + exhausted_drive
                    * active_inventory_loop.exit_exhausted_weight,
                )
                decision_race = self._race(
                    (continue_cell, exit_cell),
                    decision_scores,
                )
                decision = (
                    "continue"
                    if decision_race.winner == continue_cell
                    else "exit"
                )
                inventory_loop_races.append(
                    InventoryLoopRace(
                        item_index=inventory_item_index,
                        candidate_cells=(continue_cell, exit_cell),
                        candidate_scores=decision_scores,
                        winner_cell=decision_race.winner,
                        decision=decision,
                        spike_time_ms=decision_race.spike_time_ms,
                        simulated_steps=decision_race.simulated_steps,
                    )
                )
                history.append(decision_race.winner)
                if decision == "exit":
                    active_inventory_loop = None
                else:
                    inventory_item_index += 1
                continue

            context, branch = self._branch(history)
            candidate_set = set(branch)
            if stop_enabled:
                candidate_set.add(stop_cell)
            else:
                candidate_set.discard(stop_cell)
            if not candidate_set:
                break
            candidates = tuple(sorted(candidate_set))
            scores: list[float] = []
            for cell in candidates:
                fragment = self._fragments[cell]
                transition = branch.get(cell, 0.0)
                edge = (context, cell)
                if cell == stop_cell:
                    score = transition + _STOP_TONIC
                elif (
                    fragment.kind == "input"
                    and (
                        inventory_item_index
                        if active_inventory_loop is not None
                        and cell == active_inventory_loop.item_cell
                        else fragment.index
                    )
                    >= len(symbols)
                ):
                    score = 0.0
                elif edge in self._projection_edges:
                    cue_drive = self._cue_drive(cues, context, cell)
                    score = (
                        max(0.0, cue_drive) * (1.0 + 0.05 * transition)
                        if projection_enabled
                        else 0.0
                    )
                else:
                    score = transition
                if (
                    cell != stop_cell
                    and self._inventory_loops
                    and not emitted_cells
                ):
                    score *= inventory_loop_cue_drive
                scores.append(score)
            race = self._race(candidates, scores)
            winner = race.winner
            fragment = self._fragments[winner]
            fragment_steps.append(
                FragmentRaceStep(
                    step=len(fragment_steps),
                    candidate_cells=candidates,
                    candidate_scores=tuple(scores),
                    winner_cell=winner,
                    winner_kind=fragment.kind,
                    winner_text=fragment.text,
                    spike_time_ms=race.spike_time_ms,
                    simulated_steps=race.simulated_steps,
                )
            )
            if fragment.kind == "stop":
                if branch.get(stop_cell, 0.0) > 0.0 and emitted_cells:
                    terminal_stop = True
                else:
                    selected_stop = True
                break

            selected_edge = (context, winner)
            starting_inventory_loop = (
                self._inventory_loop_by_edge.get(selected_edge)
                if inventory_loop_enabled
                else None
            )
            recurrent_inventory_loop = (
                active_inventory_loop
                if active_inventory_loop is not None
                and winner == active_inventory_loop.item_cell
                else None
            )
            inventory_loop = starting_inventory_loop or recurrent_inventory_loop
            if starting_inventory_loop is not None:
                inventory_item_index = fragment.index

            if fragment.kind in {"literal", "inventory-separator"}:
                piece = fragment.text
            elif fragment.kind == "input":
                if binding_enabled:
                    piece = symbols[
                        inventory_item_index
                        if inventory_loop is not None
                        else fragment.index
                    ]
                elif inventory_loop is not None:
                    piece = self._inventory_placeholder(inventory_item_index)
                else:
                    piece = self._input_placeholder(fragment.index)
            elif fragment.kind == "local":
                if not binding_enabled:
                    piece = self._local_placeholder(fragment.index)
                else:
                    piece = local_bindings.get(fragment.index, "")
                    if not piece:
                        piece, binding_race = self._fresh_binding(
                            fragment.index,
                            forbidden | frozenset(local_bindings.values()),
                        )
                        local_bindings[fragment.index] = piece
                        binding_races.append(binding_race)
            else:
                raise RuntimeError("non-emittable fragment won the output race")
            emitted_text.append(piece)
            emitted_cells.append(winner)
            if selected_edge in self._projection_edges:
                projected_edges.append(selected_edge)
            if inventory_loop is not None:
                history.append(winner)
            elif selected_edge in self._canonical_edges:
                history.append(self._projection_context_cell)
            else:
                history.append(winner)
            if starting_inventory_loop is not None:
                active_inventory_loop = starting_inventory_loop

        if selected_stop:
            output = "<stop>insufficient learned fragment evidence</stop>"
            output_token_ids = tuple(self._tokenizer.encode(output))
        else:
            output = "".join(emitted_text)
            output_token_ids = tuple(self._tokenizer.encode(output))
        trace = FactorizedCompletionTrace(
            query_sha256=query_sha,
            model_digest=model_digest,
            ablation=ablation,
            cue_words=cues,
            fragment_steps=tuple(fragment_steps),
            binding_races=tuple(binding_races),
            inventory_loop_races=tuple(inventory_loop_races),
            emitted_fragment_cells=tuple(emitted_cells),
            emitted_token_ids=output_token_ids,
            selected_stop=selected_stop,
            terminal_stop=terminal_stop,
            output=output,
        )
        self._last_trace = trace
        self._last_completion = _LastCompletion(
            query_sha,
            cues,
            tuple(projected_edges),
            output,
        )
        return output

    def complete(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        max_new_tokens: int = 64,
        stop_text: str | tuple[str, ...] | None = None,
        ablation: str | None = None,
    ) -> str:
        return self.raw_complete(
            prompt,
            inventory,
            cue_text=cue_text,
            max_new_tokens=max_new_tokens,
            stop_text=stop_text,
            ablation=ablation,
        )

    def _require_last(
        self,
        prompt: str,
        inventory: SymbolInventory,
        cue_text: str,
    ) -> _LastCompletion:
        query_sha = _query_sha256(prompt, inventory, cue_text)
        if self._last_completion is None or self._last_completion.query_sha256 != query_sha:
            raise ValueError("plasticity is not bound to the last neuronal completion")
        return self._last_completion

    def _apply_plasticity(
        self,
        *,
        completion: _LastCompletion,
        delta: float,
        polarity: str,
        evidence_text: str,
    ) -> FactorizedLearningUpdate | None:
        selected = tuple(sorted(set(completion.projected_edges)))
        if not selected:
            return None
        prior_digest = self.training_digest
        changed = 0
        per_cue = delta / len(completion.cue_words)
        for cue in completion.cue_words:
            for context, cell in selected:
                key = (cue, context, cell)
                self._plasticity_ticks[key] = (
                    self._plasticity_ticks.get(key, 0.0) + per_cue
                )
                changed += 1
        self._plasticity_root = sha256(
            (
                "humanbrain-factorized-pointer-plasticity-event-v1\0"
                + self._plasticity_root
                + "\0"
                + polarity
                + "\0"
                + completion.query_sha256
                + "\0"
                + sha256(evidence_text.encode("utf-8")).hexdigest()
            ).encode("ascii")
        ).hexdigest()
        self._training_digest_cache = None
        return FactorizedLearningUpdate(
            prior_training_digest=prior_digest,
            training_digest=self.training_digest,
            polarity=polarity,
            synapses_changed=changed,
        )

    def observe_failure(
        self,
        prompt: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        feedback_text: str,
    ) -> FactorizedLearningUpdate | None:
        """Depress the exact selected cue-fragment eligibility trace."""

        if not isinstance(feedback_text, str) or not feedback_text:
            raise ValueError("feedback_text must be one non-empty string")
        completion = self._require_last(prompt, inventory, cue_text)
        return self._apply_plasticity(
            completion=completion,
            delta=-1.0,
            polarity="failure",
            evidence_text=feedback_text,
        )

    def learn(
        self,
        prompt: str,
        output: str,
        inventory: SymbolInventory,
        *,
        cue_text: str,
        strength: float = 1.0,
    ) -> FactorizedLearningUpdate | None:
        """Reinforce the selected reusable fragments after one public success."""

        if not isinstance(output, str) or not output:
            raise ValueError("output must be one non-empty string")
        if not np.isfinite(strength) or strength <= 0.0:
            raise ValueError("strength must be one positive finite value")
        completion = self._require_last(prompt, inventory, cue_text)
        return self._apply_plasticity(
            completion=completion,
            delta=float(strength),
            polarity="success",
            evidence_text=output,
        )
