"""TRAIN-only sizing for the ideal shared-column R3 replacement.

The primary model is 4,096 columns backed by a 131,072-address fitted-GLIF3
pyramidal sheet, one injective fitted-GLIF3 relay twin per pyramidal address,
one fitted-GLIF3 basket cell per column, and one generic proximal/relay pair
per column for the A3 mass-matched control.  A seeded token SDR activates K
columns and a context-conditioned winner is selected in every active column.
``--check`` validates the committed artifact without importing NumPy or
decoding corpus bytes.  ``--replay-check`` rebuilds the canonical TRAIN stream
and repeats the three-pass structural allocation.

This is an ideal capacity model, not capability evidence or proof that a
physical plateau/race realizes the assumed context code.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import struct
import sys
import time
from array import array
from decimal import Decimal, localcontext
from fractions import Fraction
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = Path(__file__).with_name("relay_pool_capacity.json")

ARTIFACT_ID = "r3-relay-pool-capacity-v5-paired-sources"
SCHEMA_VERSION = 5
CORPUS_SHA256 = "d585e5644505b34ca516446f9b4fa2e95ee8ffa860d27858a7ca6450ee9ad3aa"
TOKENIZER_SHA256 = "130f42fc92e5c1f42c53d8904aeefbc4db21c189dc5b9fae98157dc06c9218a2"
PREFIX_200_SHA256 = "de5d527b12968762d6bd614c3a77c705e053e272361a72d20f3c0a21b272ef3c"
TRAIN_STREAM_SHA256 = "f5d2ef9a096e1bb184af925d0523e3a3b80c5bdf7e821b9fca9f65c66059e7b5"

TRAIN_BOOKS = 49
HELD_OUT_BOOKS = 17
TRAIN_TOKENS = 2_117_412
EXPOSURE_PASSES = 3
TOKEN_IDS = tuple(range(256)) + tuple(range(1024, 1536))
TOKEN_ID_SET = frozenset(TOKEN_IDS)
SDR_SEED = 17
COLUMNS = 4_096
CELLS_PER_COLUMN = 32
TOTAL_CELLS = COLUMNS * CELLS_PER_COLUMN
WINNER_RELAY_CELLS = TOTAL_CELLS
BASKET_CELLS = COLUMNS
GENERIC_PROXIMAL_CELLS = COLUMNS
GENERIC_RELAY_CELLS = COLUMNS
WINNER_RELAY_OFFSET = TOTAL_CELLS
BASKET_OFFSET = WINNER_RELAY_OFFSET + WINNER_RELAY_CELLS
GENERIC_PROXIMAL_OFFSET = BASKET_OFFSET + BASKET_CELLS
GENERIC_RELAY_OFFSET = GENERIC_PROXIMAL_OFFSET + GENERIC_PROXIMAL_CELLS
TOTAL_PHYSICAL_GLIF3_CELLS = GENERIC_RELAY_OFFSET + GENERIC_RELAY_CELLS
PHYSICAL_GID_BITS = 19
K_VALUES = (11, 20, 40)
CONTACTS_PER_SEGMENT = 20
PHYSICAL_CONTACTS_PER_PAIR = 2
PAIR_SLOTS_PER_SEGMENT = CONTACTS_PER_SEGMENT // PHYSICAL_CONTACTS_PER_PAIR
PYRAMIDAL_GID_BITS = 17
COLUMN_GID_BITS = 12
MAX_SEGMENTS_PER_CELL = 255
INITIAL_PERMANENCE = "0.51"
CONNECTED_PERMANENCE = "0.50"
PERMANENCE_INCREMENT = "0.10"
NEGATIVE_DECREMENT = "0.004"
PERMANENCE_TICKS_PER_UNIT = 500
PERMANENCE_BITS = PERMANENCE_TICKS_PER_UNIT.bit_length()
MUTABLE_RECOGNITION_INDEX_CAP_BYTES = 996_199_200
THETA_M = 10
THETA_A = 13
BOUNDARY_CONTEXT = -1

# Retained only as explicitly rejected/comparison geometries from the design
# review.  They are not the primary architecture.
TOKEN_POOL_BASE_CELLS = 32
TOKEN_POOL_TOTAL_BUDGET = 49_152
TOKEN_POOL_RESERVE = TOKEN_POOL_TOTAL_BUDGET - len(TOKEN_IDS) * TOKEN_POOL_BASE_CELLS


class CapacityError(RuntimeError):
    """The artifact or replay violates the frozen sizing contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CapacityError(message)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _architecture() -> dict[str, Any]:
    return {
        "basket_cells": BASKET_CELLS,
        "basket_offset": BASKET_OFFSET,
        "cells_per_column": CELLS_PER_COLUMN,
        "columns": COLUMNS,
        "contact_source_pairs": {
            "A1": "winner_pyramidal + injective_winner_relay",
            "A3": "generic_proximal + injective_generic_relay",
        },
        "contacts_per_segment": CONTACTS_PER_SEGMENT,
        "generic_proximal_cells": GENERIC_PROXIMAL_CELLS,
        "generic_proximal_offset": GENERIC_PROXIMAL_OFFSET,
        "generic_relay_cells": GENERIC_RELAY_CELLS,
        "generic_relay_offset": GENERIC_RELAY_OFFSET,
        "initial_permanence": INITIAL_PERMANENCE,
        "max_segments_per_cell": MAX_SEGMENTS_PER_CELL,
        "physical_gid_bits": PHYSICAL_GID_BITS,
        "pyramidal_cell_cap": TOTAL_CELLS,
        "sdr_seed": SDR_SEED,
        "token_sdr": "token_sdr_indices(seed=17,n_bits=4096,w_ones=K)",
        "total_physical_glif3_cells": TOTAL_PHYSICAL_GLIF3_CELLS,
        "winner_relay_cells": WINNER_RELAY_CELLS,
        "winner_relay_offset": WINNER_RELAY_OFFSET,
    }


def _permanence_encoding() -> dict[str, Any]:
    def ticks(value: str) -> int:
        scaled = Decimal(value) * PERMANENCE_TICKS_PER_UNIT
        _require(
            scaled == scaled.to_integral_value(),
            f"permanence {value} is not exact at scale {PERMANENCE_TICKS_PER_UNIT}",
        )
        return int(scaled)

    return {
        "connected_ticks": ticks(CONNECTED_PERMANENCE),
        "initial_ticks": ticks(INITIAL_PERMANENCE),
        "minimum_bits": PERMANENCE_TICKS_PER_UNIT.bit_length(),
        "native_hot_dtype": "uint16",
        "negative_decrement_ticks": ticks(NEGATIVE_DECREMENT),
        "permanence_increment_ticks": ticks(PERMANENCE_INCREMENT),
        "reachable_tick_count": PERMANENCE_TICKS_PER_UNIT + 1,
        "reachable_tick_range": [0, PERMANENCE_TICKS_PER_UNIT],
        "ticks_per_unit": PERMANENCE_TICKS_PER_UNIT,
    }


def _claim_boundary() -> dict[str, bool]:
    return {
        "capability_evidence": False,
        "ideal_structural_sizing_only": True,
        "physical_race_evidence": False,
    }


def _collision_tail(
    assembly_size: int,
    threshold: int,
    *,
    population_size: int,
) -> dict[str, Any]:
    denominator = math.comb(population_size, CONTACTS_PER_SEGMENT)
    numerator = sum(
        math.comb(assembly_size, overlap)
        * math.comb(
            population_size - assembly_size,
            CONTACTS_PER_SEGMENT - overlap,
        )
        for overlap in range(threshold, min(assembly_size, CONTACTS_PER_SEGMENT) + 1)
    )
    exact = Fraction(numerator, denominator)
    with localcontext() as context:
        context.prec = 50
        decimal = Decimal(exact.numerator) / Decimal(exact.denominator)
    return {
        "approx_probability": format(decimal, ".20E"),
        "assembly_size": assembly_size,
        "exact_denominator": str(exact.denominator),
        "exact_numerator": str(exact.numerator),
        "population_size": population_size,
        "threshold": threshold,
    }


def _grouped_relay_tail(physical_threshold: int) -> dict[str, Any]:
    """Exact tail for ten sampled winner/relay pairs vs an 11-pair context."""
    sampled_pairs = CONTACTS_PER_SEGMENT // 2
    assembly_pairs = 11
    pair_threshold = _ceil_div(physical_threshold, 2)
    denominator = math.comb(TOTAL_CELLS, sampled_pairs)
    numerator = sum(
        math.comb(assembly_pairs, overlap)
        * math.comb(TOTAL_CELLS - assembly_pairs, sampled_pairs - overlap)
        for overlap in range(pair_threshold, min(assembly_pairs, sampled_pairs) + 1)
    )
    exact = Fraction(numerator, denominator)
    with localcontext() as context:
        context.prec = 50
        decimal = Decimal(exact.numerator) / Decimal(exact.denominator)
    return {
        "approx_probability": format(decimal, ".20E"),
        "assembly_pairs": assembly_pairs,
        "exact_denominator": str(exact.denominator),
        "exact_numerator": str(exact.numerator),
        "paired_sources": True,
        "pair_threshold": pair_threshold,
        "physical_contacts_per_pair": 2,
        "physical_threshold": physical_threshold,
        "population_size": TOTAL_CELLS,
        "sampled_pairs": sampled_pairs,
    }


def validate_artifact(artifact: dict[str, Any]) -> None:
    """Validate the source-controlled evidence without corpus/simulator I/O."""
    _require(artifact.get("artifact_id") == ARTIFACT_ID, "artifact id mismatch")
    _require(artifact.get("schema_version") == SCHEMA_VERSION, "schema mismatch")
    _require(artifact.get("claim_boundary") == _claim_boundary(), "claim drift")
    _require(artifact.get("architecture") == _architecture(), "architecture drift")
    _require(
        artifact.get("permanence_encoding") == _permanence_encoding(),
        "permanence encoding drift",
    )

    scope = artifact.get("scope")
    _require(isinstance(scope, dict), "scope is missing")
    _require(scope.get("corpus_sha256") == CORPUS_SHA256, "corpus hash mismatch")
    _require(scope.get("tokenizer_sha256") == TOKENIZER_SHA256, "tokenizer drift")
    _require(
        scope.get("first_200_token_ids_sha256") == PREFIX_200_SHA256,
        "TRAIN prefix drift",
    )
    if TRAIN_STREAM_SHA256 != "DISCOVER_ON_FIRST_REPLAY":
        _require(
            scope.get("train_stream_sha256") == TRAIN_STREAM_SHA256,
            "TRAIN stream drift",
        )
    _require(scope.get("train_books_decoded") == TRAIN_BOOKS, "TRAIN split drift")
    _require(scope.get("held_out_books_skipped") == HELD_OUT_BOOKS, "VAL split drift")
    _require(scope.get("held_out_books_decoded") == 0, "HELD-OUT was decoded")
    _require(scope.get("train_tokens_per_pass") == TRAIN_TOKENS, "token drift")

    schedule = artifact.get("schedule")
    _require(isinstance(schedule, dict), "schedule is missing")
    _require(schedule.get("exposure_passes") == EXPOSURE_PASSES, "pass drift")
    _require(
        schedule.get("book_boundary_resets") == TRAIN_BOOKS * EXPOSURE_PASSES,
        "book/pass reset drift",
    )
    _require(
        schedule.get("transition_events_per_pass") == TRAIN_TOKENS - TRAIN_BOOKS,
        "transition-count drift",
    )
    passes = schedule.get("passes")
    _require(isinstance(passes, list) and len(passes) == 3, "pass rows drift")
    _require(passes[1].get("novel_contexts") == 0, "pass 2 grew mappings")
    _require(passes[2].get("novel_contexts") == 0, "pass 3 grew mappings")

    arms = artifact.get("shared_column_arms")
    _require(isinstance(arms, dict), "shared-column arms are missing")
    _require(set(arms) == {"K11", "K20", "K40"}, "K comparison drift")
    for k in K_VALUES:
        arm = arms[f"K{k}"]
        _require(arm.get("winner_cells_per_context") == k, f"K{k} width drift")
        _require(arm.get("segments") == arm.get("distinct_contexts") * k, "segment drift")
        _require(arm.get("contacts") == arm.get("segments") * 20, "contact drift")
        _require(arm.get("max_segments_per_cell") >= 0, "negative cell load")
        expected_memory = (
            _paired_memory(arm["segments"])
            if k == 11
            else _shared_memory(
                arm["segments"],
                contact_cell_gid_bits=PYRAMIDAL_GID_BITS,
            )
        )
        _require(
            arm.get("memory_estimate") == expected_memory,
            f"K{k} memory estimate drift",
        )
    _require(arms["K11"].get("physical_relay_multiplier") == 2, "K11 relay drift")
    _require(arms["K20"].get("physical_relay_multiplier") == 1, "K20 relay drift")
    _require(arms["K40"].get("physical_relay_multiplier") == 1, "K40 relay drift")
    _require(
        arms["K20"].get("fits_segment_capacity") is not None,
        "K20 fit verdict is missing",
    )

    developmental = artifact.get("developmental_variable_depth")
    _require(isinstance(developmental, dict), "variable-depth arms are missing")
    _require(set(developmental) == {"base4_K11", "base8_K11"}, "depth grid drift")
    for label, base_depth in (("base4_K11", 4), ("base8_K11", 8)):
        arm = developmental[label]
        _require(arm.get("base_depth") == base_depth, f"{label} base drift")
        _require(arm.get("winner_cells_per_context") == 11, f"{label} K drift")
        _require(arm.get("physical_relay_multiplier") == 2, f"{label} relay drift")
        _require(
            arm.get("all_train_unique_states_fit") is not None,
            f"{label} fit verdict is missing",
        )
        _require(
            arm.get("memory_estimate")
            == _variable_memory(
                arm["segments"],
                arm["pyramidal_cells_required"],
                base_cells_per_column=base_depth,
            ),
            f"{label} memory estimate drift",
        )

    comparisons = artifact.get("token_pool_comparisons")
    _require(isinstance(comparisons, dict), "token-pool comparisons are missing")
    _require(
        comparisons["q1"]["representationally_valid_for_train"] is False,
        "q1 alias rejection drift",
    )

    tails = artifact.get("collision_tails")
    _require(isinstance(tails, dict), "collision tails are missing")
    expected_sizes = {"W20": 20, "W40": 40}
    for label, size in expected_sizes.items():
        _require(
            tails[label]["theta_m_10"]
            == _collision_tail(size, THETA_M, population_size=TOTAL_CELLS),
            f"{label} theta_m tail drift",
        )
        _require(
            tails[label]["theta_a_13"]
            == _collision_tail(size, THETA_A, population_size=TOTAL_CELLS),
            f"{label} theta_a tail drift",
        )
    _require(
        tails["W22"]["theta_m_10"] == _grouped_relay_tail(THETA_M),
        "W22 grouped theta_m tail drift",
    )
    _require(
        tails["W22"]["theta_a_13"] == _grouped_relay_tail(THETA_A),
        "W22 grouped theta_a tail drift",
    )


def load_and_validate(path: Path) -> dict[str, Any]:
    artifact = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(artifact, dict), "artifact root must be an object")
    validate_artifact(artifact)
    return artifact


def _load_train_books() -> tuple[list[Any], dict[str, Any]]:
    """Encode canonical TRAIN books; the role branch precedes every decode."""
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from experiments.r3_pilot.run_c1_cohort import (  # noqa: PLC0415
        build_concept_stack,
        segment_verses,
    )

    freeze_dir = Path(
        os.environ.get("TEMP", os.environ.get("LOCALAPPDATA", ""))
    ) / "hb_bible_freeze"
    freeze = json.loads(
        (freeze_dir / "FREEZE-CANDIDATE-v3-BIBLE.json").read_text(encoding="utf-8")
    )
    spans = json.loads(
        (freeze_dir / "evidence" / "pg10_book_spans_v3.json").read_text(
            encoding="utf-8"
        )
    )
    trimmed = (freeze_dir / "evidence_pg10.trimmed.bin").read_bytes()
    corpus_sha = _sha256(trimmed)
    _require(corpus_sha == CORPUS_SHA256, "canonical corpus bytes drifted")
    _require(
        corpus_sha == freeze["corpus"]["sha256_trimmed"],
        "freeze manifest disagrees with corpus bytes",
    )

    train_books: list[tuple[str, str]] = []
    skipped_held_out = 0
    decoded_held_out = 0
    for book in spans["books"]:
        if book["role"] != "train":
            skipped_held_out += 1
            continue
        start, end = book["span_offsets"]
        train_books.append(
            (book["short_name"], trimmed[start:end].decode("utf-8"))
        )
    _require(len(train_books) == TRAIN_BOOKS, "TRAIN split must contain 49 books")
    _require(skipped_held_out == HELD_OUT_BOOKS, "HELD-OUT split must contain 17 books")
    _require(decoded_held_out == 0, "held-out decode counter must stay zero")

    stack, stack_info = build_concept_stack([text for _, text in train_books])
    tokenizer_payload = {
        "concepts_hex": [payload.hex() for payload in stack.concepts],
        "tag": stack_info["tokenizer"],
    }
    tokenizer_sha = _sha256(_canonical_json_bytes(tokenizer_payload))
    _require(tokenizer_sha == TOKENIZER_SHA256, "canonical tokenizer drifted")

    encoded_books: list[Any] = []
    book_lengths: list[tuple[str, int]] = []
    first_ids: list[int] = []
    observed_ids: set[int] = set()
    full_hasher = hashlib.sha256()
    for name, text in train_books:
        ids: list[int] = []
        for verse in segment_verses(text):
            ids.extend(int(token) for token in stack.encode(verse))
        array = np.asarray(ids, dtype="<i4")
        _require(array.size > 0, f"TRAIN book {name} encoded empty")
        encoded_books.append(array)
        book_lengths.append((name, int(array.size)))
        full_hasher.update(array.tobytes())
        observed_ids.update(int(token) for token in np.unique(array))
        if len(first_ids) < 200:
            first_ids.extend(ids[: 200 - len(first_ids)])

    total_tokens = sum(length for _, length in book_lengths)
    _require(total_tokens == TRAIN_TOKENS, "canonical TRAIN token count drifted")
    _require(observed_ids <= TOKEN_ID_SET, "TRAIN stream left the sparse vocabulary")
    first_200_sha = _sha256(",".join(map(str, first_ids)).encode("utf-8"))
    _require(first_200_sha == PREFIX_200_SHA256, "canonical TRAIN prefix drifted")
    stream_sha = full_hasher.hexdigest()
    if TRAIN_STREAM_SHA256 != "DISCOVER_ON_FIRST_REPLAY":
        _require(stream_sha == TRAIN_STREAM_SHA256, "canonical TRAIN stream drifted")

    return encoded_books, {
        "book_token_lengths_sha256": _sha256(_canonical_json_bytes(book_lengths)),
        "corpus_sha256": corpus_sha,
        "first_200_token_ids_sha256": first_200_sha,
        "held_out_books_decoded": decoded_held_out,
        "held_out_books_skipped": skipped_held_out,
        "observed_token_ids": len(observed_ids),
        "tokenizer_sha256": tokenizer_sha,
        "train_books_decoded": len(train_books),
        "train_stream_sha256": stream_sha,
        "train_tokens_per_pass": total_tokens,
    }


def _context_demand(
    encoded_books: Sequence[Any],
) -> tuple[list[int], array, array, list[dict[str, Any]], str]:
    """Replay an injective abstract context code over all exposure passes."""
    mappings: list[dict[int, int]] = [dict() for _ in range(1536)]
    context_counts = [0] * 1536
    allocation_tokens = array("H")
    allocation_ordinals = array("I")
    next_state = 1
    mapping_hasher = hashlib.sha256()
    path_hasher = hashlib.sha256()
    transitions = 0
    reused = 0
    for book in encoded_books:
        previous_state: int | None = None
        for raw_token in book:
            token = int(raw_token)
            if previous_state is None:
                previous_state = -(token + 1)
                path_hasher.update(struct.pack("<iq", token, previous_state))
                continue
            transitions += 1
            winner_state = mappings[token].get(previous_state)
            if winner_state is None:
                ordinal = context_counts[token]
                context_counts[token] += 1
                winner_state = next_state
                next_state += 1
                mappings[token][previous_state] = winner_state
                allocation_tokens.append(token)
                allocation_ordinals.append(ordinal)
                mapping_hasher.update(
                    struct.pack("<iqq", token, previous_state, winner_state)
                )
            else:
                reused += 1
            path_hasher.update(struct.pack("<iq", token, winner_state))
            previous_state = winner_state

    novel = len(allocation_tokens)
    path_sha = path_hasher.hexdigest()
    first_pass = {
        "boundary_resets": len(encoded_books),
        "novel_contexts": novel,
        "pass_index_one_based": 1,
        "reused_contexts": reused,
        "transition_events": transitions,
        "winner_path_sha256": path_sha,
    }
    pass_rows = [first_pass]
    for pass_index in (2, 3):
        replay_hasher = hashlib.sha256()
        replay_transitions = 0
        replay_reused = 0
        for book in encoded_books:
            previous_state = None
            for raw_token in book:
                token = int(raw_token)
                if previous_state is None:
                    previous_state = -(token + 1)
                    replay_hasher.update(struct.pack("<iq", token, previous_state))
                    continue
                replay_transitions += 1
                winner_state = mappings[token].get(previous_state)
                _require(winner_state is not None, f"pass {pass_index} found novel context")
                replay_reused += 1
                replay_hasher.update(struct.pack("<iq", token, winner_state))
                previous_state = winner_state
        replay_sha = replay_hasher.hexdigest()
        _require(replay_sha == path_sha, f"pass {pass_index} winner path drifted")
        pass_rows.append(
            {
                "boundary_resets": len(encoded_books),
                "novel_contexts": 0,
                "pass_index_one_based": pass_index,
                "reused_contexts": replay_reused,
                "transition_events": replay_transitions,
                "winner_path_sha256": replay_sha,
            }
        )
    return (
        context_counts,
        allocation_tokens,
        allocation_ordinals,
        pass_rows,
        mapping_hasher.hexdigest(),
    )


def _token_sdrs(k: int) -> tuple[dict[int, tuple[int, ...]], str]:
    import numpy as np

    sys.path.insert(0, str(REPO_ROOT / "src"))
    from humanbrain.sdr_union import token_sdr_indices  # noqa: PLC0415

    output: dict[int, tuple[int, ...]] = {}
    hasher = hashlib.sha256()
    seen: set[tuple[int, ...]] = set()
    for token in TOKEN_IDS:
        columns = tuple(
            int(value)
            for value in token_sdr_indices(
                token,
                n_bits=COLUMNS,
                w_ones=k,
                seed=SDR_SEED,
            )
        )
        _require(len(columns) == k and len(set(columns)) == k, "invalid token SDR")
        _require(columns not in seen, f"K{k} token SDR collision")
        seen.add(columns)
        output[token] = columns
        hasher.update(struct.pack("<I", token))
        hasher.update(np.asarray(columns, dtype="<u2").tobytes())
    return output, hasher.hexdigest()


def _winner_cells(columns: Sequence[int], ordinal: int) -> tuple[int, ...]:
    """Injective base-32 code mixed across columns for balanced tie-breaking."""
    remaining = ordinal
    prefix = 0
    cells: list[int] = []
    for position, column in enumerate(columns):
        digit = remaining & (CELLS_PER_COLUMN - 1)
        remaining >>= 5
        prefix = (prefix + digit) & (CELLS_PER_COLUMN - 1)
        offset = (prefix + position) & (CELLS_PER_COLUMN - 1)
        cells.append(column * CELLS_PER_COLUMN + offset)
    _require(remaining == 0, "context code exhausted 32^K combinations")
    return tuple(cells)


def _mib_string(byte_count: int) -> str:
    with localcontext() as context:
        context.prec = 30
        value = Decimal(byte_count) / Decimal(1 << 20)
    return format(value, ".6f")


def _shared_memory(
    segments: int,
    *,
    contact_cell_gid_bits: int,
) -> dict[str, Any]:
    contacts = segments * CONTACTS_PER_SEGMENT
    contact_bits = contacts * (contact_cell_gid_bits + PERMANENCE_BITS)
    cell_offset_bits = (TOTAL_CELLS + 1) * 32
    total_bytes = _ceil_div(contact_bits + cell_offset_bits, 8)
    return {
        "assumptions": {
            "cell_segment_index": "CSR u32 offsets",
            "contact_cell_gid_bits": contact_cell_gid_bits,
            "permanence_bits": PERMANENCE_BITS,
            "recognition_inverted_index": "excluded; separately costed",
            "token_sdr_storage": "procedural seed; zero resident bytes",
        },
        "cell_offset_bits": cell_offset_bits,
        "contact_bits": contact_bits,
        "ideal_bitpacked_bytes": total_bytes,
        "ideal_bitpacked_mib": _mib_string(total_bytes),
        "recognition_inverted_index_included": False,
    }


def _compact_paired_arm(pair_slots: int, *, source_bits: int) -> dict[str, Any]:
    segment_count = pair_slots // PAIR_SLOTS_PER_SEGMENT
    segment_owner_bits = segment_count * PYRAMIDAL_GID_BITS
    segment_owner_bytes = _ceil_div(segment_owner_bits, 8)
    pair_source_bits = pair_slots * source_bits
    pair_permanence_bits = pair_slots * PERMANENCE_BITS
    pair_record_bits = pair_source_bits + pair_permanence_bits
    pair_record_bytes = _ceil_div(pair_record_bits, 8)
    total_bits = segment_owner_bits + pair_record_bits
    total_bytes = segment_owner_bytes + pair_record_bytes
    return {
        "mib": _mib_string(total_bytes),
        "pair_permanence_bits": pair_permanence_bits,
        "pair_record_bits": pair_record_bits,
        "pair_record_bytes": pair_record_bytes,
        "pair_source_bits": pair_source_bits,
        "segment_owner_bits": segment_owner_bits,
        "segment_owner_bytes": segment_owner_bytes,
        "source_bits_per_pair": source_bits,
        "total_bits": total_bits,
        "total_bytes": total_bytes,
    }


def _paired_segment_store(segments: int) -> dict[str, Any]:
    pair_slots = segments * PAIR_SLOTS_PER_SEGMENT
    segment_owner_bytes = segments * 4
    pair_source_bytes = pair_slots * 4
    pair_permanence_bytes = pair_slots * 2
    owner_segment_count_bytes = TOTAL_CELLS
    native_hot_bytes = (
        segment_owner_bytes
        + pair_source_bytes
        + pair_permanence_bytes
        + owner_segment_count_bytes
    )
    static_offset_bytes = (TOTAL_CELLS + 1) * 4
    static_posting_bytes = pair_slots * 4
    return {
        "compact_bitpacked": {
            "A1": _compact_paired_arm(pair_slots, source_bits=PYRAMIDAL_GID_BITS),
            "A3": _compact_paired_arm(pair_slots, source_bits=COLUMN_GID_BITS),
        },
        "native_hot_soa": {
            "mib": _mib_string(native_hot_bytes),
            "owner_segment_count_bytes": owner_segment_count_bytes,
            "owner_segment_count_dtype": "uint8",
            "pair_permanence_bytes": pair_permanence_bytes,
            "pair_permanence_dtype": "uint16",
            "pair_source_bytes": pair_source_bytes,
            "pair_source_dtype": "uint32",
            "segment_owner_bytes": segment_owner_bytes,
            "segment_owner_dtype": "uint32",
            "total_bytes": native_hot_bytes,
        },
        "pair_slot_count": pair_slots,
        "pair_slots_per_segment": PAIR_SLOTS_PER_SEGMENT,
        "permanence_bits": PERMANENCE_BITS,
        "physical_contacts_per_pair": PHYSICAL_CONTACTS_PER_PAIR,
        "relay_projection_bits": 0,
        "segment_contact_offset_bytes": 0,
        "segment_count": segments,
        "slot_rule": (
            "fixed source/relay pairs; relay identity is arithmetic and both contacts "
            "share one permanence tick; tick zero marks an empty slot"
        ),
        "recognition_index_cost": {
            "excluded_mutable_64_entry_block_ceiling_estimate_bytes": (
                MUTABLE_RECOGNITION_INDEX_CAP_BYTES
            ),
            "included_in_store_totals": False,
            "required_static_csr_floor_bytes": (
                static_offset_bytes + static_posting_bytes
            ),
            "static_offset_bytes": static_offset_bytes,
            "static_posting_bytes": static_posting_bytes,
        },
    }


def _paired_memory(
    segments: int,
    *,
    allocated_cells: int | None = None,
    base_cells_per_column: int | None = None,
) -> dict[str, Any]:
    store = _paired_segment_store(segments)
    membership = {
        "base_memberships_arithmetic": 0,
        "compact_recruited_column_bits": 0,
        "compact_recruited_column_bytes": 0,
        "compact_recruited_column_dtype_bits": COLUMN_GID_BITS,
        "hot_cell_gid_bytes": 0,
        "hot_column_offset_bytes": 0,
        "native_hot_soa_bytes": 0,
        "recruited_memberships": 0,
        "restore_rule": (
            "base memberships are arithmetic; recruited physical ids are contiguous "
            "and store one packed column id each; rebuild hot CSR on restore"
        ),
    }
    if allocated_cells is not None:
        _require(base_cells_per_column is not None, "base depth missing")
        base_memberships = COLUMNS * base_cells_per_column
        recruited_memberships = allocated_cells - base_memberships
        _require(recruited_memberships >= 0, "negative recruited membership count")
        compact_recruited_column_bits = recruited_memberships * COLUMN_GID_BITS
        hot_column_offset_bytes = (COLUMNS + 1) * 4
        hot_cell_gid_bytes = allocated_cells * 4
        membership = {
            "base_memberships_arithmetic": base_memberships,
            "compact_recruited_column_bits": compact_recruited_column_bits,
            "compact_recruited_column_bytes": _ceil_div(
                compact_recruited_column_bits,
                8,
            ),
            "compact_recruited_column_dtype_bits": COLUMN_GID_BITS,
            "hot_cell_gid_bytes": hot_cell_gid_bytes,
            "hot_column_offset_bytes": hot_column_offset_bytes,
            "native_hot_soa_bytes": hot_column_offset_bytes + hot_cell_gid_bytes,
            "recruited_memberships": recruited_memberships,
            "restore_rule": (
                "base memberships are arithmetic; recruited physical ids are contiguous "
                "and store one packed column id each; rebuild hot CSR on restore"
            ),
        }
    compact = store["compact_bitpacked"]
    hot = store["native_hot_soa"]
    recognition = store.pop("recognition_index_cost")
    return {
        "developmental_column_membership": membership,
        "recognition_inverted_index": recognition,
        "segment_store": store,
        "total_excluding_recognition_index": {
            "A1_compact_bitpacked_bytes": (
                compact["A1"]["total_bytes"]
                + membership["compact_recruited_column_bytes"]
            ),
            "A3_compact_bitpacked_bytes": (
                compact["A3"]["total_bytes"]
                + membership["compact_recruited_column_bytes"]
            ),
            "native_hot_soa_bytes": (
                hot["total_bytes"] + membership["native_hot_soa_bytes"]
            ),
        },
    }


def _shared_arm(
    k: int,
    allocation_tokens: Sequence[int],
    allocation_ordinals: Sequence[int],
    context_counts: Sequence[int],
) -> dict[str, Any]:
    import numpy as np

    sdrs, sdr_sha = _token_sdrs(k)
    loads = np.zeros((COLUMNS, CELLS_PER_COLUMN), dtype=np.uint32)
    for token in TOKEN_IDS:
        count = context_counts[token]
        if count == 0:
            continue
        ordinals = np.arange(count, dtype=np.uint64)
        prefix = np.zeros(count, dtype=np.uint64)
        for position, column in enumerate(sdrs[token]):
            digit = (ordinals >> (5 * position)) & (CELLS_PER_COLUMN - 1)
            prefix = (prefix + digit) & (CELLS_PER_COLUMN - 1)
            offsets = (prefix + position) & (CELLS_PER_COLUMN - 1)
            loads[column] += np.bincount(
                offsets.astype(np.int64), minlength=CELLS_PER_COLUMN
            ).astype(np.uint32)

    first_stop: dict[str, int] | None = None
    chronological = np.zeros((COLUMNS, CELLS_PER_COLUMN), dtype=np.uint16)
    for allocation_index, (token, ordinal) in enumerate(
        zip(allocation_tokens, allocation_ordinals, strict=True)
    ):
        cells = _winner_cells(sdrs[token], ordinal)
        for cell in cells:
            column, offset = divmod(cell, CELLS_PER_COLUMN)
            chronological[column, offset] += 1
            if chronological[column, offset] > MAX_SEGMENTS_PER_CELL:
                first_stop = {
                    "allocation_index_zero_based": allocation_index,
                    "cell_id": int(cell),
                    "load": int(chronological[column, offset]),
                    "token_id": int(token),
                    "token_context_ordinal": int(ordinal),
                }
                break
        if first_stop is not None:
            break
    distinct_contexts = len(allocation_tokens)
    segments = distinct_contexts * k
    max_load = int(loads.max())
    relay_multiplier = 2 if k == 11 else 1
    physical_width = k * relay_multiplier
    return {
        "allocation_rule": (
            "base32 context ordinal; invertible prefix mix plus column position modulo 32"
        ),
        "capacity_stop": first_stop,
        "contacts": segments * CONTACTS_PER_SEGMENT,
        "direct_20_contact_source": k >= CONTACTS_PER_SEGMENT,
        "distinct_contexts": distinct_contexts,
        "first_pass_segments": segments,
        "fits_segment_capacity": first_stop is None,
        "max_segments_per_cell": max_load,
        "memory_estimate": (
            _paired_memory(segments)
            if relay_multiplier == 2
            else _shared_memory(
                segments,
                contact_cell_gid_bits=PYRAMIDAL_GID_BITS,
            )
        ),
        "physical_relay_multiplier": relay_multiplier,
        "relay_population_cells": WINNER_RELAY_CELLS if relay_multiplier == 2 else 0,
        "relay_population_cells_sized": True,
        "physical_winner_width": physical_width,
        "cell_load_sha256": _sha256(loads.astype("<u4", copy=False).tobytes()),
        "sdr_table_sha256": sdr_sha,
        "segments": segments,
        "three_pass_segments_created": segments,
        "winner_cells_per_context": k,
    }


def _variable_memory(
    segments: int,
    allocated_cells: int,
    *,
    base_cells_per_column: int,
) -> dict[str, Any]:
    return _paired_memory(
        segments,
        allocated_cells=allocated_cells,
        base_cells_per_column=base_cells_per_column,
    )


def _variable_depth_arm(
    context_counts: Sequence[int],
    *,
    base_depth: int,
) -> dict[str, Any]:
    """Exact column demand under an ideal balanced developmental allocator."""
    import numpy as np

    sdrs, sdr_sha = _token_sdrs(11)
    column_demand = np.zeros(COLUMNS, dtype=np.uint64)
    for token in TOKEN_IDS:
        count = context_counts[token]
        if count:
            column_demand[np.asarray(sdrs[token], dtype=np.int64)] += count
    required_depth = np.maximum(
        base_depth,
        (column_demand + MAX_SEGMENTS_PER_CELL - 1) // MAX_SEGMENTS_PER_CELL,
    ).astype(np.uint32)
    allocated_cells = int(required_depth.sum())
    recruited = allocated_cells - COLUMNS * base_depth
    max_depth = int(required_depth.max())
    depth_u64 = required_depth.astype(np.uint64)
    max_load = int(np.max((column_demand + depth_u64 - 1) // depth_u64))
    code_capacity_failures = 0
    for token in TOKEN_IDS:
        capacity = 1
        for column in sdrs[token]:
            capacity *= int(required_depth[column])
            if capacity >= context_counts[token]:
                break
        if capacity < context_counts[token]:
            code_capacity_failures += 1
    first_stop = None
    if allocated_cells > TOTAL_CELLS:
        first_stop = {
            "global_cells_required": allocated_cells,
            "global_cell_cap": TOTAL_CELLS,
        }
    distinct_contexts = sum(context_counts)
    segments_allocated = distinct_contexts * 11
    return {
        "all_train_unique_states_fit": (
            first_stop is None and code_capacity_failures == 0
        ),
        "allocation_rule": (
            "ideal least-loaded local cell with seed17 context permutation; "
            "mixed-radix K-tuple uniqueness asserted separately"
        ),
        "base_depth": base_depth,
        "capacity_stop": first_stop,
        "code_capacity_failures": code_capacity_failures,
        "column_demand_sha256": _sha256(
            column_demand.astype("<u8", copy=False).tobytes()
        ),
        "contacts": segments_allocated * CONTACTS_PER_SEGMENT,
        "first_pass_segments": segments_allocated,
        "global_cell_cap": TOTAL_CELLS,
        "max_column_depth": max_depth,
        "max_segments_per_cell": max_load,
        "physical_relay_multiplier": 2,
        "relay_population_cells": WINNER_RELAY_CELLS,
        "relay_population_cells_sized": True,
        "relay_population_rule": (
            "one fitted-GLIF3 twin per pyramidal address; "
            "relay_id = pyramidal_cap + winner_id"
        ),
        "physical_winner_width": 22,
        "pyramidal_cells_required": allocated_cells,
        "pyramidal_cells_remaining": TOTAL_CELLS - allocated_cells,
        "pyramidal_cells_recruited": recruited,
        "required_depth_sha256": _sha256(
            required_depth.astype("<u4", copy=False).tobytes()
        ),
        "sdr_table_sha256": sdr_sha,
        "segments": segments_allocated,
        "three_pass_segments_created": segments_allocated,
        "winner_cells_per_context": 11,
        "memory_estimate": _variable_memory(
            segments_allocated,
            allocated_cells,
            base_cells_per_column=base_depth,
        ),
    }


def _token_pool_capacity(pool_cells: int, q: int) -> int:
    return min(
        math.comb(pool_cells, q),
        pool_cells * MAX_SEGMENTS_PER_CELL // q,
    )


def _minimum_token_pool_cells(contexts: int, q: int) -> int:
    cells = TOKEN_POOL_BASE_CELLS
    while _token_pool_capacity(cells, q) < contexts:
        cells += 1
    return cells


def _token_pool_comparisons(context_counts: Sequence[int]) -> dict[str, Any]:
    total_contexts = sum(context_counts)
    q1_exhausted = sum(count > TOKEN_POOL_BASE_CELLS for count in context_counts)
    q1_max_load = max(_ceil_div(count, TOKEN_POOL_BASE_CELLS) for count in context_counts)

    q4_cells = {
        token: _minimum_token_pool_cells(context_counts[token], 4)
        for token in TOKEN_IDS
    }
    q4_total_cells = sum(q4_cells.values())
    q4_max_load = max(
        _ceil_div(4 * context_counts[token], q4_cells[token]) for token in TOKEN_IDS
    )
    return {
        "q1": {
            "base_cells_per_token": TOKEN_POOL_BASE_CELLS,
            "contexts_exceeding_32_codes": q1_exhausted,
            "distinct_contexts": total_contexts,
            "first_pass_segments": total_contexts,
            "max_segments_per_cell_if_aliased": q1_max_load,
            "representational_capacity_per_token": 32,
            "representationally_valid_for_train": False,
            "rejection_reason": (
                "segments sharing one winner cell emit the same fixed K40 relay projection"
            ),
            "three_pass_segments_created": total_contexts,
        },
        "q4": {
            "base_cells_per_token": TOKEN_POOL_BASE_CELLS,
            "base_context_capacity": _token_pool_capacity(TOKEN_POOL_BASE_CELLS, 4),
            "capacity_formula": "min(C(m,4),floor(m*255/4))",
            "capacity_stop": q4_total_cells > TOKEN_POOL_TOTAL_BUDGET or q4_max_load > 255,
            "contacts": total_contexts * 4 * CONTACTS_PER_SEGMENT,
            "distinct_contexts": total_contexts,
            "first_pass_segments": total_contexts * 4,
            "max_segments_per_cell": q4_max_load,
            "pool_cells_required": q4_total_cells,
            "reserve_cells_available": TOKEN_POOL_RESERVE,
            "reserve_cells_used": q4_total_cells - len(TOKEN_IDS) * TOKEN_POOL_BASE_CELLS,
            "segments": total_contexts * 4,
            "three_pass_segments_created": total_contexts * 4,
            "winner_cells_per_context": 4,
            "winner_projection_targets_per_cell": 10,
        },
    }


def run_replay() -> dict[str, Any]:
    encoded_books, scope = _load_train_books()
    (
        context_counts,
        allocation_tokens,
        allocation_ordinals,
        pass_rows,
        mapping_sha,
    ) = _context_demand(encoded_books)
    distinct_contexts = len(allocation_tokens)
    max_contexts = max(context_counts)
    max_context_tokens = [
        token for token, count in enumerate(context_counts) if count == max_contexts
    ]
    arms = {
        f"K{k}": _shared_arm(
            k,
            allocation_tokens,
            allocation_ordinals,
            context_counts,
        )
        for k in K_VALUES
    }
    developmental = {
        f"base{base_depth}_K11": _variable_depth_arm(
            context_counts,
            base_depth=base_depth,
        )
        for base_depth in (4, 8)
    }

    return {
        "architecture": _architecture(),
        "artifact_id": ARTIFACT_ID,
        "claim_boundary": _claim_boundary(),
        "collision_tails": {
            "W20": {
                "theta_a_13": _collision_tail(
                    20,
                    THETA_A,
                    population_size=TOTAL_CELLS,
                ),
                "theta_m_10": _collision_tail(
                    20,
                    THETA_M,
                    population_size=TOTAL_CELLS,
                ),
            },
            "W22": {
                "theta_a_13": _grouped_relay_tail(THETA_A),
                "theta_m_10": _grouped_relay_tail(THETA_M),
            },
            "W40": {
                "theta_a_13": _collision_tail(
                    40,
                    THETA_A,
                    population_size=TOTAL_CELLS,
                ),
                "theta_m_10": _collision_tail(
                    40,
                    THETA_M,
                    population_size=TOTAL_CELLS,
                ),
            },
        },
        "context_demand": {
            "distinct_contexts": distinct_contexts,
            "mapping_sha256": mapping_sha,
            "max_contexts_per_token": max_contexts,
            "max_context_token_ids": max_context_tokens,
        },
        "developmental_variable_depth": developmental,
        "permanence_encoding": _permanence_encoding(),
        "schedule": {
            "book_boundary_resets": TRAIN_BOOKS * EXPOSURE_PASSES,
            "boundary_rule": (
                "clear prior state; first token emits a token-local code and grows no segment"
            ),
            "exposure_passes": EXPOSURE_PASSES,
            "final_evaluation_included": False,
            "mapping_persistence": "across books and passes",
            "pass_replay_method": "literal_three_pass",
            "passes": pass_rows,
            "transition_events_per_pass": TRAIN_TOKENS - TRAIN_BOOKS,
        },
        "schema_version": SCHEMA_VERSION,
        "scope": scope,
        "shared_column_arms": arms,
        "token_pool_comparisons": _token_pool_comparisons(context_counts),
    }


def _write_artifact(path: Path, artifact: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--check", action="store_true", help="fast artifact check")
    actions.add_argument(
        "--replay-check",
        action="store_true",
        help="rebuild canonical TRAIN and compare the committed artifact",
    )
    actions.add_argument(
        "--write",
        action="store_true",
        help="rebuild canonical TRAIN and write the artifact",
    )
    parser.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.check:
            load_and_validate(args.artifact)
            print("relay-pool capacity artifact check: PASS")
            return 0
        started = time.perf_counter()
        observed = run_replay()
        validate_artifact(observed)
        elapsed = time.perf_counter() - started
        if args.write:
            _write_artifact(args.artifact, observed)
            print(f"relay-pool capacity artifact written: {args.artifact} ({elapsed:.1f}s)")
            return 0
        expected = load_and_validate(args.artifact)
        _require(
            _canonical_json_bytes(observed) == _canonical_json_bytes(expected),
            "TRAIN replay differs from committed artifact",
        )
        print(f"relay-pool capacity replay check: PASS ({elapsed:.1f}s)")
        return 0
    except (CapacityError, OSError, json.JSONDecodeError) as error:
        print(f"relay-pool capacity calibration: FAIL: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
