"""Target-independent scoring for Path-2 successor identity.

The scorer observes native temporal steps after they have happened.  It never
receives a target token or an expected allocation count.  Identity is bound by
the complete ``(source winner tuple, observed token)`` key and by the owner of
each already-seen ``(source winner tuple, output column)`` factor.
"""

from __future__ import annotations

import hashlib
import struct
from collections import Counter
from collections.abc import Sequence


ACTIVE_COLUMNS = 11
_WINNER_TRACE_ROW = struct.Struct("<11I")


class FactorizedSuccessorStop(RuntimeError):
    """A target-independent successor-identity contract violation."""

    def __init__(self, kind: str, step_index: int, details: dict[str, object]) -> None:
        super().__init__(kind)
        self.kind = kind
        self.step_index = step_index
        self.details = details


def _counter(step: object, name: str) -> int:
    value = getattr(step, name)
    if type(value) is not int or value < 0:
        raise ValueError(name)
    return value


def _columns(values: Sequence[int]) -> tuple[int, ...]:
    columns = tuple(values)
    if (
        len(columns) != ACTIVE_COLUMNS
        or len(set(columns)) != ACTIVE_COLUMNS
        or any(type(value) is not int or value < 0 for value in columns)
    ):
        raise ValueError("columns")
    return columns


def _winners(step: object) -> tuple[int, ...]:
    try:
        winners = tuple(int(spike.cell_id) for spike in step.winner_spikes)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("winner_spikes") from error
    if (
        len(winners) != ACTIVE_COLUMNS
        or len(set(winners)) != ACTIVE_COLUMNS
        or any(not 0 <= winner <= 0xFFFF_FFFF for winner in winners)
    ):
        raise ValueError("winner_spikes")
    return winners


class FactorizedSuccessorScorer:
    """Score complete-tuple stability with per-column owner factorization."""

    def __init__(self) -> None:
        self._complete_keys: dict[
            tuple[tuple[int, ...], int], tuple[int, ...]
        ] = {}
        self._source_tokens: dict[tuple[int, ...], list[int]] = {}
        self._token_columns: dict[int, tuple[int, ...]] = {}
        self._owner_columns: dict[int, int] = {}
        self._tuple_owners: dict[tuple[int, ...], tuple[str, int]] = {}
        self._column_binding_count = 0
        self._previous: tuple[int, ...] | None = None
        self._event_count = 0
        self._observed_steps = 0
        self._novel_keys = 0
        self._repeated_keys = 0
        self._partial_novel_allocations = 0
        self._actual_allocations = 0
        self._allocation_histogram: Counter[int] = Counter()
        self._matching_active_histogram: Counter[tuple[int, int]] = Counter()
        self._matching_total = 0
        self._active_total = 0
        self._matching_max = 0
        self._active_max = 0
        self._matching_nonzero_steps = 0
        self._active_nonzero_steps = 0
        self._matching_exceeds_active_steps = 0
        self._matching_with_zero_active_steps = 0
        self._predicted_columns_total = 0
        self._punished_segments_total = 0
        self._trace_digest = hashlib.sha256()

    def observe(
        self,
        step: object,
        *,
        columns: Sequence[int],
        reset_before: bool,
    ) -> tuple[int, ...]:
        """Observe one committed native step and mutate only after it passes."""

        location = self._observed_steps
        try:
            normalized_columns = _columns(columns)
            token_id = _counter(step, "token_id")
            event_count = _counter(step, "event_count")
            candidate = _counter(step, "candidate_segments")
            matching = _counter(step, "matching_segments")
            active = _counter(step, "active_segments")
            postings = _counter(step, "postings_visited")
            allocated = _counter(step, "allocated_segments")
            predicted_columns = _counter(step, "predicted_columns")
            punished = _counter(step, "punished_segments")
            winners = _winners(step)
        except (AttributeError, ValueError) as error:
            raise FactorizedSuccessorStop(
                "STEP_SHAPE_CONTRACT", location, {"field": str(error)}
            ) from error

        expected_event_count = 1 if reset_before else self._event_count + 1
        diagnostics = {
            "active_segments": active,
            "matching_segments": matching,
            "candidate_segments": candidate,
            "postings_visited": postings,
            "allocated_segments": allocated,
            "predicted_columns": predicted_columns,
            "punished_segments": punished,
        }
        if event_count != expected_event_count:
            raise FactorizedSuccessorStop(
                "EVENT_COUNT_DRIFT",
                location,
                {"expected": expected_event_count, "observed": event_count},
            )
        if not (
            active <= matching <= candidate <= postings
            and allocated <= ACTIVE_COLUMNS
            and predicted_columns <= ACTIVE_COLUMNS
        ):
            raise FactorizedSuccessorStop(
                "NATIVE_COUNTER_CONTRACT_DRIFT", location, diagnostics
            )
        prior_columns = self._token_columns.get(token_id)
        if prior_columns is not None and prior_columns != normalized_columns:
            raise FactorizedSuccessorStop(
                "TOKEN_COLUMN_CODE_DRIFT",
                location,
                {
                    "token_id": token_id,
                    "expected": list(prior_columns),
                    "observed": list(normalized_columns),
                },
            )
        for column, winner in zip(normalized_columns, winners, strict=True):
            prior_column = self._owner_columns.get(winner)
            if prior_column is not None and prior_column != column:
                raise FactorizedSuccessorStop(
                    "OWNER_COLUMN_ALIAS",
                    location,
                    {
                        "owner": winner,
                        "expected_column": prior_column,
                        "observed_column": column,
                    },
                )

        if reset_before:
            if allocated or punished:
                raise FactorizedSuccessorStop(
                    "RESET_MUTATED_LEARNED_STATE", location, diagnostics
                )
            seed_identity = ("reset_seed", token_id)
            prior = self._tuple_owners.get(winners)
            if prior is not None and prior != seed_identity:
                raise FactorizedSuccessorStop(
                    "RESET_REUSED_LEARNED_COMPLETE_TUPLE",
                    location,
                    {"prior_kind": prior[0], "prior_token": prior[1]},
                )
            self._tuple_owners[winners] = seed_identity
        else:
            if self._previous is None:
                raise FactorizedSuccessorStop(
                    "TRANSITION_WITHOUT_SOURCE", location, {}
                )
            source = self._previous
            key = (source, token_id)
            stored = self._complete_keys.get(key)
            if stored is not None:
                if allocated != 0 or winners != stored:
                    raise FactorizedSuccessorStop(
                        "REPEATED_COMPLETE_KEY_DRIFT",
                        location,
                        {
                            "allocated": allocated,
                            "tuple_equal": winners == stored,
                        },
                    )
                self._repeated_keys += 1
            else:
                prior = self._tuple_owners.get(winners)
                if prior is not None:
                    raise FactorizedSuccessorStop(
                        "NOVEL_KEY_REUSED_COMPLETE_TUPLE",
                        location,
                        {"prior_kind": prior[0], "prior_token": prior[1]},
                    )
                bound_owners: dict[int, int] = {}
                for prior_token in self._source_tokens.get(source, ()):
                    prior_winners = self._complete_keys[(source, prior_token)]
                    for column, owner in zip(
                        self._token_columns[prior_token],
                        prior_winners,
                        strict=True,
                    ):
                        previous_owner = bound_owners.setdefault(column, owner)
                        if previous_owner != owner:
                            raise FactorizedSuccessorStop(
                                "STORED_COLUMN_OWNER_CONTRADICTION",
                                location,
                                {
                                    "column": column,
                                    "first_owner": previous_owner,
                                    "later_owner": owner,
                                },
                            )
                bound_columns = 0
                for column, winner in zip(normalized_columns, winners, strict=True):
                    owner = bound_owners.get(column)
                    if owner is None:
                        continue
                    bound_columns += 1
                    if winner != owner:
                        raise FactorizedSuccessorStop(
                            "EXACT_SOURCE_COLUMN_OWNER_DRIFT",
                            location,
                            {
                                "column": column,
                                "expected": owner,
                                "observed": winner,
                            },
                        )
                unbound_columns = ACTIVE_COLUMNS - bound_columns
                if allocated > unbound_columns:
                    raise FactorizedSuccessorStop(
                        "ALLOCATION_EXCEEDS_UNBOUND_COLUMNS",
                        location,
                        {
                            "allocated": allocated,
                            "known_bindings": bound_columns,
                            "unbound_columns": unbound_columns,
                        },
                    )
                self._complete_keys[key] = winners
                self._tuple_owners[winners] = ("complete_key", token_id)
                self._source_tokens.setdefault(source, []).append(token_id)
                self._column_binding_count += unbound_columns
                self._novel_keys += 1
                self._partial_novel_allocations += int(allocated < ACTIVE_COLUMNS)

        self._token_columns.setdefault(token_id, normalized_columns)
        for column, winner in zip(normalized_columns, winners, strict=True):
            self._owner_columns.setdefault(winner, column)
        self._previous = winners
        self._event_count = event_count
        self._observed_steps += 1
        self._actual_allocations += allocated
        self._allocation_histogram[allocated] += 1
        self._matching_active_histogram[(matching, active)] += 1
        self._matching_total += matching
        self._active_total += active
        self._matching_max = max(self._matching_max, matching)
        self._active_max = max(self._active_max, active)
        self._matching_nonzero_steps += int(matching > 0)
        self._active_nonzero_steps += int(active > 0)
        self._matching_exceeds_active_steps += int(matching > active)
        self._matching_with_zero_active_steps += int(matching > 0 and active == 0)
        self._predicted_columns_total += predicted_columns
        self._punished_segments_total += punished
        self._trace_digest.update(_WINNER_TRACE_ROW.pack(*winners))
        return winners

    def finalize(self, *, terminal_segment_count: int) -> dict[str, object]:
        """Require native storage to equal the sum of actual allocations."""

        if type(terminal_segment_count) is not int or terminal_segment_count < 0:
            raise ValueError("terminal_segment_count")
        if terminal_segment_count != self._actual_allocations:
            raise FactorizedSuccessorStop(
                "TERMINAL_SEGMENT_ACCOUNTING_DRIFT",
                self._observed_steps,
                {
                    "expected_sum_of_actual_allocations": self._actual_allocations,
                    "observed_terminal_segment_count": terminal_segment_count,
                },
            )
        return self.summary()

    def summary(self) -> dict[str, object]:
        """Return identity and non-gating matching/active telemetry separately."""

        return {
            "observed_steps": self._observed_steps,
            "complete_key_count": len(self._complete_keys),
            "column_binding_count": self._column_binding_count,
            "owner_column_binding_count": len(self._owner_columns),
            "novel_complete_keys": self._novel_keys,
            "repeated_complete_keys": self._repeated_keys,
            "partial_novel_allocation_events": self._partial_novel_allocations,
            "actual_allocated_segments": self._actual_allocations,
            "allocation_histogram": {
                str(value): count
                for value, count in sorted(self._allocation_histogram.items())
            },
            "matching_active_telemetry": {
                "matching_segments_total": self._matching_total,
                "active_segments_total": self._active_total,
                "matching_minus_active_segments_total": (
                    self._matching_total - self._active_total
                ),
                "maximum_matching_segments": self._matching_max,
                "maximum_active_segments": self._active_max,
                "matching_nonzero_steps": self._matching_nonzero_steps,
                "active_nonzero_steps": self._active_nonzero_steps,
                "matching_exceeds_active_steps": self._matching_exceeds_active_steps,
                "matching_with_zero_active_steps": (
                    self._matching_with_zero_active_steps
                ),
                "pair_histogram": {
                    f"{matching}:{active}": count
                    for (matching, active), count in sorted(
                        self._matching_active_histogram.items()
                    )
                },
            },
            "predicted_columns_total": self._predicted_columns_total,
            "punished_segments_total": self._punished_segments_total,
            "winner_trace_sha256_le_u32": self._trace_digest.hexdigest(),
        }
