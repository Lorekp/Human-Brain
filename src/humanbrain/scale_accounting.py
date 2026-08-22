"""Canonical numerical scale counters for Human Brain benchmark claims.

This module does not define the manifest metadata needed to establish a wide
address space: region/type counts, ID ranges and width, query behavior,
allocation tiers, and base-edge rule versions remain separate contracts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite


def _require_counts(owner: str, **counts: int) -> None:
    for name, value in counts.items():
        if type(value) is not int:
            raise TypeError(f"{owner}.{name} must be an exact non-negative integer")
        if value < 0:
            raise ValueError(f"{owner}.{name} must be an exact non-negative integer")


def _require_label(owner: str, name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{owner}.{name} must be a non-empty string")
    if not value.strip():
        raise ValueError(f"{owner}.{name} must be a non-empty string")


_ACTIVE_QUALIFIERS = frozenset({"addressable", "resident", "updated", "spiking"})


def _reject_unqualified_active(field: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"claim.{field} must be a string")
    separated = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    words = [word.lower() for word in re.findall(r"[A-Za-z0-9]+", separated)]
    for index, word in enumerate(words):
        if word != "active":
            continue
        previous = words[index - 1] if index > 0 else None
        following = words[index + 1] if index + 1 < len(words) else None
        if previous not in _ACTIVE_QUALIFIERS and following not in _ACTIVE_QUALIFIERS:
            raise ValueError(
                f"claim.{field} contains unqualified 'active'; use addressable, "
                "resident, updated, or spiking"
            )


@dataclass(frozen=True, slots=True)
class ScaleClaim:
    label: str
    text: str


@dataclass(frozen=True, slots=True)
class MeasurementInterval:
    """Named wall interval and its biological fixed-step interpretation."""

    name: str
    start_step: int
    step_count: int
    biological_step_seconds: float | None
    wall_seconds: float


@dataclass(frozen=True, slots=True)
class CellCounts:
    """Configured, addressable, resident, updated, and spiking identities."""

    configured: int
    addressable: int
    resident_peak: int
    resident_distinct: int
    updated_distinct: int
    spiking_distinct: int

    def __post_init__(self) -> None:
        _require_counts(
            "cells",
            configured=self.configured,
            addressable=self.addressable,
            resident_peak=self.resident_peak,
            resident_distinct=self.resident_distinct,
            updated_distinct=self.updated_distinct,
            spiking_distinct=self.spiking_distinct,
        )


@dataclass(frozen=True, slots=True)
class WorkCounts:
    """Measured cell work, separate from analytic step equivalents."""

    cell_step_evaluations: int
    spike_events: int
    analytic_catchups: int
    analytic_step_equivalents: int | None

    def __post_init__(self) -> None:
        _require_counts(
            "work",
            cell_step_evaluations=self.cell_step_evaluations,
            spike_events=self.spike_events,
            analytic_catchups=self.analytic_catchups,
        )
        if self.analytic_step_equivalents is not None:
            _require_counts(
                "work",
                analytic_step_equivalents=self.analytic_step_equivalents,
            )


@dataclass(frozen=True, slots=True)
class EdgeCounts:
    """Configured and resident records plus interval generation work."""

    configured: int
    resident: int
    mutable: int
    generated: int

    def __post_init__(self) -> None:
        _require_counts(
            "edges",
            configured=self.configured,
            resident=self.resident,
            mutable=self.mutable,
            generated=self.generated,
        )


@dataclass(frozen=True, slots=True)
class ShardPairTraffic:
    source_shard: str
    target_shard: str
    bytes: int

    def __post_init__(self) -> None:
        _require_label("shard pair traffic", "source_shard", self.source_shard)
        _require_label("shard pair traffic", "target_shard", self.target_shard)
        _require_counts("shard pair traffic", bytes=self.bytes)


@dataclass(frozen=True, slots=True)
class EventCounts:
    """Delivered events, maximum queue occupancy, and routed byte counts."""

    delivered: int
    queue_peak: int
    sent_bytes_by_shard_pair: tuple[ShardPairTraffic, ...]

    def __post_init__(self) -> None:
        _require_counts(
            "events",
            delivered=self.delivered,
            queue_peak=self.queue_peak,
        )
        if type(self.sent_bytes_by_shard_pair) is not tuple:
            raise TypeError(
                "events.sent_bytes_by_shard_pair must be an immutable tuple"
            )
        if not all(
            isinstance(item, ShardPairTraffic)
            for item in self.sent_bytes_by_shard_pair
        ):
            raise TypeError(
                "events.sent_bytes_by_shard_pair must contain ShardPairTraffic records"
            )
        pairs = [
            (item.source_shard, item.target_shard)
            for item in self.sent_bytes_by_shard_pair
        ]
        if len(pairs) != len(set(pairs)):
            raise ValueError("events contain a duplicate shard pair")


@dataclass(frozen=True, slots=True)
class ShardMemory:
    shard: str
    host_bytes: int
    device_bytes: int

    def __post_init__(self) -> None:
        _require_label("shard memory", "shard", self.shard)
        _require_counts(
            "shard memory",
            host_bytes=self.host_bytes,
            device_bytes=self.device_bytes,
        )


@dataclass(frozen=True, slots=True)
class MemoryCounts:
    """Exact accounted host and device payload bytes for each shard."""

    bytes_by_shard: tuple[ShardMemory, ...]

    def __post_init__(self) -> None:
        if type(self.bytes_by_shard) is not tuple:
            raise TypeError("memory.bytes_by_shard must be an immutable tuple")
        if not all(isinstance(item, ShardMemory) for item in self.bytes_by_shard):
            raise TypeError("memory.bytes_by_shard must contain ShardMemory records")
        shards = [item.shard for item in self.bytes_by_shard]
        if len(shards) != len(set(shards)):
            raise ValueError("memory contains a duplicate shard")


@dataclass(frozen=True, slots=True)
class ScaleAccounting:
    """Validate one immutable numerical scale claim.

    Callers supply counters maintained by their existing cell, edge, and event
    loops. Validation and serialization depend only on shard records, not the
    configured, addressable, or resident population size; this module never scans
    a population to infer a distinct count.
    """

    claim: ScaleClaim
    interval: MeasurementInterval
    cells: CellCounts
    work: WorkCounts
    edges: EdgeCounts
    events: EventCounts
    memory: MemoryCounts

    def __post_init__(self) -> None:
        _require_label("claim", "label", self.claim.label)
        _require_label("claim", "text", self.claim.text)
        _reject_unqualified_active("label", self.claim.label)
        _reject_unqualified_active("text", self.claim.text)

        interval = self.interval
        if not isinstance(interval.name, str):
            raise TypeError("interval.name must be a non-empty string")
        if not interval.name.strip():
            raise ValueError("interval.name must be a non-empty string")
        _require_counts(
            "interval",
            start_step=interval.start_step,
            step_count=interval.step_count,
        )
        step_seconds = interval.biological_step_seconds
        if interval.step_count > 0 and step_seconds is None:
            raise ValueError(
                "interval.biological_step_seconds is required when step_count is "
                "nonzero"
            )
        if step_seconds is not None and (
            isinstance(step_seconds, bool)
            or not isinstance(step_seconds, (int, float))
            or not isfinite(step_seconds)
            or step_seconds <= 0
        ):
            raise ValueError(
                "interval.biological_step_seconds must be a finite positive number"
            )
        wall_seconds = interval.wall_seconds
        if (
            isinstance(wall_seconds, bool)
            or not isinstance(wall_seconds, (int, float))
            or not isfinite(wall_seconds)
            or wall_seconds < 0
        ):
            raise ValueError(
                "interval.wall_seconds must be a finite non-negative number"
            )

        cells = self.cells
        if not (
            cells.spiking_distinct
            <= cells.updated_distinct
            <= cells.resident_distinct
            <= cells.addressable
            <= cells.configured
        ) or cells.resident_peak > cells.resident_distinct:
            raise ValueError(
                "cell scale relation must satisfy "
                "spiking_distinct <= updated_distinct <= resident_distinct <= "
                "addressable <= configured and resident_peak <= resident_distinct"
            )
        if (cells.resident_peak == 0) != (cells.resident_distinct == 0):
            raise ValueError(
                "resident_peak and resident_distinct must either both be zero or both "
                "describe resident cells"
            )

        work = self.work
        update_operations = work.cell_step_evaluations + work.analytic_catchups
        if (
            (cells.updated_distinct == 0) != (update_operations == 0)
            or cells.updated_distinct > update_operations
        ):
            raise ValueError(
                "updated_distinct must be supported by cell_step_evaluations or "
                "analytic_catchups"
            )
        if (
            (cells.spiking_distinct == 0) != (work.spike_events == 0)
            or cells.spiking_distinct > work.spike_events
        ):
            raise ValueError(
                "spiking_distinct and spike_events must describe the same spikes"
            )
        equivalents = work.analytic_step_equivalents
        if equivalents is not None and (
            equivalents < work.analytic_catchups
            or (work.analytic_catchups == 0 and equivalents != 0)
        ):
            raise ValueError(
                "reported analytic_step_equivalents must be zero without "
                "analytic_catchups and at least analytic_catchups otherwise"
            )
        if interval.step_count == 0 and (
            update_operations != 0
            or work.spike_events != 0
            or self.events.delivered != 0
        ):
            raise ValueError(
                "step_count must be nonzero when dynamic work or event delivery is "
                "reported"
            )
    def to_dict(self) -> dict[str, object]:
        traffic = sorted(
            self.events.sent_bytes_by_shard_pair,
            key=lambda item: (item.source_shard, item.target_shard),
        )
        shards = sorted(self.memory.bytes_by_shard, key=lambda item: item.shard)
        return {
            "claim": {"label": self.claim.label, "text": self.claim.text},
            "interval": {
                "name": self.interval.name,
                "start_step": self.interval.start_step,
                "step_count": self.interval.step_count,
                "biological_step_seconds": self.interval.biological_step_seconds,
                "wall_seconds": self.interval.wall_seconds,
            },
            "cells": {
                "configured": self.cells.configured,
                "addressable": self.cells.addressable,
                "resident_peak": self.cells.resident_peak,
                "resident_distinct": self.cells.resident_distinct,
                "updated_distinct": self.cells.updated_distinct,
                "spiking_distinct": self.cells.spiking_distinct,
            },
            "work": {
                "cell_step_evaluations": self.work.cell_step_evaluations,
                "spike_events": self.work.spike_events,
                "analytic_catchups": self.work.analytic_catchups,
                "analytic_step_equivalents": (
                    self.work.analytic_step_equivalents
                ),
            },
            "edges": {
                "configured": self.edges.configured,
                "resident": self.edges.resident,
                "mutable": self.edges.mutable,
                "generated": self.edges.generated,
            },
            "events": {
                "delivered": self.events.delivered,
                "queue_peak": self.events.queue_peak,
                "sent_bytes_by_shard_pair": [
                    {
                        "source_shard": item.source_shard,
                        "target_shard": item.target_shard,
                        "bytes": item.bytes,
                    }
                    for item in traffic
                ],
            },
            "memory": {
                "host_bytes": sum(item.host_bytes for item in shards),
                "device_bytes": sum(item.device_bytes for item in shards),
                "bytes_by_shard": [
                    {
                        "shard": item.shard,
                        "host_bytes": item.host_bytes,
                        "device_bytes": item.device_bytes,
                    }
                    for item in shards
                ],
            },
        }


__all__ = [
    "CellCounts",
    "EdgeCounts",
    "EventCounts",
    "MeasurementInterval",
    "MemoryCounts",
    "ScaleAccounting",
    "ScaleClaim",
    "ShardMemory",
    "ShardPairTraffic",
    "WorkCounts",
]
