"""Bounded exact-query interface over materialized and procedural networks."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Mapping, Protocol

import numpy as np

from humanbrain.observatory import evenly_spaced_gids
from humanbrain.procedural_topology import (
    ProceduralTopology,
    TOPOLOGY_PROTOCOL,
    TopologyManifest,
)
from humanbrain.topology import build_recurrent_targets

from .model import ExperimentRun


MAX_RECONSTRUCTED_DENSE_NEURONS = 4_096


class StructureUnavailable(RuntimeError):
    """The result lacks a bounded exact-query representation of its topology."""


@dataclass(frozen=True, slots=True)
class StructureInfo:
    configured_neurons: int
    enabled_neurons: int
    fanout: int
    topology: str
    representation: str
    arbitrary_gid_queries: bool


@dataclass(frozen=True, slots=True)
class StructureNode:
    gid: int
    outgoing_role: str
    model_family: str
    coordinates: tuple[int, ...] | None = None
    coordinate_system: str | None = None
    groups: tuple[tuple[str, int], ...] = ()
    layer: int | None = None


@dataclass(frozen=True, slots=True)
class StructureEdge:
    source_gid: int
    target_gid: int
    slot: int


@dataclass(frozen=True, slots=True)
class Neighborhood:
    root_gid: int
    hops: int
    nodes: tuple[StructureNode, ...]
    edges: tuple[StructureEdge, ...]
    depth_by_gid: Mapping[int, int]
    complete: bool
    omitted_nodes: int


@dataclass(frozen=True, slots=True)
class DensityOverview:
    bins: tuple[tuple[int, ...], ...]
    sampled_sources: int
    total_sources: int
    exact: bool


class _StructureAdapter(Protocol):
    info: StructureInfo

    def describe(self, gid: int) -> StructureNode: ...

    def outgoing(self, gid: int) -> tuple[StructureEdge, ...]: ...


class NetworkStructure:
    """Deep bounded interface shared by every topology representation."""

    def __init__(self, adapter: _StructureAdapter) -> None:
        self._adapter = adapter

    @property
    def info(self) -> StructureInfo:
        return self._adapter.info

    def describe(self, gid: int) -> StructureNode:
        return self._adapter.describe(gid)

    def neighborhood(
        self,
        root_gid: int,
        *,
        hops: int = 2,
        max_nodes: int = 160,
    ) -> Neighborhood:
        self.describe(root_gid)
        if not 1 <= hops <= 3:
            raise ValueError("hops must be between 1 and 3")
        if max_nodes < 2:
            raise ValueError("max_nodes must be at least two")
        visible = {root_gid}
        depth_by_gid = {root_gid: 0}
        frontier = [root_gid]
        edges: list[StructureEdge] = []
        omitted: set[int] = set()
        for depth in range(1, hops + 1):
            next_frontier: list[int] = []
            for source_gid in frontier:
                for edge in self._adapter.outgoing(source_gid):
                    if edge.target_gid not in visible:
                        if len(visible) >= max_nodes:
                            omitted.add(edge.target_gid)
                            continue
                        visible.add(edge.target_gid)
                        depth_by_gid[edge.target_gid] = depth
                        next_frontier.append(edge.target_gid)
                    edges.append(edge)
            frontier = next_frontier
            if not frontier:
                break
        nodes = tuple(self._adapter.describe(gid) for gid in sorted(visible))
        return Neighborhood(
            root_gid=root_gid,
            hops=hops,
            nodes=nodes,
            edges=tuple(edges),
            depth_by_gid=depth_by_gid,
            complete=not omitted,
            omitted_nodes=len(omitted),
        )

    def density_overview(
        self,
        *,
        bin_count: int = 24,
        sample_limit: int = 4_096,
    ) -> DensityOverview:
        if not 4 <= bin_count <= 64:
            raise ValueError("bin_count must be between 4 and 64")
        if sample_limit < 1:
            raise ValueError("sample_limit must be positive")
        total = self.info.enabled_neurons
        sources = evenly_spaced_gids(total, min(total, sample_limit))
        bins = [[0 for _ in range(bin_count)] for _ in range(bin_count)]
        for source_gid in sources:
            source_bin = min(bin_count - 1, source_gid * bin_count // total)
            for edge in self._adapter.outgoing(source_gid):
                target_bin = min(
                    bin_count - 1,
                    edge.target_gid * bin_count // total,
                )
                bins[source_bin][target_bin] += 1
        return DensityOverview(
            bins=tuple(tuple(row) for row in bins),
            sampled_sources=len(sources),
            total_sources=total,
            exact=len(sources) == total,
        )


class _DenseLadderAdapter:
    def __init__(self, run: ExperimentRun) -> None:
        configured = run.population.configured_neurons
        if configured > MAX_RECONSTRUCTED_DENSE_NEURONS:
            raise StructureUnavailable(
                "this materialized ladder result did not record adjacency; exact "
                f"reconstruction is capped at {MAX_RECONSTRUCTED_DENSE_NEURONS:,} "
                "neurons. Record a procedural topology manifest for larger runs."
            )
        circuit = run.configuration.get("circuit_config")
        if not isinstance(circuit, dict):
            raise StructureUnavailable("result has no circuit_config")
        enabled_value = circuit.get("active_reservoir_cells")
        enabled = configured if enabled_value is None else int(enabled_value)
        fanout = int(circuit["recurrent_fanout"])
        topology = str(circuit.get("recurrent_topology", "random"))
        seed = int(circuit["random_seed"])
        token_cells = int(run.configuration["token_cells"])
        input_fanout = int(run.configuration["input_fanout"])

        rng = np.random.default_rng(seed)
        for _ in range(token_cells):
            rng.choice(enabled, input_fanout, replace=False)
        self._targets = np.zeros((configured, fanout), dtype=np.int32)
        self._targets[:enabled] = build_recurrent_targets(
            enabled,
            fanout,
            rng,
            topology,
        )
        inhibitory_fraction = float(circuit["inhibitory_fraction"])
        self._roles = np.full(configured, "dormant", dtype=object)
        self._roles[:enabled] = "excitatory"
        inhibitory_count = round(inhibitory_fraction * enabled)
        if inhibitory_count:
            inhibitory = rng.choice(enabled, inhibitory_count, replace=False)
            self._roles[inhibitory] = "inhibitory"
        self._enabled = enabled
        self._model_family = run.population.model_family
        self.info = StructureInfo(
            configured_neurons=configured,
            enabled_neurons=enabled,
            fanout=fanout,
            topology=topology,
            representation="exact reconstructed materialized topology",
            arbitrary_gid_queries=True,
        )

    def describe(self, gid: int) -> StructureNode:
        _validate_gid(gid, self.info.configured_neurons)
        return StructureNode(
            gid=gid,
            outgoing_role=str(self._roles[gid]),
            model_family=self._model_family,
        )

    def outgoing(self, gid: int) -> tuple[StructureEdge, ...]:
        _validate_gid(gid, self.info.configured_neurons)
        if gid >= self._enabled:
            return ()
        return tuple(
            StructureEdge(source_gid=gid, target_gid=int(target), slot=slot)
            for slot, target in enumerate(self._targets[gid])
        )


class _ProceduralAdapter:
    def __init__(self, manifest: TopologyManifest, *, model_family: str) -> None:
        self._topology = ProceduralTopology(manifest)
        self._model_family = model_family
        self.info = StructureInfo(
            configured_neurons=manifest.cell_count,
            enabled_neurons=manifest.cell_count,
            fanout=manifest.fanout,
            topology=manifest.topology,
            representation=f"exact procedural oracle / {manifest.protocol}",
            arbitrary_gid_queries=True,
        )

    def describe(self, gid: int) -> StructureNode:
        descriptor = self._topology.describe_cell(gid)
        return StructureNode(
            gid=gid,
            outgoing_role=descriptor.outgoing_connectivity_role,
            model_family=self._model_family,
            coordinates=descriptor.coordinates,
            coordinate_system=descriptor.coordinate_system,
            groups=descriptor.groups,
            layer=descriptor.layer,
        )

    def outgoing(self, gid: int) -> tuple[StructureEdge, ...]:
        return tuple(
            StructureEdge(
                source_gid=edge.source_gid,
                target_gid=edge.target_gid,
                slot=edge.edge_slot,
            )
            for edge in self._topology.base_edges(gid)
        )


def open_structure(run: ExperimentRun) -> NetworkStructure:
    manifest_record = run.configuration.get("procedural_topology_manifest")
    if isinstance(manifest_record, dict):
        return open_procedural_structure(
            manifest_record,
            model_family=run.population.model_family,
        )
    return NetworkStructure(_DenseLadderAdapter(run))


def open_procedural_structure(
    manifest_record: Mapping[str, object],
    *,
    model_family: str = "GLIF5",
) -> NetworkStructure:
    manifest = TopologyManifest(
        cell_count=manifest_record["cell_count"],  # type: ignore[arg-type]
        fanout=manifest_record["fanout"],  # type: ignore[arg-type]
        topology=manifest_record["topology"],  # type: ignore[arg-type]
        seed=manifest_record["seed"],  # type: ignore[arg-type]
        protocol=manifest_record.get("protocol", TOPOLOGY_PROTOCOL),  # type: ignore[arg-type]
    )
    return NetworkStructure(_ProceduralAdapter(manifest, model_family=model_family))


def load_structure_manifest(path: str | Path) -> NetworkStructure:
    source = Path(path)
    try:
        record = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise StructureUnavailable(f"cannot read structure manifest: {error}") from error
    if not isinstance(record, dict):
        raise StructureUnavailable("structure manifest root must be an object")
    nested = record.get("procedural_topology_manifest")
    if nested is not None:
        if not isinstance(nested, dict):
            raise StructureUnavailable("procedural_topology_manifest must be an object")
        record = nested
    try:
        return open_procedural_structure(record)
    except (KeyError, TypeError, ValueError) as error:
        raise StructureUnavailable(f"invalid structure manifest: {error}") from error


def _validate_gid(gid: object, cell_count: int) -> None:
    if type(gid) is not int or not 0 <= gid < cell_count:
        raise ValueError(f"gid must be an exact integer in [0, {cell_count})")
