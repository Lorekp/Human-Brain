"""Scalar 64-bit procedural topology queries for configured cell identities."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd
from types import MappingProxyType
from typing import Callable


_UINT64_MASK = (1 << 64) - 1
_UINT64_CARDINALITY = 1 << 64
_MAX_EDGE_SLOTS = 1 << 48
_PHILOX_M2X64_0 = 0xD2B74407B1CE6E93
_PHILOX_W64_0 = 0x9E3779B97F4A7C15
_MAX_CANDIDATE_ATTEMPTS = 128
TOPOLOGY_PROTOCOL = "random123-philox2x64-10-v1"

__all__ = [
    "BaseEdge",
    "CellDescriptor",
    "IntrinsicModelIdentity",
    "ProceduralTopology",
    "TOPOLOGY_PROTOCOL",
    "TopologyManifest",
]


def _require_exact_int(owner: str, value: object, *, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{owner} must be an exact integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{owner} must be in [{minimum}, {maximum}]")
    return value


@dataclass(frozen=True, slots=True)
class TopologyManifest:
    """Constant-space declaration of one contiguous uint64 GID address range."""

    cell_count: int
    fanout: int
    topology: str
    seed: int
    protocol: str = TOPOLOGY_PROTOCOL

    def __post_init__(self) -> None:
        _require_exact_int(
            "manifest.cell_count",
            self.cell_count,
            minimum=2,
            maximum=_UINT64_CARDINALITY,
        )
        _require_exact_int(
            "manifest.fanout",
            self.fanout,
            minimum=1,
            maximum=min(self.cell_count - 1, _MAX_EDGE_SLOTS - 1),
        )
        _require_exact_int(
            "manifest.seed",
            self.seed,
            minimum=0,
            maximum=_UINT64_MASK,
        )
        if not isinstance(self.topology, str):
            raise TypeError("manifest.topology must be a string")
        if self.topology not in _TOPOLOGY_SPECS:
            raise ValueError(f"unknown topology: {self.topology}")
        if not isinstance(self.protocol, str):
            raise TypeError("manifest.protocol must be a string")
        if self.protocol != TOPOLOGY_PROTOCOL:
            raise ValueError(f"unsupported topology protocol: {self.protocol}")

    @property
    def gid_range(self) -> tuple[int, int]:
        return (0, self.cell_count)

    @property
    def gid_width_bits(self) -> int:
        return 64


@dataclass(frozen=True, slots=True)
class IntrinsicModelIdentity:
    model_id: int
    specimen_id: int
    model_name: str
    species: str
    structure: str
    config_sha256: str


_PINNED_HUMAN_GLIF5 = IntrinsicModelIdentity(
    model_id=523000114,
    specimen_id=508378649,
    model_name=(
        "5 LIF-R-ASC + Threshold Adaptation "
        "(LIF-R-ASC-A)_H16.06.004.01.05.02"
    ),
    species="Homo sapiens",
    structure="middle temporal gyrus",
    config_sha256=(
        "69fe726023e3289ac85857ea9125c55fee1618891e7b65403784c1b961bad149"
    ),
)


@dataclass(frozen=True, slots=True)
class CellDescriptor:
    """Immutable identity metadata; it carries no resident dynamical state."""

    gid: int
    role: str
    cell_type: str
    outgoing_connectivity_role: str
    coordinates: tuple[int, ...] | None
    coordinate_system: str | None
    groups: tuple[tuple[str, int], ...]
    layer: int | None
    intrinsic_model: IntrinsicModelIdentity


@dataclass(frozen=True, slots=True)
class BaseEdge:
    source_gid: int
    edge_slot: int
    target_gid: int


def _ceil_root(value: int, dimensions: int) -> int:
    high = 1 << ((value.bit_length() + dimensions - 1) // dimensions)
    low = 0
    while low + 1 < high:
        midpoint = (low + high) // 2
        if midpoint**dimensions >= value:
            high = midpoint
        else:
            low = midpoint
    return high


def _grid_coordinates(gid: int, side: int, dimensions: int) -> tuple[int, ...]:
    coordinates: list[int] = []
    value = gid
    for _ in range(dimensions):
        coordinates.append(value % side)
        value //= side
    return tuple(coordinates)


def _balanced_part_for_gid(total: int, parts: int, gid: int) -> int:
    small_size, large_parts = divmod(total, parts)
    large_size = small_size + 1
    large_span = large_parts * large_size
    if gid < large_span:
        return gid // large_size
    return large_parts + (gid - large_span) // small_size


def _balanced_part_bounds(total: int, parts: int, part: int) -> tuple[int, int]:
    small_size, large_parts = divmod(total, parts)
    if part < large_parts:
        start = part * (small_size + 1)
        return start, start + small_size + 1
    start = large_parts * (small_size + 1) + (part - large_parts) * small_size
    return start, start + small_size


def _micro_count(manifest: TopologyManifest) -> int:
    nominal_size = max(manifest.fanout + 1, _ceil_root(manifest.cell_count, 2))
    return max(1, manifest.cell_count // nominal_size)


def _mixed_group_count(manifest: TopologyManifest) -> int:
    count = min(manifest.cell_count, max(2, _ceil_root(manifest.cell_count, 2)))
    if gcd(5, count) == 1:
        return count
    return count + 1 if count < manifest.cell_count else count - 1


_DescriptorFields = tuple[
    tuple[int, ...] | None,
    tuple[tuple[str, int], ...],
    int | None,
]


def _plain_descriptor(_: TopologyManifest, __: int) -> _DescriptorFields:
    return None, (), None


def _single_group_descriptor(manifest: TopologyManifest, gid: int) -> _DescriptorFields:
    micro = _balanced_part_for_gid(manifest.cell_count, _micro_count(manifest), gid)
    return None, (("micro", micro),), None


def _nested_group_descriptor(manifest: TopologyManifest, gid: int) -> _DescriptorFields:
    micro = _balanced_part_for_gid(manifest.cell_count, _micro_count(manifest), gid)
    return None, (("micro", micro), ("meso", micro // 4)), None


def _mixed_descriptor(manifest: TopologyManifest, gid: int) -> _DescriptorFields:
    group_count = _mixed_group_count(manifest)
    return (
        None,
        (
            ("primary", gid % group_count),
            ("secondary", (gid * 5 + 1) % group_count),
        ),
        None,
    )


def _ring_descriptor(_: TopologyManifest, gid: int) -> _DescriptorFields:
    return (gid,), (), None


def _sheet_descriptor(manifest: TopologyManifest, gid: int) -> _DescriptorFields:
    return _grid_coordinates(gid, _ceil_root(manifest.cell_count, 2), 2), (), None


def _layered_descriptor(manifest: TopologyManifest, gid: int) -> _DescriptorFields:
    layers = 6
    surface_gid, layer = divmod(gid, layers)
    surface_cells = (manifest.cell_count + layers - 1) // layers
    x, y = _grid_coordinates(surface_gid, _ceil_root(surface_cells, 2), 2)
    return (x, y, layer), (), layer


def _volume_descriptor(manifest: TopologyManifest, gid: int) -> _DescriptorFields:
    return _grid_coordinates(gid, _ceil_root(manifest.cell_count, 3), 3), (), None


def _hypercube_descriptor(manifest: TopologyManifest, gid: int) -> _DescriptorFields:
    return _grid_coordinates(gid, _ceil_root(manifest.cell_count, 4), 4), (), None


_Candidate = Callable[["ProceduralTopology", int, int, int], int | None]


def _counter_words(
    oracle: "ProceduralTopology",
    source: int,
    edge_slot: int,
    purpose: int,
    attempt: int,
) -> tuple[int, int]:
    context = (edge_slot << 16) | (purpose << 8) | attempt
    key = (
        oracle.manifest.seed + oracle._spec.stream_id * _PHILOX_W64_0
    ) & _UINT64_MASK
    return _philox2x64_10(source, context, key)


def _bounded_index(word: int, size: int) -> int:
    return (word * size) >> 64


def _global_candidate(
    oracle: "ProceduralTopology",
    source: int,
    edge_slot: int,
    purpose: int,
    attempt: int,
) -> int:
    word, _ = _counter_words(oracle, source, edge_slot, purpose, attempt)
    candidate = _bounded_index(word, oracle.manifest.cell_count - 1)
    return candidate if candidate < source else candidate + 1


def _ranges_candidate(
    ranges: tuple[tuple[int, int], ...],
    source: int,
    word: int,
) -> int | None:
    sizes = tuple(
        stop - start - int(start <= source < stop)
        for start, stop in ranges
    )
    total = sum(sizes)
    if total == 0:
        return None
    offset = _bounded_index(word, total)
    for (start, stop), size in zip(ranges, sizes, strict=True):
        if offset >= size:
            offset -= size
            continue
        candidate = start + offset
        if start <= source < stop and candidate >= source:
            candidate += 1
        return candidate
    raise AssertionError("a non-empty range union must yield a candidate")


def _progression_candidate(
    base: int,
    step: int,
    stop: int,
    word: int,
) -> int | None:
    if base >= stop:
        return None
    size = (stop - 1 - base) // step + 1
    return base + _bounded_index(word, size) * step


def _build_edges_with_stats(
    oracle: "ProceduralTopology",
    source: int,
    candidate_for: _Candidate,
) -> tuple[tuple[BaseEdge, ...], int]:
    """Build one row with a worst-case ``129*fanout + 1`` work bound.

    The bounded topology sampler contributes at most 128 probes per slot. A
    single monotonic fallback cursor is shared by the row, so each fallback GID
    is examined at most once and the cursor advances at most ``fanout + 1``
    times in total.
    """

    used = {source}
    edges: list[BaseEdge] = []
    fallback_word, _ = _counter_words(oracle, source, 0, 255, 0)
    fallback_start = _bounded_index(fallback_word, oracle.manifest.cell_count)
    fallback_offset = 0
    candidate_examinations = 0
    for edge_slot in range(oracle.manifest.fanout):
        candidate = None
        for attempt in range(_MAX_CANDIDATE_ATTEMPTS):
            proposed = candidate_for(oracle, source, edge_slot, attempt)
            candidate_examinations += 1
            if proposed is not None and proposed not in used:
                candidate = proposed
                break
        if candidate is None:
            while fallback_offset <= oracle.manifest.fanout:
                proposed = (
                    fallback_start + fallback_offset
                ) % oracle.manifest.cell_count
                fallback_offset += 1
                candidate_examinations += 1
                if proposed not in used:
                    candidate = proposed
                    break
        if candidate is None:
            raise AssertionError(
                "fanout smaller than the population guarantees a fallback target"
            )
        used.add(candidate)
        edges.append(BaseEdge(source, edge_slot, candidate))
    return tuple(edges), candidate_examinations


def _build_edges(
    oracle: "ProceduralTopology",
    source: int,
    candidate_for: _Candidate,
) -> tuple[BaseEdge, ...]:
    edges, _ = _build_edges_with_stats(oracle, source, candidate_for)
    return edges


def _global_edges(oracle: "ProceduralTopology", source: int) -> tuple[BaseEdge, ...]:
    def candidate(
        inner: "ProceduralTopology", gid: int, slot: int, attempt: int
    ) -> int:
        return _global_candidate(inner, gid, slot, 1, attempt)

    return _build_edges(oracle, source, candidate)


def _single_group_edges(
    oracle: "ProceduralTopology", source: int
) -> tuple[BaseEdge, ...]:
    manifest = oracle.manifest
    micro_count = _micro_count(manifest)
    micro = _balanced_part_for_gid(manifest.cell_count, micro_count, source)
    micro_start, micro_stop = _balanced_part_bounds(
        manifest.cell_count, micro_count, micro
    )
    local_slots = manifest.fanout - 1
    outside = tuple(
        interval
        for interval in ((0, micro_start), (micro_stop, manifest.cell_count))
        if interval[0] < interval[1]
    )

    def candidate(
        inner: "ProceduralTopology", gid: int, slot: int, attempt: int
    ) -> int | None:
        word, _ = _counter_words(inner, gid, slot, 2 if slot < local_slots else 3, attempt)
        ranges = ((micro_start, micro_stop),) if slot < local_slots else outside
        return _ranges_candidate(ranges, gid, word)

    return _build_edges(oracle, source, candidate)


def _nested_group_edges(
    oracle: "ProceduralTopology", source: int
) -> tuple[BaseEdge, ...]:
    manifest = oracle.manifest
    micro_count = _micro_count(manifest)
    micro = _balanced_part_for_gid(manifest.cell_count, micro_count, source)
    micro_start, micro_stop = _balanced_part_bounds(
        manifest.cell_count, micro_count, micro
    )
    meso_first = (micro // 4) * 4
    meso_last = min(micro_count, meso_first + 4) - 1
    meso_start = _balanced_part_bounds(
        manifest.cell_count, micro_count, meso_first
    )[0]
    meso_stop = _balanced_part_bounds(
        manifest.cell_count, micro_count, meso_last
    )[1]
    micro_slots = min((manifest.fanout + 1) // 2, micro_stop - micro_start - 1)
    meso_available = meso_stop - meso_start - (micro_stop - micro_start)
    meso_slots = min(
        max(1, manifest.fanout // 3),
        meso_available,
        manifest.fanout - micro_slots,
    )
    meso_ranges = tuple(
        interval
        for interval in ((meso_start, micro_start), (micro_stop, meso_stop))
        if interval[0] < interval[1]
    )
    outside = tuple(
        interval
        for interval in ((0, meso_start), (meso_stop, manifest.cell_count))
        if interval[0] < interval[1]
    )

    def candidate(
        inner: "ProceduralTopology", gid: int, slot: int, attempt: int
    ) -> int | None:
        if slot < micro_slots:
            ranges = ((micro_start, micro_stop),)
            purpose = 4
        elif slot < micro_slots + meso_slots:
            ranges = meso_ranges
            purpose = 5
        else:
            ranges = outside
            purpose = 6
        word, _ = _counter_words(inner, gid, slot, purpose, attempt)
        return _ranges_candidate(ranges, gid, word)

    return _build_edges(oracle, source, candidate)


def _mixed_membership_edges(
    oracle: "ProceduralTopology", source: int
) -> tuple[BaseEdge, ...]:
    manifest = oracle.manifest
    group_count = _mixed_group_count(manifest)
    memberships = (source % group_count, (source * 5 + 1) % group_count)
    shared_slots = manifest.fanout - max(1, manifest.fanout // 4)

    def candidate(
        inner: "ProceduralTopology", gid: int, slot: int, attempt: int
    ) -> int | None:
        if slot >= shared_slots:
            return _global_candidate(inner, gid, slot, 8, attempt)
        membership_word, member_word = _counter_words(inner, gid, slot, 7, attempt)
        membership = memberships[_bounded_index(membership_word, len(memberships))]
        return _progression_candidate(
            membership,
            group_count,
            manifest.cell_count,
            member_word,
        )

    return _build_edges(oracle, source, candidate)


def _ring_edges(oracle: "ProceduralTopology", source: int) -> tuple[BaseEdge, ...]:
    manifest = oracle.manifest
    local_slots = manifest.fanout - max(1, (15 * manifest.fanout + 50) // 100)
    radius = max(2, _ceil_root(manifest.cell_count, 4))

    def candidate(
        inner: "ProceduralTopology", gid: int, slot: int, attempt: int
    ) -> int | None:
        if slot < local_slots:
            distance_word, sign_word = _counter_words(inner, gid, slot, 9, attempt)
            distance = 1 + _bounded_index(distance_word, radius)
            if sign_word & 1:
                distance = -distance
            return (gid + distance) % manifest.cell_count
        proposed = _global_candidate(inner, gid, slot, 10, attempt)
        separation = abs(proposed - gid)
        cyclic = min(separation, manifest.cell_count - separation)
        return proposed if cyclic >= max(2, manifest.cell_count // 4) else None

    return _build_edges(oracle, source, candidate)


def _grid_edges(
    oracle: "ProceduralTopology",
    source: int,
    dimensions: int,
) -> tuple[BaseEdge, ...]:
    manifest = oracle.manifest
    side = _ceil_root(manifest.cell_count, dimensions)
    source_coordinates = _grid_coordinates(source, side, dimensions)
    local_slots = manifest.fanout - max(1, (15 * manifest.fanout + 50) // 100)

    def candidate(
        inner: "ProceduralTopology", gid: int, slot: int, attempt: int
    ) -> int | None:
        if slot < local_slots:
            offsets = tuple(
                _bounded_index(
                    _counter_words(inner, gid, slot, 11 + dimension, attempt)[0],
                    5,
                )
                - 2
                for dimension in range(dimensions)
            )
            if not any(offsets):
                return None
            coordinates = tuple(
                coordinate + offset
                for coordinate, offset in zip(
                    source_coordinates, offsets, strict=True
                )
            )
            if not all(0 <= coordinate < side for coordinate in coordinates):
                return None
            proposed = sum(
                coordinate * side**dimension
                for dimension, coordinate in enumerate(coordinates)
            )
            return proposed if proposed < manifest.cell_count else None
        proposed = _global_candidate(inner, gid, slot, 20, attempt)
        target_coordinates = _grid_coordinates(proposed, side, dimensions)
        squared_distance = sum(
            (left - right) ** 2
            for left, right in zip(
                source_coordinates, target_coordinates, strict=True
            )
        )
        return proposed if squared_distance >= max(2, side // 4) ** 2 else None

    return _build_edges(oracle, source, candidate)


def _sheet_edges(oracle: "ProceduralTopology", source: int) -> tuple[BaseEdge, ...]:
    return _grid_edges(oracle, source, 2)


def _volume_edges(oracle: "ProceduralTopology", source: int) -> tuple[BaseEdge, ...]:
    return _grid_edges(oracle, source, 3)


def _hypercube_edges(
    oracle: "ProceduralTopology", source: int
) -> tuple[BaseEdge, ...]:
    return _grid_edges(oracle, source, 4)


def _layered_edges(
    oracle: "ProceduralTopology", source: int
) -> tuple[BaseEdge, ...]:
    manifest = oracle.manifest
    layers = 6
    surface_cells = (manifest.cell_count + layers - 1) // layers
    side = _ceil_root(surface_cells, 2)
    source_surface, source_layer = divmod(source, layers)
    source_x, source_y = _grid_coordinates(source_surface, side, 2)
    local_slots = manifest.fanout - max(1, (15 * manifest.fanout + 50) // 100)

    def candidate(
        inner: "ProceduralTopology", gid: int, slot: int, attempt: int
    ) -> int | None:
        if slot < local_slots:
            dx = _bounded_index(_counter_words(inner, gid, slot, 15, attempt)[0], 5) - 2
            dy = _bounded_index(_counter_words(inner, gid, slot, 16, attempt)[0], 5) - 2
            depth = _bounded_index(
                _counter_words(inner, gid, slot, 17, attempt)[0], 3
            ) - 1
            if dx == 0 and dy == 0 and depth == 0:
                return None
            x, y, layer = source_x + dx, source_y + dy, source_layer + depth
            if not (0 <= x < side and 0 <= y < side and 0 <= layer < layers):
                return None
            proposed = (y * side + x) * layers + layer
            return proposed if proposed < manifest.cell_count else None
        proposed = _global_candidate(inner, gid, slot, 18, attempt)
        target_surface = proposed // layers
        target_x, target_y = _grid_coordinates(target_surface, side, 2)
        squared_distance = (source_x - target_x) ** 2 + (source_y - target_y) ** 2
        return proposed if squared_distance >= max(2, side // 4) ** 2 else None

    return _build_edges(oracle, source, candidate)


@dataclass(frozen=True, slots=True)
class _TopologySpec:
    stream_id: int
    coordinate_system: str | None
    describe: Callable[[TopologyManifest, int], _DescriptorFields]
    edges: Callable[["ProceduralTopology", int], tuple[BaseEdge, ...]]


_TOPOLOGY_SPECS = MappingProxyType(
    {
        "random": _TopologySpec(1, None, _plain_descriptor, _global_edges),
        "uniform_no_self": _TopologySpec(2, None, _plain_descriptor, _global_edges),
        "single_scale_groups": _TopologySpec(
            3, None, _single_group_descriptor, _single_group_edges
        ),
        "nested_groups": _TopologySpec(
            4, None, _nested_group_descriptor, _nested_group_edges
        ),
        "mixed_membership": _TopologySpec(
            5, None, _mixed_descriptor, _mixed_membership_edges
        ),
        "ring_1d": _TopologySpec(
            6, "ring_1d_integer_lattice", _ring_descriptor, _ring_edges
        ),
        "sheet_2d": _TopologySpec(
            7, "sheet_2d_integer_lattice", _sheet_descriptor, _sheet_edges
        ),
        "layered_sheet": _TopologySpec(
            8,
            "layered_sheet_integer_lattice",
            _layered_descriptor,
            _layered_edges,
        ),
        "volume_3d": _TopologySpec(
            9,
            "volume_3d_integer_lattice",
            _volume_descriptor,
            _volume_edges,
        ),
        "hypercube_4d": _TopologySpec(
            10,
            "abstract_hypercube_4d_integer_lattice",
            _hypercube_descriptor,
            _hypercube_edges,
        ),
    }
)


class ProceduralTopology:
    """Deterministic scalar oracle over one immutable topology manifest."""

    __slots__ = ("_manifest", "_spec")

    def __init__(self, manifest: TopologyManifest) -> None:
        if not isinstance(manifest, TopologyManifest):
            raise TypeError("manifest must be a TopologyManifest")
        self._manifest = manifest
        self._spec = _TOPOLOGY_SPECS[manifest.topology]

    @property
    def manifest(self) -> TopologyManifest:
        return self._manifest

    def describe_cell(self, gid: int) -> CellDescriptor:
        self._validate_gid(gid)
        coordinates, groups, layer = self._spec.describe(self.manifest, gid)
        role_word, _ = _philox2x64_10(gid, 0x524F4C45, self.manifest.seed)
        outgoing_role = (
            "inhibitory"
            if role_word < _UINT64_CARDINALITY // 5
            else "excitatory"
        )
        return CellDescriptor(
            gid=gid,
            role="reservoir",
            cell_type="adaptive_glif5",
            outgoing_connectivity_role=outgoing_role,
            coordinates=coordinates,
            coordinate_system=self._spec.coordinate_system,
            groups=groups,
            layer=layer,
            intrinsic_model=_PINNED_HUMAN_GLIF5,
        )

    def base_edges(self, gid: int) -> tuple[BaseEdge, ...]:
        self._validate_gid(gid)
        return self._spec.edges(self, gid)

    def _validate_gid(self, gid: object) -> None:
        _require_exact_int(
            "gid",
            gid,
            minimum=0,
            maximum=self.manifest.cell_count - 1,
        )


def _philox2x64_10(counter_0: int, counter_1: int, key: int) -> tuple[int, int]:
    """Return Random123 Philox-2x64-10 words in numeric ``v[0], v[1]`` order.

    Constants, round order, and word order follow D. E. Shaw Research's
    Random123 ``philox.h``. Numeric word order makes the result independent of
    host byte order, as specified by the Random123 known-answer vectors.
    """

    left = counter_0 & _UINT64_MASK
    right = counter_1 & _UINT64_MASK
    round_key = key & _UINT64_MASK
    for round_index in range(10):
        product = _PHILOX_M2X64_0 * left
        high = product >> 64
        low = product & _UINT64_MASK
        left, right = (high ^ round_key ^ right) & _UINT64_MASK, low
        if round_index != 9:
            round_key = (round_key + _PHILOX_W64_0) & _UINT64_MASK
    return left, right
