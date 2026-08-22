from __future__ import annotations

from math import ceil, gcd

import numpy as np

RECURRENT_TOPOLOGIES = frozenset(
    {
        "random",
        "uniform_no_self",
        "single_scale_groups",
        "nested_groups",
        "ring_1d",
        "sheet_2d",
        "layered_sheet",
        "volume_3d",
        "hypercube_4d",
        "mixed_membership",
    }
)

# At and above this population size, uniform_no_self switches to the vectorized
# O(cells * fanout) builder. Below it the legacy row-at-a-time sampler keeps
# its exact registered seeded streams (pinned by test_uniform_sampling_
# preserves_legacy_seeded_targets). The threshold matches the CUDA engineering
# scale boundary where Python-loop construction otherwise dominates wall time.
_UNIFORM_NO_SELF_VECTOR_MIN_CELLS = 100_000


def _grid_coordinates(cell_count: int, dimensions: int) -> np.ndarray:
    side = ceil(cell_count ** (1.0 / dimensions))
    coordinates = np.empty((cell_count, dimensions), dtype=np.float64)
    for cell in range(cell_count):
        value = cell
        for dimension in range(dimensions):
            coordinates[cell, dimension] = value % side
            value //= side
    denominator = max(1, side - 1)
    return coordinates / denominator


def topology_coordinates(cell_count: int, topology: str) -> np.ndarray:
    if topology == "ring_1d":
        return np.arange(cell_count, dtype=np.float64)[:, None] / cell_count
    if topology == "sheet_2d":
        return _grid_coordinates(cell_count, 2)
    if topology == "volume_3d":
        return _grid_coordinates(cell_count, 3)
    if topology == "hypercube_4d":
        return _grid_coordinates(cell_count, 4)
    if topology == "layered_sheet":
        layers = 6
        surface_cells = ceil(cell_count / layers)
        surface = _grid_coordinates(surface_cells, 2)
        coordinates = np.empty((cell_count, 3), dtype=np.float64)
        for cell in range(cell_count):
            coordinates[cell, :2] = surface[cell // layers]
            coordinates[cell, 2] = (cell % layers) / (layers - 1)
        # Cortical depth is much smaller than either sheet dimension.
        coordinates[:, 2] *= 0.2
        return coordinates
    raise ValueError(f"topology {topology!r} does not have physical coordinates")


def _spatial_targets(
    cell_count: int,
    fanout: int,
    rng: np.random.Generator,
    topology: str,
) -> np.ndarray:
    targets = np.empty((cell_count, fanout), dtype=np.int32)
    local_edges = fanout - max(1, round(0.15 * fanout))
    for source in range(cell_count):
        row: list[int] = []
        used = {source}
        while len(row) < local_edges:
            candidate = _local_spatial_candidate(source, cell_count, topology, rng)
            if candidate not in used:
                used.add(candidate)
                row.append(candidate)
        while len(row) < fanout:
            candidate = _long_spatial_candidate(
                source,
                cell_count,
                topology,
                used,
                rng,
            )
            used.add(candidate)
            row.append(candidate)
        targets[source] = row
    return targets


def _grid_index_coordinates(index: int, side: int, dimensions: int) -> tuple[int, ...]:
    coordinates: list[int] = []
    value = index
    for _ in range(dimensions):
        coordinates.append(value % side)
        value //= side
    return tuple(coordinates)


def _local_spatial_candidate(
    source: int,
    cell_count: int,
    topology: str,
    rng: np.random.Generator,
) -> int:
    if topology == "ring_1d":
        radius = max(2, round(cell_count**0.25))
        offset = int(rng.integers(1, radius + 1))
        if int(rng.integers(0, 2)):
            offset = -offset
        return (source + offset) % cell_count

    if topology == "layered_sheet":
        layers = 6
        surface_cells = ceil(cell_count / layers)
        side = ceil(surface_cells**0.5)
        surface = source // layers
        x, y = _grid_index_coordinates(surface, side, 2)
        layer = source % layers
        while True:
            dx = int(rng.integers(-2, 3))
            dy = int(rng.integers(-2, 3))
            depth = int(rng.integers(-1, 2))
            if dx == 0 and dy == 0 and depth == 0:
                continue
            next_x, next_y, next_layer = x + dx, y + dy, layer + depth
            if not (0 <= next_x < side and 0 <= next_y < side and 0 <= next_layer < layers):
                continue
            candidate = (next_y * side + next_x) * layers + next_layer
            if candidate < cell_count:
                return candidate

    dimensions = {
        "sheet_2d": 2,
        "volume_3d": 3,
        "hypercube_4d": 4,
    }[topology]
    side = ceil(cell_count ** (1.0 / dimensions))
    coordinates = _grid_index_coordinates(source, side, dimensions)
    while True:
        offsets = tuple(int(rng.integers(-2, 3)) for _ in range(dimensions))
        if not any(offsets):
            continue
        candidate_coordinates = tuple(
            coordinate + offset for coordinate, offset in zip(coordinates, offsets, strict=True)
        )
        if not all(0 <= coordinate < side for coordinate in candidate_coordinates):
            continue
        candidate = sum(
            coordinate * side**dimension
            for dimension, coordinate in enumerate(candidate_coordinates)
        )
        if candidate < cell_count:
            return candidate


def _is_long_range(
    source: int,
    candidate: int,
    cell_count: int,
    topology: str,
) -> bool:
    if topology == "ring_1d":
        separation = abs(candidate - source)
        return min(separation, cell_count - separation) >= max(2, cell_count // 4)
    if topology == "layered_sheet":
        surface_cells = ceil(cell_count / 6)
        side = ceil(surface_cells**0.5)
        source_xy = _grid_index_coordinates(source // 6, side, 2)
        candidate_xy = _grid_index_coordinates(candidate // 6, side, 2)
        squared_distance = sum(
            (left - right) ** 2
            for left, right in zip(source_xy, candidate_xy, strict=True)
        )
        return squared_distance >= max(2, side // 4) ** 2
    dimensions = {
        "sheet_2d": 2,
        "volume_3d": 3,
        "hypercube_4d": 4,
    }[topology]
    side = ceil(cell_count ** (1.0 / dimensions))
    source_coordinates = _grid_index_coordinates(source, side, dimensions)
    candidate_coordinates = _grid_index_coordinates(candidate, side, dimensions)
    squared_distance = sum(
        (left - right) ** 2
        for left, right in zip(source_coordinates, candidate_coordinates, strict=True)
    )
    return squared_distance >= max(2, side // 4) ** 2


def _long_spatial_candidate(
    source: int,
    cell_count: int,
    topology: str,
    used: set[int],
    rng: np.random.Generator,
) -> int:
    for _ in range(64):
        candidate = int(rng.integers(0, cell_count))
        if candidate not in used and _is_long_range(
            source,
            candidate,
            cell_count,
            topology,
        ):
            return candidate
    # Tiny/incompletely filled grids can have no point beyond the nominal
    # long-range threshold for a central source. Fall back to a global edge
    # without ever allocating or scoring a population-sized candidate array.
    while True:
        candidate = int(rng.integers(0, cell_count))
        if candidate not in used:
            return candidate


def _draw_from_ranges(
    ranges: tuple[tuple[int, int], ...],
    rng: np.random.Generator,
) -> int:
    sizes = tuple(max(0, stop - start) for start, stop in ranges)
    offset = int(rng.integers(0, sum(sizes)))
    for (start, _), size in zip(ranges, sizes, strict=True):
        if offset < size:
            return start + offset
        offset -= size
    raise AssertionError("nonempty ranges must yield a candidate")


def _append_from_ranges(
    row: list[int],
    used: set[int],
    ranges: tuple[tuple[int, int], ...],
    count: int,
    rng: np.random.Generator,
) -> None:
    target_length = len(row) + count
    while len(row) < target_length:
        candidate = _draw_from_ranges(ranges, rng)
        if candidate not in used:
            used.add(candidate)
            row.append(candidate)


def _group_targets(
    cell_count: int,
    fanout: int,
    rng: np.random.Generator,
    nested: bool,
) -> np.ndarray:
    micro_size = max(fanout + 1, round(cell_count**0.5))
    microgroups = np.arange(cell_count) // micro_size
    mesogroups = microgroups // 4
    targets = np.empty((cell_count, fanout), dtype=np.int32)
    for source in range(cell_count):
        micro = int(microgroups[source])
        micro_start = micro * micro_size
        micro_stop = min(cell_count, micro_start + micro_size)
        row: list[int] = []
        used = {source}
        micro_quota = min((fanout + 1) // 2 if nested else fanout - 1, micro_stop - micro_start - 1)
        _append_from_ranges(
            row,
            used,
            ((micro_start, micro_stop),),
            micro_quota,
            rng,
        )
        if nested:
            meso = int(mesogroups[source])
            meso_start = meso * micro_size * 4
            meso_stop = min(cell_count, meso_start + micro_size * 4)
            meso_ranges = tuple(
                interval
                for interval in (
                    (meso_start, micro_start),
                    (micro_stop, meso_stop),
                )
                if interval[1] > interval[0]
            )
            meso_quota = min(max(1, fanout // 3), sum(stop - start for start, stop in meso_ranges))
            _append_from_ranges(row, used, meso_ranges, meso_quota, rng)
            global_ranges = tuple(
                interval
                for interval in ((0, meso_start), (meso_stop, cell_count))
                if interval[1] > interval[0]
            )
        else:
            global_ranges = tuple(
                interval
                for interval in ((0, micro_start), (micro_stop, cell_count))
                if interval[1] > interval[0]
            )
        if not global_ranges:
            global_ranges = ((0, cell_count),)
        _append_from_ranges(
            row,
            used,
            global_ranges,
            fanout - len(row),
            rng,
        )
        targets[source] = row
    return targets


def _mixed_membership_targets(
    cell_count: int, fanout: int, rng: np.random.Generator
) -> np.ndarray:
    group_count = max(4, round(cell_count**0.5))
    while gcd(5, group_count) != 1:
        group_count += 1
    inverse_five = pow(5, -1, group_count)
    primary = np.arange(cell_count) % group_count
    secondary = (np.arange(cell_count) * 5 + 1) % group_count
    targets = np.empty((cell_count, fanout), dtype=np.int32)
    for source in range(cell_count):
        source_groups = (int(primary[source]), int(secondary[source]))
        progressions = tuple(
            (base, group_count)
            for group in source_groups
            for base in (
                group,
                ((group - 1) * inverse_five) % group_count,
            )
        )
        row: list[int] = []
        used = {source}
        shared_quota = fanout - max(1, fanout // 4)
        while len(row) < shared_quota:
            base, step = progressions[int(rng.integers(0, len(progressions)))]
            progression_size = (cell_count - 1 - base) // step + 1
            candidate = base + int(rng.integers(0, progression_size)) * step
            if candidate not in used:
                used.add(candidate)
                row.append(candidate)
        _append_from_ranges(
            row,
            used,
            ((0, cell_count),),
            fanout - len(row),
            rng,
        )
        targets[source] = row
    return targets


def _uniform_no_self_vectorized(
    cell_count: int,
    fanout: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """O(cell_count * fanout) uniform self-excluding targets for large populations.

    Draws ``fanout`` i.i.d. candidates per source row and redraws whole rows
    whose candidate set contains a duplicate value until every row holds
    distinct targets. Conditioning i.i.d. rows on distinctness stays uniform,
    matching the legacy per-row sampler's distribution, and convergence is
    geometric because duplicates need fanout**2 << cell_count collisions.
    Rows then apply the identical candidate-to-target map as the legacy path:
    candidate c in [0, cell_count - 2] becomes c + (c >= source), which equals
    sampling from the population with the source neuron removed.
    """
    population = cell_count - 1
    candidates = rng.integers(0, population, size=(cell_count, fanout))
    for _ in range(64):
        ordered = np.sort(candidates, axis=1)
        adjacent_equal = np.zeros(candidates.shape, dtype=bool)
        adjacent_equal[:, 1:] = ordered[:, 1:] == ordered[:, :-1]
        conflicted_rows = np.flatnonzero(adjacent_equal.any(axis=1))
        if conflicted_rows.size == 0:
            sources = np.arange(cell_count, dtype=np.int64)[:, None]
            return (candidates + (candidates >= sources)).astype(
                np.int32, copy=False
            )
        candidates[conflicted_rows] = rng.integers(
            0, population, size=(int(conflicted_rows.size), fanout)
        )
    # Unreachable in practice (each round succeeds with overwhelming
    # probability); fall back to the exact legacy sampler rather than risk an
    # infinite loop.
    return build_recurrent_targets_legacy_uniform_no_self(cell_count, fanout, rng)


def build_recurrent_targets_legacy_uniform_no_self(
    cell_count: int,
    fanout: int,
    rng: np.random.Generator,
) -> np.ndarray:
    targets = np.empty((cell_count, fanout), dtype=np.int32)
    for source in range(cell_count):
        # Sampling candidate indices is bit-for-bit identical to sampling
        # np.delete(arange(cell_count), source), without allocating an
        # almost-cell-count temporary for every source neuron.
        row = rng.choice(cell_count - 1, fanout, replace=False).astype(
            np.int32,
            copy=False,
        )
        row += row >= source
        targets[source] = row
    return targets


def build_recurrent_targets(
    cell_count: int,
    fanout: int,
    rng: np.random.Generator,
    topology: str,
) -> np.ndarray:
    """Build a degree-matched recurrent graph for one preregistered topology."""
    if topology not in RECURRENT_TOPOLOGIES:
        raise ValueError(f"unknown recurrent topology: {topology}")
    if not 1 <= fanout <= cell_count:
        raise ValueError("fanout must fit the cell population")
    if topology == "random":
        targets = np.empty((cell_count, fanout), dtype=np.int32)
        for source in range(cell_count):
            targets[source] = rng.choice(cell_count, fanout, replace=False)
        return targets
    if fanout >= cell_count:
        raise ValueError("non-random topology requires room to exclude self-connections")
    if topology == "uniform_no_self":
        if cell_count >= _UNIFORM_NO_SELF_VECTOR_MIN_CELLS:
            return _uniform_no_self_vectorized(cell_count, fanout, rng)
        return build_recurrent_targets_legacy_uniform_no_self(
            cell_count,
            fanout,
            rng,
        )
    if topology == "single_scale_groups":
        return _group_targets(cell_count, fanout, rng, nested=False)
    if topology == "nested_groups":
        return _group_targets(cell_count, fanout, rng, nested=True)
    if topology == "mixed_membership":
        return _mixed_membership_targets(cell_count, fanout, rng)
    return _spatial_targets(cell_count, fanout, rng, topology)
