from __future__ import annotations

from math import ceil

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
        "mixed_membership",
    }
)


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
    coordinates = topology_coordinates(cell_count, topology)
    targets = np.empty((cell_count, fanout), dtype=np.int32)
    local_edges = fanout - max(1, round(0.15 * fanout))
    for source in range(cell_count):
        candidates = np.delete(np.arange(cell_count, dtype=np.int32), source)
        differences = coordinates[candidates] - coordinates[source]
        if topology == "ring_1d":
            differences = np.minimum(np.abs(differences), 1.0 - np.abs(differences))
        distances = np.linalg.norm(differences, axis=1)
        scale = float(np.median(distances[distances > 0.0]))
        local_weights = np.exp(-distances / scale)
        local = rng.choice(
            candidates,
            size=local_edges,
            replace=False,
            p=local_weights / local_weights.sum(),
        )
        remaining = candidates[~np.isin(candidates, local)]
        remaining_differences = coordinates[remaining] - coordinates[source]
        if topology == "ring_1d":
            remaining_differences = np.minimum(
                np.abs(remaining_differences), 1.0 - np.abs(remaining_differences)
            )
        remaining_distances = np.linalg.norm(remaining_differences, axis=1)
        long_range = rng.choice(
            remaining,
            size=fanout - local_edges,
            replace=False,
            p=remaining_distances / remaining_distances.sum(),
        )
        targets[source] = np.concatenate((local, long_range))
    return targets


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
        candidates = np.delete(np.arange(cell_count, dtype=np.int32), source)
        same_micro = microgroups[candidates] == microgroups[source]
        if nested:
            same_meso = mesogroups[candidates] == mesogroups[source]
            weights = np.where(same_micro, 16.0, np.where(same_meso, 4.0, 1.0))
        else:
            weights = np.where(same_micro, 8.0, 1.0)
        targets[source] = rng.choice(
            candidates, size=fanout, replace=False, p=weights / weights.sum()
        )
    return targets


def _mixed_membership_targets(
    cell_count: int, fanout: int, rng: np.random.Generator
) -> np.ndarray:
    group_count = max(4, round(cell_count**0.5))
    primary = np.arange(cell_count) % group_count
    secondary = (np.arange(cell_count) * 5 + 1) % group_count
    targets = np.empty((cell_count, fanout), dtype=np.int32)
    for source in range(cell_count):
        candidates = np.delete(np.arange(cell_count, dtype=np.int32), source)
        shared = (
            (primary[candidates] == primary[source])
            | (primary[candidates] == secondary[source])
            | (secondary[candidates] == primary[source])
            | (secondary[candidates] == secondary[source])
        )
        weights = np.where(shared, 8.0, 1.0)
        targets[source] = rng.choice(
            candidates, size=fanout, replace=False, p=weights / weights.sum()
        )
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
    if topology == "single_scale_groups":
        return _group_targets(cell_count, fanout, rng, nested=False)
    if topology == "nested_groups":
        return _group_targets(cell_count, fanout, rng, nested=True)
    if topology == "mixed_membership":
        return _mixed_membership_targets(cell_count, fanout, rng)
    return _spatial_targets(cell_count, fanout, rng, topology)
