import numpy as np
from humanbrain.topology import (
    RECURRENT_TOPOLOGIES,
    build_recurrent_targets,
    topology_coordinates,
)


def test_every_topology_is_deterministic_and_degree_matched() -> None:
    for topology in sorted(RECURRENT_TOPOLOGIES):
        first = build_recurrent_targets(48, 6, np.random.default_rng(19), topology)
        second = build_recurrent_targets(48, 6, np.random.default_rng(19), topology)

        assert np.array_equal(first, second)
        assert first.shape == (48, 6)
        assert np.all((first >= 0) & (first < 48))
        assert all(len(set(row)) == 6 for row in first.tolist())


def test_structured_topologies_exclude_self_connections() -> None:
    for topology in RECURRENT_TOPOLOGIES - {"random"}:
        targets = build_recurrent_targets(48, 6, np.random.default_rng(23), topology)

        assert not any(source in row for source, row in enumerate(targets.tolist()))


def test_uniform_sampling_scales_without_population_sized_candidate_rows() -> None:
    targets = build_recurrent_targets(
        100_003,
        6,
        np.random.default_rng(41),
        "uniform_no_self",
    )

    assert targets.shape == (100_003, 6)
    assert targets.dtype == np.int32
    assert np.all(targets >= 0)
    assert np.all(targets < 100_003)
    assert not np.any(targets == np.arange(100_003, dtype=np.int32)[:, None])
    assert np.all(np.diff(np.sort(targets, axis=1), axis=1) != 0)


def test_uniform_sampling_preserves_legacy_seeded_targets() -> None:
    cells = 48
    fanout = 6
    seed = 19
    expected = np.empty((cells, fanout), dtype=np.int32)
    rng = np.random.default_rng(seed)
    candidates = np.arange(cells, dtype=np.int32)
    for source in range(cells):
        expected[source] = rng.choice(
            np.delete(candidates, source),
            fanout,
            replace=False,
        )

    actual = build_recurrent_targets(
        cells,
        fanout,
        np.random.default_rng(seed),
        "uniform_no_self",
    )

    np.testing.assert_array_equal(actual, expected)


def test_group_samplers_enforce_direct_micro_meso_and_membership_quotas() -> None:
    cells = 4_096
    fanout = 6
    micro_size = round(cells**0.5)
    sources = np.arange(cells, dtype=np.int64)

    single = build_recurrent_targets(
        cells,
        fanout,
        np.random.default_rng(51),
        "single_scale_groups",
    )
    assert np.all(single[:, :5] // micro_size == sources[:, None] // micro_size)

    nested = build_recurrent_targets(
        cells,
        fanout,
        np.random.default_rng(53),
        "nested_groups",
    )
    assert np.all(nested[:, :3] // micro_size == sources[:, None] // micro_size)
    assert np.all(
        nested[:, :5] // (micro_size * 4) == sources[:, None] // (micro_size * 4)
    )

    mixed = build_recurrent_targets(
        cells,
        fanout,
        np.random.default_rng(55),
        "mixed_membership",
    )
    group_count = round(cells**0.5)
    while group_count % 5 == 0:
        group_count += 1
    source_primary = sources % group_count
    source_secondary = (sources * 5 + 1) % group_count
    local = mixed[:, :5]
    target_primary = local % group_count
    target_secondary = (local * 5 + 1) % group_count
    shared = (
        (target_primary == source_primary[:, None])
        | (target_primary == source_secondary[:, None])
        | (target_secondary == source_primary[:, None])
        | (target_secondary == source_secondary[:, None])
    )
    assert np.all(shared)


def test_spatial_sampler_places_local_offsets_before_sparse_long_links() -> None:
    cells = 4_096
    fanout = 6
    sources = np.arange(cells, dtype=np.int64)
    ring = build_recurrent_targets(
        cells,
        fanout,
        np.random.default_rng(57),
        "ring_1d",
    )
    local_separation = np.abs(ring[:, :5] - sources[:, None])
    local_separation = np.minimum(local_separation, cells - local_separation)
    long_separation = np.abs(ring[:, 5] - sources)
    long_separation = np.minimum(long_separation, cells - long_separation)

    assert np.all(local_separation <= round(cells**0.25))
    assert np.all(long_separation >= cells // 4)


def test_hypercube_4d_is_an_abstract_local_lattice_with_sparse_long_links() -> None:
    cells = 4_096
    fanout = 6
    side = 8
    sources = np.arange(cells, dtype=np.int64)
    targets = build_recurrent_targets(
        cells,
        fanout,
        np.random.default_rng(58),
        "hypercube_4d",
    )

    source_coordinates = np.column_stack(
        [(sources // side**dimension) % side for dimension in range(4)]
    )
    local_coordinates = np.stack(
        [
            np.column_stack(
                [
                    (targets[:, edge] // side**dimension) % side
                    for dimension in range(4)
                ]
            )
            for edge in range(5)
        ],
        axis=1,
    )
    long_coordinates = np.column_stack(
        [(targets[:, 5] // side**dimension) % side for dimension in range(4)]
    )

    assert topology_coordinates(257, "hypercube_4d").shape == (257, 4)
    assert np.all(np.abs(local_coordinates - source_coordinates[:, None, :]) <= 2)
    squared_long_distance = np.square(
        long_coordinates - source_coordinates
    ).sum(axis=1)
    assert np.all(squared_long_distance >= max(2, side // 4) ** 2)


def test_structured_sampler_has_linear_output_memory_at_100k_cells() -> None:
    cells = 100_003
    fanout = 6
    for topology in ("nested_groups", "hypercube_4d"):
        targets = build_recurrent_targets(
            cells,
            fanout,
            np.random.default_rng(59),
            topology,
        )

        assert targets.nbytes == cells * fanout * np.dtype(np.int32).itemsize
        assert not np.any(targets == np.arange(cells, dtype=np.int32)[:, None])
        assert np.all(np.diff(np.sort(targets, axis=1), axis=1) != 0)
