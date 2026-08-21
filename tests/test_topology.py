import numpy as np
from humanbrain.topology import RECURRENT_TOPOLOGIES, build_recurrent_targets


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
