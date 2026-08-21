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
