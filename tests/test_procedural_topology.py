from dataclasses import FrozenInstanceError
import gc
import tracemalloc

import pytest

from humanbrain.procedural_topology import (
    ProceduralTopology,
    TOPOLOGY_PROTOCOL,
    TopologyManifest,
    _build_edges_with_stats,
    topology_implementation_contract,
    verify_topology_implementation,
)
from humanbrain.human_cell_identity import HUMAN_GLIF5_IDENTITY


TOPOLOGIES = (
    "random",
    "uniform_no_self",
    "single_scale_groups",
    "nested_groups",
    "mixed_membership",
    "ring_1d",
    "sheet_2d",
    "layered_sheet",
    "volume_3d",
    "hypercube_4d",
)


def test_philox2x64_10_matches_random123_known_answers() -> None:
    result = verify_topology_implementation()

    assert result.philox_known_answers == (
        {
            "counter_hex": ("0000000000000000", "0000000000000000"),
            "key_hex": "0000000000000000",
            "expected_hex": ("ca00a0459843d731", "66c24222c9a845b5"),
            "actual_hex": ("ca00a0459843d731", "66c24222c9a845b5"),
            "passed": True,
        },
        {
            "counter_hex": ("243f6a8885a308d3", "13198a2e03707344"),
            "key_hex": "a4093822299f31d0",
            "expected_hex": ("0a5e742c2997341c", "b0f883d38000de5d"),
            "actual_hex": ("0a5e742c2997341c", "b0f883d38000de5d"),
            "passed": True,
        },
    )


def test_manifest_is_an_immutable_uint64_address_range() -> None:
    manifest = TopologyManifest(
        cell_count=100_000_000_000,
        fanout=6,
        topology="volume_3d",
        seed=71,
    )

    assert manifest.protocol == TOPOLOGY_PROTOCOL
    assert manifest.gid_range == (0, 100_000_000_000)
    assert manifest.gid_width_bits == 64
    with pytest.raises(FrozenInstanceError):
        manifest.cell_count = 8  # type: ignore[misc]

    boundary = TopologyManifest(
        cell_count=1 << 64,
        fanout=1,
        topology="uniform_no_self",
        seed=(1 << 64) - 1,
    )
    assert boundary.gid_range == (0, 1 << 64)

    with pytest.raises(ValueError, match="unsupported topology protocol"):
        TopologyManifest(
            17,
            4,
            "random",
            71,
            protocol="random123-philox2x64-10-v1",
        )
    oracle = ProceduralTopology(manifest)
    with pytest.raises(AttributeError):
        oracle.manifest = boundary  # type: ignore[misc]


def test_every_topology_returns_a_truthful_pinned_human_cell_descriptor() -> None:
    descriptors = {
        topology: ProceduralTopology(
            TopologyManifest(4_096, 6, topology, 71)
        ).describe_cell(100)
        for topology in TOPOLOGIES
    }

    for descriptor in descriptors.values():
        assert descriptor.gid == 100
        assert descriptor.role == "reservoir"
        assert descriptor.cell_type == "adaptive_glif5"
        assert descriptor.outgoing_connectivity_role in {"excitatory", "inhibitory"}
        assert descriptor.intrinsic_model == HUMAN_GLIF5_IDENTITY
    assert len({descriptor.intrinsic_model for descriptor in descriptors.values()}) == 1

    assert descriptors["random"].coordinates is None
    assert descriptors["uniform_no_self"].groups == ()
    assert descriptors["single_scale_groups"].groups == (("micro", 1),)
    assert descriptors["nested_groups"].groups == (("micro", 1), ("meso", 0))
    assert descriptors["mixed_membership"].groups == (
        ("primary", 36),
        ("secondary", 38),
    )
    assert descriptors["ring_1d"].coordinates == (100,)
    assert descriptors["sheet_2d"].coordinates == (36, 1)
    assert descriptors["layered_sheet"].coordinates == (16, 0, 4)
    assert descriptors["layered_sheet"].layer == 4
    assert descriptors["volume_3d"].coordinates == (4, 6, 0)
    assert descriptors["hypercube_4d"].coordinates == (4, 4, 1, 0)
    assert descriptors["hypercube_4d"].coordinate_system == (
        "abstract_hypercube_4d_integer_lattice"
    )


def test_mixed_membership_is_distinct_for_every_gid_in_exhaustive_small_domain() -> None:
    for cell_count in range(2, 513):
        oracle = ProceduralTopology(
            TopologyManifest(
                cell_count,
                min(6, cell_count - 1),
                "mixed_membership",
                71,
            )
        )
        for gid in range(cell_count):
            groups = dict(oracle.describe_cell(gid).groups)
            assert groups["primary"] != groups["secondary"], (cell_count, gid)


def test_mixed_membership_edges_use_the_same_groups_as_descriptors() -> None:
    for cell_count in range(8, 129):
        oracle = ProceduralTopology(
            TopologyManifest(cell_count, 2, "mixed_membership", 71)
        )
        for gid in range(cell_count):
            source_groups = frozenset(dict(oracle.describe_cell(gid).groups).values())
            shared_target = oracle.base_edges(gid)[0].target_gid
            target_primary = dict(oracle.describe_cell(shared_target).groups)["primary"]
            assert target_primary in source_groups, (cell_count, gid, shared_target)


@pytest.mark.parametrize(
    "cell_count",
    [1_000_000, 1_000_000_000, 100_000_000_000, 1 << 64],
)
def test_mixed_membership_is_distinct_at_large_address_boundaries(
    cell_count: int,
) -> None:
    oracle = ProceduralTopology(
        TopologyManifest(cell_count, 6, "mixed_membership", 71)
    )
    for gid in (0, 1, (1 << 32) - 1, 1 << 32, cell_count - 2, cell_count - 1):
        if gid >= cell_count:
            continue
        groups = dict(oracle.describe_cell(gid).groups)
        assert groups["primary"] != groups["secondary"]


def test_every_topology_has_order_independent_uint64_base_edge_queries() -> None:
    gids = (0, (1 << 32) - 1, 1 << 32, (1 << 32) + 1, 99_999_999_999)
    for topology in TOPOLOGIES:
        manifest = TopologyManifest(100_000_000_000, 6, topology, 71)
        first = ProceduralTopology(manifest)
        forward = {gid: first.base_edges(gid) for gid in gids}
        reverse = {
            gid: ProceduralTopology(manifest).base_edges(gid)
            for gid in reversed(gids)
        }

        assert forward == reverse
        for source, edges in forward.items():
            assert tuple(edge.edge_slot for edge in edges) == tuple(range(6))
            targets = tuple(edge.target_gid for edge in edges)
            assert len(targets) == 6
            assert len(set(targets)) == 6
            assert source not in targets
            assert all(0 <= target < manifest.cell_count for target in targets)


def test_group_topologies_preserve_their_fixed_slot_quotas() -> None:
    for source in (0, 63, 64, 100, 4_095):
        single = ProceduralTopology(
            TopologyManifest(4_096, 6, "single_scale_groups", 71)
        )
        source_micro = single.describe_cell(source).groups[0][1]
        assert all(
            single.describe_cell(edge.target_gid).groups[0][1] == source_micro
            for edge in single.base_edges(source)[:5]
        )

        nested = ProceduralTopology(
            TopologyManifest(4_096, 6, "nested_groups", 71)
        )
        source_groups = dict(nested.describe_cell(source).groups)
        targets = [nested.describe_cell(edge.target_gid) for edge in nested.base_edges(source)]
        assert all(dict(target.groups)["micro"] == source_groups["micro"] for target in targets[:3])
        assert all(dict(target.groups)["meso"] == source_groups["meso"] for target in targets[3:5])
        assert all(
            dict(target.groups)["micro"] != source_groups["micro"]
            for target in targets[3:5]
        )

        mixed = ProceduralTopology(
            TopologyManifest(4_096, 6, "mixed_membership", 71)
        )
        source_memberships = {value for _, value in mixed.describe_cell(source).groups}
        assert all(
            dict(mixed.describe_cell(edge.target_gid).groups)["primary"]
            in source_memberships
            for edge in mixed.base_edges(source)[:5]
        )


def test_spatial_topologies_reserve_local_slots_and_a_long_range_slot() -> None:
    cells = 4_096
    source = 1_755
    cases = {
        "sheet_2d": (2, (2, 2)),
        "layered_sheet": (3, (2, 2, 1)),
        "volume_3d": (3, (2, 2, 2)),
        "hypercube_4d": (4, (2, 2, 2, 2)),
    }
    for topology, (dimensions, radii) in cases.items():
        oracle = ProceduralTopology(TopologyManifest(cells, 6, topology, 71))
        source_coordinates = oracle.describe_cell(source).coordinates
        assert source_coordinates is not None
        edges = oracle.base_edges(source)
        for edge in edges[:5]:
            coordinates = oracle.describe_cell(edge.target_gid).coordinates
            assert coordinates is not None
            assert len(coordinates) == dimensions
            assert all(
                abs(left - right) <= radius
                for left, right, radius in zip(
                    source_coordinates, coordinates, radii, strict=True
                )
            )
        long_coordinates = oracle.describe_cell(edges[5].target_gid).coordinates
        assert long_coordinates is not None
        distance_dimensions = 2 if topology == "layered_sheet" else dimensions
        squared_distance = sum(
            (source_coordinates[index] - long_coordinates[index]) ** 2
            for index in range(distance_dimensions)
        )
        side = 27 if topology == "layered_sheet" else round(cells ** (1 / dimensions))
        assert squared_distance >= max(2, side // 4) ** 2

    ring = ProceduralTopology(TopologyManifest(cells, 6, "ring_1d", 71))
    ring_targets = [edge.target_gid for edge in ring.base_edges(source)]
    for target in ring_targets[:5]:
        separation = abs(source - target)
        assert min(separation, cells - separation) <= 8
    separation = abs(source - ring_targets[5])
    assert min(separation, cells - separation) >= cells // 4


@pytest.mark.parametrize(
    ("kwargs", "error"),
    (
        ({"cell_count": True}, TypeError),
        ({"cell_count": 1}, ValueError),
        ({"cell_count": (1 << 64) + 1}, ValueError),
        ({"fanout": False}, TypeError),
        ({"fanout": 8}, ValueError),
        ({"topology": "unknown"}, ValueError),
        ({"seed": -1}, ValueError),
        ({"seed": 1 << 64}, ValueError),
        ({"protocol": "legacy-numpy"}, ValueError),
    ),
)
def test_manifest_rejects_values_outside_the_frozen_protocol(
    kwargs: dict[str, object], error: type[Exception]
) -> None:
    values: dict[str, object] = {
        "cell_count": 8,
        "fanout": 6,
        "topology": "random",
        "seed": 71,
    }
    values.update(kwargs)
    with pytest.raises(error):
        TopologyManifest(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("gid", (-1, 8, 1 << 64))
def test_cell_and_edge_queries_reject_out_of_range_gids(gid: int) -> None:
    oracle = ProceduralTopology(TopologyManifest(8, 6, "random", 71))

    with pytest.raises(ValueError, match="gid"):
        oracle.describe_cell(gid)
    with pytest.raises(ValueError, match="gid"):
        oracle.base_edges(gid)


@pytest.mark.parametrize("gid", (True, 1.0, "1", None))
def test_cell_and_edge_queries_require_exact_integer_gids(gid: object) -> None:
    oracle = ProceduralTopology(TopologyManifest(8, 6, "random", 71))

    with pytest.raises(TypeError, match="gid"):
        oracle.describe_cell(gid)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="gid"):
        oracle.base_edges(gid)  # type: ignore[arg-type]


def test_manifest_construction_and_one_query_have_bounded_traced_memory() -> None:
    def peak_bytes(cell_count: int) -> int:
        gc.collect()
        tracemalloc.start()
        oracle = ProceduralTopology(
            TopologyManifest(cell_count, 6, "hypercube_4d", 71)
        )
        descriptor = oracle.describe_cell(cell_count - 1)
        edges = oracle.base_edges(cell_count - 1)
        assert descriptor.gid == cell_count - 1
        assert len(edges) == 6
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        return peak

    million_peak = peak_bytes(1_000_000)
    hundred_billion_peak = peak_bytes(100_000_000_000)

    assert million_peak < 64 * 1024
    assert hundred_billion_peak < 64 * 1024
    assert abs(hundred_billion_peak - million_peak) < 8 * 1024


def test_all_small_rows_match_the_frozen_counter64_v2_materialized_reference() -> None:
    result = verify_topology_implementation()

    assert result.materialized_reference == {
        "cell_count": 17,
        "fanout": 4,
        "seed": 71,
        "byte_order": "little",
        "sha256": "ba55ded147218d524bf4d004d97a97f42d3471c3079ad110fa4a0451f4f92b60",
        "passed": True,
    }


def test_topology_implementation_contract_is_a_complete_v2_record() -> None:
    contract = topology_implementation_contract()

    assert contract["topology_protocol"] == "random123-philox2x64-10-v2"
    assert contract["topology_names"] == list(TOPOLOGIES)
    assert contract["human_cell_identity"] == HUMAN_GLIF5_IDENTITY.to_record()
    assert contract["mixed_membership_rule"] == (
        "primary-plus-positive-quotient-offset-v2"
    )


@pytest.mark.parametrize("fanout", (1, 8, 64, 257))
def test_forced_collisions_have_linear_candidate_work(fanout: int) -> None:
    oracle = ProceduralTopology(
        TopologyManifest(fanout + 2, fanout, "uniform_no_self", 71)
    )

    def always_collides(
        _: ProceduralTopology, source: int, __: int, ___: int
    ) -> int:
        return source

    edges, candidate_examinations = _build_edges_with_stats(
        oracle,
        0,
        always_collides,
    )

    assert len(edges) == fanout
    assert len({edge.target_gid for edge in edges}) == fanout
    assert candidate_examinations >= 128 * fanout
    assert candidate_examinations <= 129 * fanout + 1
