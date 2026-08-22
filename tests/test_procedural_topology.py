from dataclasses import FrozenInstanceError
import gc
from hashlib import sha256
import tracemalloc

import pytest

from humanbrain.procedural_topology import (
    ProceduralTopology,
    TOPOLOGY_PROTOCOL,
    TopologyManifest,
    _build_edges_with_stats,
    _philox2x64_10,
)


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
    assert _philox2x64_10(0, 0, 0) == (
        0xCA00A0459843D731,
        0x66C24222C9A845B5,
    )
    assert _philox2x64_10(
        0x243F6A8885A308D3,
        0x13198A2E03707344,
        0xA4093822299F31D0,
    ) == (0x0A5E742C2997341C, 0xB0F883D38000DE5D)


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
        assert descriptor.intrinsic_model.model_id == 523000114
        assert descriptor.intrinsic_model.specimen_id == 508378649
        assert descriptor.intrinsic_model.species == "Homo sapiens"
        assert descriptor.intrinsic_model.structure == "middle temporal gyrus"
        assert descriptor.intrinsic_model.config_sha256 == (
            "69fe726023e3289ac85857ea9125c55fee1618891e7b65403784c1b961bad149"
        )
    assert len({descriptor.intrinsic_model for descriptor in descriptors.values()}) == 1

    assert descriptors["random"].coordinates is None
    assert descriptors["uniform_no_self"].groups == ()
    assert descriptors["single_scale_groups"].groups == (("micro", 1),)
    assert descriptors["nested_groups"].groups == (("micro", 1), ("meso", 0))
    assert descriptors["mixed_membership"].groups == (
        ("primary", 36),
        ("secondary", 53),
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


def test_all_small_rows_match_the_frozen_counter64_v1_materialized_reference() -> None:
    reference_sha256 = {
        "random": "c7e96790156c726c6d0d7412d80546aabc6ed9d990863e6fd9c1867fc94a151d",
        "uniform_no_self": "0a2f5d4f240459f65b6ce7414614f6e8fbb59bcc2310b367a2703aa8fae0a266",
        "single_scale_groups": "bea4cfc6226a11b89f02b1484aa4dc214eab501ab39c716984d1cf318fb1a073",
        "nested_groups": "0030325506ce3546751b17938fd3e8b06b7caf1fc775f3e07f6e6b81249e2c69",
        "mixed_membership": "a2cf5068be0e017ec98e6a17368831542b058b2f5d491907c7ec8aab9d658b25",
        "ring_1d": "49b241759469282d8c8c60c65cb93f0f9076767aed42fe23c509d596a2fd7ac3",
        "sheet_2d": "708071ffa9165652c10b3c6a0eb14b32fa22ee2b9d230466cf9f5739e0f36097",
        "layered_sheet": "7f6019dc550aedbd0dff1b3ffc8f93669d67b7b041654195c76106695ca3ceca",
        "volume_3d": "4aecf094940249c57b89df0cbf8f08d39f3af0656e8dc897afe4bc3cd5d18c84",
        "hypercube_4d": "c299aff7b52a3b5a068171d9eeea4eee8ad28e0fb076837693dd5b43acfd3b6c",
    }

    for topology, expected_sha256 in reference_sha256.items():
        oracle = ProceduralTopology(TopologyManifest(17, 4, topology, 71))
        materialized = b"".join(
            edge.target_gid.to_bytes(8, "little")
            for gid in range(17)
            for edge in oracle.base_edges(gid)
        )
        assert sha256(materialized).hexdigest() == expected_sha256


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
