from __future__ import annotations

import pytest

from experiments.r3_pilot import python_entry_return_provenance_v1 as provenance
from experiments.r3_pilot.common_repository_source_views_v1 import (
    project_python_repository_lexical_view_v1,
)
from experiments.r3_pilot.common_rooted_structured_event_graph_v1 import (
    MAX_CARRIERS,
    MAX_EVENTS,
    MAX_INCIDENCES,
    project_rooted_structured_event_graph_v1,
)


def _observe(source: bytes, entry: str) -> provenance.PythonEntryReturnProvenanceV1:
    result = provenance.observe_python_entry_return_provenance_v1(
        source, entry_point=entry
    )
    result.authenticate()
    return result


def _legacy_factor_roots(source: bytes) -> tuple[str, ...]:
    view = project_python_repository_lexical_view_v1(source)
    view.authenticate()
    return tuple(factor.semantic_sha256 for factor in view.factors)


def _boundary_edge(
    result: provenance.PythonEntryReturnProvenanceV1,
) -> tuple[int, int, int]:
    boundary_index = result.graph.carrier_count - 1
    edges = tuple(
        edge
        for edge in result.graph.incidences
        if edge.target_index == boundary_index
    )
    assert len(edges) == 1
    return (edges[0].source_index, edges[0].target_index, edges[0].ordinal)


def _selected_event_roots(
    result: provenance.PythonEntryReturnProvenanceV1,
) -> tuple[str, ...]:
    carrier_count = result.graph.carrier_count
    selected_carrier = _boundary_edge(result)[0]
    edges = sorted(
        (
            edge
            for edge in result.graph.incidences
            if edge.target_index == selected_carrier
            and edge.source_index >= carrier_count
        ),
        key=lambda edge: edge.ordinal,
    )
    return tuple(
        result.graph.event_roots[edge.source_index - carrier_count].semantic_sha256
        for edge in edges
    )


def _non_boundary_incidence_keys(
    result: provenance.PythonEntryReturnProvenanceV1,
) -> tuple[tuple[int, int, int], ...]:
    boundary_index = result.graph.carrier_count - 1
    return tuple(
        (edge.source_index, edge.target_index, edge.ordinal)
        for edge in result.graph.incidences
        if edge.target_index != boundary_index
    )


def _inventory_roots(
    result: provenance.PythonEntryReturnProvenanceV1,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(root.semantic_sha256 for root in result.graph.event_roots),
        tuple(root.semantic_sha256 for root in result.graph.carrier_roots),
    )


def test_return_binding_moves_only_boundary_over_complete_effect_inventory() -> None:
    reverse_first = b"""\
def kappa_17(arg_q7):
 tmp_a3=arg_q7[::-1]
 tmp_b8=arg_q7[:]
 return tmp_a3
"""
    identity_first = reverse_first.replace(b"return tmp_a3", b"return tmp_b8")
    reverse_second = b"""\
def kappa_17(arg_q7):
 tmp_b8=arg_q7[:]
 tmp_a3=arg_q7[::-1]
 return tmp_a3
"""
    identity_second = reverse_second.replace(b"return tmp_a3", b"return tmp_b8")
    fresh_reverse = b"""\
def omega_92(item_r4):
 out_n6=item_r4[::-1]
 keep_p2=item_r4[:]
 return out_n6
"""
    fresh_identity = fresh_reverse.replace(b"return out_n6", b"return keep_p2")
    join_reverse = b"""\
def theta_51(seq_v9):
 out_c4=''.join(reversed(seq_v9))
 keep_h2=seq_v9[:]
 return out_c4
"""

    pairs = (
        (reverse_first, identity_first, "kappa_17"),
        (reverse_second, identity_second, "kappa_17"),
        (fresh_reverse, fresh_identity, "omega_92"),
    )
    results = []
    for reverse_source, identity_source, entry in pairs:
        assert _legacy_factor_roots(reverse_source) == _legacy_factor_roots(
            identity_source
        )
        reverse_result = _observe(reverse_source, entry)
        identity_result = _observe(identity_source, entry)
        results.append((reverse_result, identity_result))
        assert _inventory_roots(reverse_result) == _inventory_roots(identity_result)
        assert _non_boundary_incidence_keys(
            reverse_result
        ) == _non_boundary_incidence_keys(identity_result)
        assert _boundary_edge(reverse_result) != _boundary_edge(identity_result)
        assert reverse_result.graph.event_count == 4
        assert reverse_result.graph.carrier_count == 5
        assert reverse_result.graph.boundary_carrier_indices == (4,)
        assert reverse_result.graph.incidence_count == 9
        assert reverse_result.graph.event_count <= MAX_EVENTS
        assert reverse_result.graph.carrier_count <= MAX_CARRIERS
        assert reverse_result.graph.incidence_count <= MAX_INCIDENCES
        assert len(set(_inventory_roots(reverse_result)[0])) == 4
        assert tuple(
            edge.ordinal
            for edge in reverse_result.graph.incidences
            if edge.target_index == _boundary_edge(reverse_result)[0]
            and edge.source_index >= reverse_result.graph.carrier_count
        ) == (0, 1)

    reverse_first_result, identity_first_result = results[0]
    reverse_second_result, identity_second_result = results[1]
    fresh_reverse_result, fresh_identity_result = results[2]
    assert _boundary_edge(reverse_first_result) == _boundary_edge(
        identity_second_result
    )
    assert _boundary_edge(identity_first_result) == _boundary_edge(
        reverse_second_result
    )
    join_result = _observe(join_reverse, "theta_51")
    assert _selected_event_roots(reverse_first_result) == _selected_event_roots(
        reverse_second_result
    )
    assert _selected_event_roots(reverse_first_result) == _selected_event_roots(
        fresh_reverse_result
    )
    assert _selected_event_roots(reverse_first_result) == _selected_event_roots(
        join_result
    )
    assert _selected_event_roots(identity_first_result) == _selected_event_roots(
        identity_second_result
    )
    assert _selected_event_roots(identity_first_result) == _selected_event_roots(
        fresh_identity_result
    )
    assert _selected_event_roots(reverse_first_result) != _selected_event_roots(
        identity_first_result
    )
    capture_named = _observe(
        b"def entry(__capture):\n out=__capture[::-1]\n keep=__capture[:]\n return out\n",
        "entry",
    )
    assert _selected_event_roots(capture_named) == _selected_event_roots(
        reverse_first_result
    )


def test_decoys_abstention_and_mixed_runtime_effects_are_not_host_labeled() -> None:
    identity = _observe(
        b"def entry(x):\n keep=x[:]\n return keep\n", "entry"
    )
    reverse = _observe(
        b"def entry(x):\n keep=x[::-1]\n return keep\n", "entry"
    )
    called_discarded = _observe(
        b"def entry(x):\n reversed(x)\n keep=x[:]\n return keep\n", "entry"
    )
    dead = _observe(
        b"def entry(x):\n keep=x[:] if True else x[::-1]\n return keep\n",
        "entry",
    )
    string_comment = _observe(
        b"def entry(x):\n 'x[::-1]'\n keep=x[:] # x[::-1]\n return keep\n",
        "entry",
    )
    non_bijection = _observe(
        b"def entry(x):\n bad=x[1:]\n decoy=x[::-1]\n return bad\n", "entry"
    )
    mixed = _observe(
        b"def entry(x):\n out=x[::-1] if len(x)%2 else x[:]\n return out\n",
        "entry",
    )

    identity_roots = _selected_event_roots(identity)
    reverse_roots = _selected_event_roots(reverse)
    assert _selected_event_roots(called_discarded) == identity_roots
    assert _selected_event_roots(dead) == identity_roots
    assert _selected_event_roots(string_comment) == identity_roots
    assert called_discarded.graph.carrier_count == 5
    assert string_comment.graph.carrier_count == 5
    non_bijection_roots = _selected_event_roots(non_bijection)
    assert len(non_bijection_roots) == 2
    assert non_bijection_roots not in (identity_roots, reverse_roots)
    assert non_bijection.graph.event_count == 4
    assert tuple(
        root.semantic_sha256 for root in non_bijection.graph.event_roots
    ) == non_bijection_roots + reverse_roots
    assert _selected_event_roots(mixed) == (reverse_roots[0], identity_roots[1])
    assert not hasattr(mixed, "status")
    assert not hasattr(mixed, "conflict")
    assert not hasattr(mixed.graph, "relevance")


def test_preflights_authentication_and_payload_erasure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_subprocess(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("unsafe or oversized source reached execution")

    monkeypatch.setattr(provenance.subprocess, "run", forbidden_subprocess)
    with pytest.raises(provenance.PythonEntryReturnProvenanceCapacityError):
        provenance.observe_python_entry_return_provenance_v1(
            b"x" * 257, entry_point="entry"
        )
    with pytest.raises(provenance.PythonEntryReturnProvenanceSafetyError):
        provenance.observe_python_entry_return_provenance_v1(
            b"import os\ndef entry(x):\n out=x[:]\n return out\n",
            entry_point="entry",
        )
    with pytest.raises(provenance.PythonEntryReturnProvenanceCapacityError):
        provenance.observe_python_entry_return_provenance_v1(
            b"def entry(x):\n a=x[:]\n b=x[:]\n c=x[:]\n d=x[:]\n e=x[:]\n return a\n",
            entry_point="entry",
        )
    with pytest.raises(provenance.PythonEntryReturnProvenanceSafetyError):
        provenance.observe_python_entry_return_provenance_v1(
            b"def entry(x):\n while True: pass\n return x\n",
            entry_point="entry",
        )
    with pytest.raises(provenance.PythonEntryReturnProvenanceSafetyError):
        provenance.observe_python_entry_return_provenance_v1(
            b"def entry(len):\n out=len[:]\n return out\n",
            entry_point="entry",
        )

    monkeypatch.undo()
    source = b"def private_entry(private_arg):\n secret=private_arg[::-1]\n return secret\n"
    result = _observe(source, "private_entry")
    rendered = repr(result)
    assert source.decode("utf-8") not in rendered
    assert "private_entry" not in rendered
    assert "private_arg" not in rendered
    assert "secret" not in rendered
    assert "[::-1]" not in rendered
    assert not hasattr(result, "source")
    assert not hasattr(result, "entry_point")
    assert not hasattr(result.graph, "payloads")
    assert all(not hasattr(root, "payload") for root in result.graph.event_roots)
    assert all(not hasattr(root, "payload") for root in result.graph.carrier_roots)
    assert all(len(root.trace_cells) == 12 for root in result.graph.event_roots)
    assert all(len(root.trace_cells) == 12 for root in result.graph.carrier_roots)
    assert result.graph.projection_cell_steps > 0
    result.authenticate()

    forged_graph = project_rooted_structured_event_graph_v1(
        (b"e" * 16, b"f" * 16),
        carrier_payloads=tuple(bytes((index,)) * 16 for index in range(4)),
        root_carrier_indices=(0, 1),
        boundary_carrier_indices=(3,),
        incidences=(
            (0, 3, 0),
            (0, 4, 0),
            (1, 5, 1),
            (2, 3, 0),
            (4, 2, 0),
            (5, 2, 1),
        ),
    )
    forged_source_root = "00" * 32
    forged_boundary_payload = {
        "source": forged_source_root,
        "schema": provenance._SCHEMA_SHA256,
        "graph": forged_graph.sha256,
        "carrier": forged_graph.carrier_roots[3].sha256,
    }
    forged_boundary_root = provenance._digest(
        provenance._BOUNDARY_NAMESPACE, forged_boundary_payload
    )
    forged_payload = {
        "source": forged_source_root,
        "boundary": forged_boundary_root,
        "schema": provenance._SCHEMA_SHA256,
        "graph": forged_graph.sha256,
    }
    forged = provenance.PythonEntryReturnProvenanceV1(
        source_sha256=forged_source_root,
        boundary_sha256=forged_boundary_root,
        schema_sha256=provenance._SCHEMA_SHA256,
        graph=forged_graph,
        sha256=provenance._digest(provenance._RESULT_NAMESPACE, forged_payload),
    )
    with pytest.raises(provenance.PythonEntryReturnProvenanceError):
        forged.authenticate()
