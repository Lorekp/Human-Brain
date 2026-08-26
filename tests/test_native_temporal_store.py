"""Native fixed-pair temporal-store behavior at the Python wrapper seam."""

from __future__ import annotations

import copy
import random
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from humanbrain.native_temporal import (
    NativeTemporalStore,
    NativeTemporalStoreConfig,
    NativeTemporalStoreError,
    TemporalOwnerDiagnostics,
    TemporalPrepareDiagnostics,
)
from humanbrain.sparse_temporal_cortex import (
    GrowthPolicy,
    SparseTemporalConfig,
    SparseTemporalCortex,
)


def _dense_reference(
    segments: list[tuple[int, tuple[int, ...], tuple[int, ...]]],
    context: tuple[int, ...],
    *,
    connected_ticks: int,
    theta_match: int,
    theta_active: int,
) -> TemporalPrepareDiagnostics:
    active_sources = set(context)
    potential: dict[int, int] = {}
    connected: dict[int, int] = {}
    for segment_id, (_owner, sources, ticks) in enumerate(segments):
        matched_slots = [
            slot for slot, source in enumerate(sources) if source in active_sources
        ]
        if not matched_slots:
            continue
        potential[segment_id] = 2 * len(matched_slots)
        connected[segment_id] = 2 * sum(
            ticks[slot] >= connected_ticks for slot in matched_slots
        )

    matching = {
        segment_id
        for segment_id, count in potential.items()
        if count >= theta_match
    }
    active = {
        segment_id
        for segment_id, count in connected.items()
        if count >= theta_active
    }
    qualifying_owners = sorted(
        {segments[segment_id][0] for segment_id in matching | active}
    )
    owner_diagnostics: list[TemporalOwnerDiagnostics] = []
    for owner in qualifying_owners:
        owner_matching = [
            segment_id
            for segment_id in matching
            if segments[segment_id][0] == owner
        ]
        owner_active = [
            segment_id
            for segment_id in active
            if segments[segment_id][0] == owner
        ]
        best_matching = max(
            owner_matching,
            key=lambda segment_id: (
                potential[segment_id],
                connected[segment_id],
                -segment_id,
            ),
            default=None,
        )
        best_active = max(
            owner_active,
            key=lambda segment_id: (
                connected[segment_id],
                potential[segment_id],
                -segment_id,
            ),
            default=None,
        )
        owner_diagnostics.append(
            TemporalOwnerDiagnostics(
                owner=owner,
                best_matching_segment=best_matching,
                matching_potential_contacts=(
                    0 if best_matching is None else potential[best_matching]
                ),
                matching_connected_contacts=(
                    0 if best_matching is None else connected[best_matching]
                ),
                best_active_segment=best_active,
                active_potential_contacts=(
                    0 if best_active is None else potential[best_active]
                ),
                active_connected_contacts=(
                    0 if best_active is None else connected[best_active]
                ),
            )
        )
    return TemporalPrepareDiagnostics(
        candidate_segments=len(potential),
        matching_segments=len(matching),
        active_segments=len(active),
        max_potential_contacts=max(potential.values(), default=0),
        max_connected_contacts=max(connected.values(), default=0),
        postings_visited=sum(count // 2 for count in potential.values()),
        owners=tuple(owner_diagnostics),
    )


def test_store_rejects_shallow_and_deep_copy_without_closing_original() -> None:
    config = NativeTemporalStoreConfig(
        source_count=10,
        owner_count=1,
        segment_capacity=1,
        theta_match=18,
        theta_active=20,
    )
    with NativeTemporalStore(config) as store:
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.copy(store)
        with pytest.raises(TypeError, match="cannot be copied"):
            copy.deepcopy(store)

        assert store.allocate_segment(0, tuple(range(10)), (250,) * 10) == 0


def test_close_is_idempotent_and_serializes_the_handle_lifetime() -> None:
    store = NativeTemporalStore(
        NativeTemporalStoreConfig(
            source_count=10,
            owner_count=1,
            segment_capacity=1,
            theta_match=18,
            theta_active=20,
        )
    )

    store.close()
    store.close()
    with pytest.raises(RuntimeError, match="closed"):
        store.prepare(())


def test_close_waits_for_inflight_prepare_before_destroying_handle() -> None:
    store = NativeTemporalStore(
        NativeTemporalStoreConfig(
            source_count=10,
            owner_count=1,
            segment_capacity=1,
            theta_match=18,
            theta_active=20,
        )
    )
    store.allocate_segment(0, tuple(range(10)), (250,) * 10)

    prepare_entered_native = threading.Event()
    release_prepare = threading.Event()
    close_reached_lock = threading.Event()
    destroy_called = threading.Event()
    close_thread_id: list[int] = []

    original_library = store._library
    original_lock = store._operation_lock

    class BlockingLibrary:
        def __getattr__(self, name: str) -> object:
            return getattr(original_library, name)

        def humanbrain_temporal_store_prepare(self, *args: object) -> int:
            prepare_entered_native.set()
            if not release_prepare.wait(timeout=5.0):
                raise AssertionError("test did not release the blocked native prepare")
            return int(original_library.humanbrain_temporal_store_prepare(*args))

        def humanbrain_temporal_store_destroy(self, handle: int) -> None:
            destroy_called.set()
            original_library.humanbrain_temporal_store_destroy(handle)

    class ObservedOperationLock:
        def __enter__(self) -> ObservedOperationLock:
            if close_thread_id and threading.get_ident() == close_thread_id[0]:
                close_reached_lock.set()
            original_lock.acquire()
            return self

        def __exit__(self, *_: object) -> None:
            original_lock.release()

    store._library = BlockingLibrary()
    store._operation_lock = ObservedOperationLock()

    def close_store() -> None:
        close_thread_id.append(threading.get_ident())
        store.close()

    with ThreadPoolExecutor(max_workers=2) as executor:
        prepare_future = executor.submit(store.prepare, tuple(range(10)))
        assert prepare_entered_native.wait(timeout=5.0)
        close_future = executor.submit(close_store)
        try:
            assert close_reached_lock.wait(timeout=5.0)
            assert not destroy_called.is_set()
            assert not close_future.done()
        finally:
            release_prepare.set()

        diagnostics = prepare_future.result(timeout=5.0)
        close_future.result(timeout=5.0)

    assert diagnostics.active_segments == 1
    assert destroy_called.is_set()
    with pytest.raises(RuntimeError, match="closed"):
        store.prepare(())


def test_config_uses_physical_source_domain_and_requires_frozen_thresholds() -> None:
    with pytest.raises(TypeError, match="theta_match"):
        NativeTemporalStoreConfig(
            source_count=10,
            owner_count=1,
            segment_capacity=1,
        )
    config = NativeTemporalStoreConfig(
        source_count=10,
        owner_count=1,
        segment_capacity=1,
        theta_match=18,
        theta_active=20,
    )

    assert config.source_gid_base == 0
    assert config.connected_permanence_ticks == 250
    assert (config.theta_match, config.theta_active) == (18, 20)
    with pytest.raises(ValueError, match=r"connected_permanence_ticks.*\[1, 500\]"):
        NativeTemporalStoreConfig(
            source_count=10,
            owner_count=1,
            segment_capacity=1,
            theta_match=18,
            theta_active=20,
            connected_permanence_ticks=0,
        )
    with pytest.raises(ValueError, match="source_gid_base domain overflows uint32"):
        NativeTemporalStoreConfig(
            source_count=10,
            source_gid_base=(1 << 32) - 9,
            owner_count=1,
            segment_capacity=1,
            theta_match=18,
            theta_active=20,
        )


def test_segment_sources_are_canonical_but_prepare_remains_order_independent() -> None:
    config = NativeTemporalStoreConfig(
        source_count=16,
        source_gid_base=100,
        owner_count=1,
        segment_capacity=1,
        theta_match=18,
        theta_active=20,
    )
    sources = tuple(range(100, 110))

    with NativeTemporalStore(config) as store:
        with pytest.raises(ValueError, match="strictly ascending"):
            store.allocate_segment(0, (101, 100, *range(102, 110)), (250,) * 10)
        assert store.allocate_segment(0, sources, (250,) * 10) == 0
        expected = store.prepare(sources)
        assert store.prepare(tuple(reversed(sources))) == expected
        with pytest.raises(ValueError, match="unique"):
            store.prepare((*sources[:-1], sources[-2]))


def test_prepare_is_exact_and_bounded_by_matching_source_postings() -> None:
    config = NativeTemporalStoreConfig(
        source_count=64,
        owner_count=32,
        segment_capacity=8,
        connected_permanence_ticks=250,
        theta_match=16,
        theta_active=18,
    )

    with NativeTemporalStore(config) as store:
        with pytest.raises(ValueError, match=r"permanence_ticks entry.*\[1, 500\]"):
            store.allocate_segment(7, tuple(range(10)), (255,) * 9 + (0,))
        assert store.allocate_segment(7, tuple(range(10)), (255,) * 10) == 0
        assert store.allocate_segment(7, (*range(9), 10), (249,) * 10) == 1
        assert store.allocate_segment(
            9,
            (*range(8), 11, 12),
            (255,) * 5 + (249,) * 5,
        ) == 2
        # An exact duplicate for the same owner pins the stable-ID tie-break.
        assert store.allocate_segment(7, tuple(range(10)), (255,) * 10) == 3

        expected = TemporalPrepareDiagnostics(
            candidate_segments=4,
            matching_segments=4,
            active_segments=2,
            max_potential_contacts=20,
            max_connected_contacts=20,
            postings_visited=37,
            owners=(
                TemporalOwnerDiagnostics(
                    owner=7,
                    best_matching_segment=0,
                    matching_potential_contacts=20,
                    matching_connected_contacts=20,
                    best_active_segment=0,
                    active_potential_contacts=20,
                    active_connected_contacts=20,
                ),
                TemporalOwnerDiagnostics(
                    owner=9,
                    best_matching_segment=2,
                    matching_potential_contacts=16,
                    matching_connected_contacts=10,
                    best_active_segment=None,
                    active_potential_contacts=0,
                    active_connected_contacts=0,
                ),
            ),
        )
        assert store.prepare(tuple(range(10))) == expected

        # A complete segment on disjoint sources increases total segment count
        # but contributes no posting visit and cannot alter any candidate.
        assert store.allocate_segment(12, tuple(range(20, 30)), (255,) * 10) == 4
        assert store.prepare(tuple(range(10))) == expected

        for owner in (13, 14, 15):
            store.allocate_segment(owner, tuple(range(20, 30)), (255,) * 10)
        with pytest.raises(NativeTemporalStoreError) as exhausted:
            store.allocate_segment(16, tuple(range(20, 30)), (255,) * 10)
        assert exhausted.value.status == 2
        with pytest.raises(NativeTemporalStoreError) as poisoned:
            store.prepare(tuple(range(10)))
        assert poisoned.value.status == 5

    # Link the adapter result to the current physical Python fixture rather
    # than treating the hand-written case above as its own oracle.
    active_columns = 11
    codebook = {
        token: tuple(range(token * active_columns, (token + 1) * active_columns))
        for token in range(3)
    }
    cortex = SparseTemporalCortex(
        codebook,
        config=SparseTemporalConfig(
            column_count=3 * active_columns,
            active_columns=active_columns,
            base_cells_per_column=2,
            max_pyramidal_cells=3 * active_columns * 2,
            contacts_per_segment=20,
            theta_match=16,
            theta_active=18,
        ),
    )
    cortex.reset_transient()
    context_step = cortex.learn(0)
    cortex.learn(1)
    learned = cortex.learned_snapshot()
    python_diagnostics = cortex.diagnose_context(context_step.context_spikes)

    imported = list(learned.segments)
    assert all(
        tuple(sorted(sources)) == sources
        and len(sources) == len(ticks) == 10
        and all(type(value) is int and 1 <= value <= 500 for value in ticks)
        for _owner, sources, ticks in imported
    )

    assert len({owner for owner, _sources, _ticks in imported}) == len(imported)
    oracle_config = NativeTemporalStoreConfig(
        source_count=cortex.config.max_pyramidal_cells,
        source_gid_base=0,
        owner_count=cortex.config.max_pyramidal_cells,
        segment_capacity=len(imported),
        connected_permanence_ticks=250,
        theta_match=16,
        theta_active=18,
    )
    with NativeTemporalStore(oracle_config) as store:
        for segment_id, (owner, sources, ticks) in enumerate(imported):
            assert store.allocate_segment(owner, sources, ticks) == segment_id
        native_diagnostics = store.prepare(context_step.winner_cells)

    assert (
        native_diagnostics.candidate_segments,
        native_diagnostics.matching_segments,
        native_diagnostics.active_segments,
        native_diagnostics.max_potential_contacts,
        native_diagnostics.max_connected_contacts,
    ) == (
        python_diagnostics.candidate_segments,
        python_diagnostics.matching_segments,
        python_diagnostics.active_segments,
        python_diagnostics.max_potential_contacts,
        python_diagnostics.max_connected_contacts,
    )
    assert native_diagnostics.postings_visited == sum(
        len(set(sources).intersection(context_step.winner_cells))
        for _owner, sources, _ticks in imported
    )
    assert native_diagnostics.owners == tuple(
        TemporalOwnerDiagnostics(
            owner=owner,
            best_matching_segment=segment_id,
            matching_potential_contacts=20,
            matching_connected_contacts=20,
            best_active_segment=segment_id,
            active_potential_contacts=20,
            active_connected_contacts=20,
        )
        for segment_id, (owner, _sources, _ticks) in sorted(
            enumerate(imported), key=lambda item: item[1][0]
        )
    )


def test_a3_tick_snapshot_imports_directly_with_compact_physical_gid_domain() -> None:
    active_columns = 11
    column_count = 4_096
    max_pyramidal_cells = 131_072
    source_gid_base = 2 * max_pyramidal_cells + column_count
    assert source_gid_base == 266_240
    codebook = {
        token: tuple(range(token * active_columns, (token + 1) * active_columns))
        for token in range(2)
    }
    cortex = SparseTemporalCortex(
        codebook,
        config=SparseTemporalConfig(
            column_count=column_count,
            active_columns=active_columns,
            base_cells_per_column=2,
            max_pyramidal_cells=max_pyramidal_cells,
            contacts_per_segment=20,
            theta_match=18,
            theta_active=20,
            growth_policy=GrowthPolicy.GENERIC_AFFERENTS,
        ),
    )
    cortex.reset_transient()
    context_step = cortex.learn(0)
    cortex.learn(1)
    imported = cortex.learned_snapshot().segments
    python_diagnostics = cortex.diagnose_context(
        context_step.proximal_context_spikes
    )

    assert imported
    assert all(
        source_gid_base <= source < source_gid_base + column_count
        for _owner, sources, _ticks in imported
        for source in sources
    )
    native_config = NativeTemporalStoreConfig(
        source_count=column_count,
        source_gid_base=source_gid_base,
        owner_count=max_pyramidal_cells,
        segment_capacity=len(imported),
        connected_permanence_ticks=250,
        theta_match=18,
        theta_active=20,
    )
    with NativeTemporalStore(native_config) as store:
        for segment_id, (owner, sources, ticks) in enumerate(imported):
            assert store.allocate_segment(owner, sources, ticks) == segment_id
        native_diagnostics = store.prepare(
            tuple(reversed(context_step.proximal_spikes))
        )

    assert (
        native_diagnostics.candidate_segments,
        native_diagnostics.matching_segments,
        native_diagnostics.active_segments,
        native_diagnostics.max_potential_contacts,
        native_diagnostics.max_connected_contacts,
    ) == (
        python_diagnostics.candidate_segments,
        python_diagnostics.matching_segments,
        python_diagnostics.active_segments,
        python_diagnostics.max_potential_contacts,
        python_diagnostics.max_connected_contacts,
    )
    active_sources = set(context_step.proximal_spikes)
    assert native_diagnostics.postings_visited == sum(
        len(active_sources.intersection(sources))
        for _owner, sources, _ticks in imported
    )


def test_randomized_tick_native_store_matches_dense_reference_bit_exactly() -> None:
    rng = random.Random(0xA13C1)
    source_gid_base = 50_000
    source_count = 128
    owner_count = 24
    segments = [
        (
            rng.randrange(owner_count),
            tuple(
                sorted(
                    source_gid_base + source
                    for source in rng.sample(range(source_count), 10)
                )
            ),
            tuple(rng.randint(1, 500) for _ in range(10)),
        )
        for _ in range(96)
    ]
    contexts: list[tuple[int, ...]] = [()]
    for _owner, sources, _ticks in segments[:24]:
        shuffled = list(sources)
        rng.shuffle(shuffled)
        contexts.append(tuple(shuffled))
    contexts.extend(
        tuple(
            source_gid_base + source
            for source in rng.sample(range(source_count), rng.randrange(1, 33))
        )
        for _ in range(96)
    )
    config = NativeTemporalStoreConfig(
        source_count=source_count,
        source_gid_base=source_gid_base,
        owner_count=owner_count,
        segment_capacity=len(segments),
        connected_permanence_ticks=250,
        theta_match=18,
        theta_active=20,
    )

    with NativeTemporalStore(config) as store:
        for segment_id, (owner, sources, ticks) in enumerate(segments):
            assert store.allocate_segment(owner, sources, ticks) == segment_id
        for context in contexts:
            assert store.prepare(context) == _dense_reference(
                segments,
                context,
                connected_ticks=config.connected_permanence_ticks,
                theta_match=config.theta_match,
                theta_active=config.theta_active,
            )
