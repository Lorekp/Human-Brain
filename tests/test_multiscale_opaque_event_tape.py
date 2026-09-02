from __future__ import annotations

from collections import Counter
from dataclasses import fields
from inspect import signature

import pytest
from humanbrain.multiscale_opaque_event_tape import (
    MultiscaleOpaqueEventTape,
    OpaqueEventTapeError,
    OpaqueWorkspaceFile,
)


def test_fresh_identifier_is_byte_identical_in_complete_query_and_source() -> None:
    fresh_atom = b"quasarLatch_927"
    query = b"Move quasarLatch_927 before the first call; keep punctuation!\r\n"
    files = (
        OpaqueWorkspaceFile(
            "nested/runtime.unit",
            b"quasarLatch_927 = ready\ncall()\n",
        ),
    )

    tape = MultiscaleOpaqueEventTape.observe(query, files)

    assert fresh_atom in tape.query_events
    assert tape.source_events.count(fresh_atom) == 1
    assert tape.reconstruct_query() == query
    assert tape.reconstruct_workspace() == files


def test_round_trip_preserves_nested_paths_binary_whitespace_and_duplicates() -> None:
    query = b"inspect echo_key::\twithout dropping [brackets] or spaces\n"
    files = (
        OpaqueWorkspaceFile("zeta/raw.data", b"\x00echo_key::\t\r\necho_key  tail\nlast"),
        OpaqueWorkspaceFile("pkg/deep/empty.bin", b""),
    )

    tape = MultiscaleOpaqueEventTape.observe(query, files)
    reordered = MultiscaleOpaqueEventTape.observe(query, tuple(reversed(files)))

    expected = tuple(sorted(files, key=lambda item: item.path))
    assert tape.reconstruct_workspace() == expected
    assert tape.source_events == reordered.source_events
    assert tape.query_events == reordered.query_events
    assert tape.source_events.count(b"echo_key") == 2
    assert (
        b"".join(item.content for item in tape.reconstruct_workspace())
        == b"\x00echo_key::\t\r\necho_key  tail\nlast"
    )


def test_complete_replacement_motor_tokens_are_live_and_materialize_exactly() -> None:
    files = (
        OpaqueWorkspaceFile(
            "pkg/deep/worker.unit",
            b"before alpha\nmid: [x]\nafter beta\r\n",
        ),
        OpaqueWorkspaceFile("notes/readme.txt", b"untouched\n"),
    )
    tape = MultiscaleOpaqueEventTape.observe(b"reorder the complete worker", files)
    replacement = b"after beta\r\nbefore alpha\nmid: [x]\n"

    motor_tokens = tape.motor_tokens_for_replacement(
        "pkg/deep/worker.unit",
        replacement,
    )
    live = Counter(tape.source_events)

    assert all(count <= live[event] for event, count in Counter(motor_tokens).items())
    materialized = tape.materialize_motor_emission(motor_tokens)
    assert materialized == (
        OpaqueWorkspaceFile("notes/readme.txt", b"untouched\n"),
        OpaqueWorkspaceFile("pkg/deep/worker.unit", replacement),
    )


def test_unobserved_replacement_unit_fails_closed() -> None:
    tape = MultiscaleOpaqueEventTape.observe(
        b"retain only observed units",
        (OpaqueWorkspaceFile("unit/a.data", b"alpha\nbeta\n"),),
    )

    with pytest.raises(OpaqueEventTapeError, match="absent from the observation"):
        tape.motor_tokens_for_replacement("unit/a.data", b"alpha\ngamma\n")


def test_public_construction_exposes_only_complete_query_and_files() -> None:
    assert tuple(field.name for field in fields(OpaqueWorkspaceFile)) == (
        "path",
        "content",
    )
    assert tuple(signature(MultiscaleOpaqueEventTape.observe).parameters) == (
        "query",
        "files",
    )
