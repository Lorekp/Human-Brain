from __future__ import annotations

from dataclasses import replace
from hashlib import sha256

import pytest
from experiments.r3_pilot.common_preceding_let_declaration_occurrences_v1 import (
    FARTHER_PRECEDING_DECLARATION,
    NEARER_PRECEDING_DECLARATION,
    PRECEDING_DECLARATION_RELATIONS,
    PrecedingLetDeclarationOccurrenceError,
    preceding_declaration_relation_component_sha256_v1,
    project_preceding_let_declaration_occurrences_v1,
)


def _primary_span(source: str, identifier: str = "focus") -> tuple[int, int]:
    start = source.index(identifier)
    return start, start + len(identifier)


def _project(source: str, identifier: str = "focus"):
    start, end = _primary_span(source, identifier)
    return project_preceding_let_declaration_occurrences_v1(
        source,
        primary_byte_start=start,
        primary_byte_end=end,
    )


def test_projects_and_authenticates_complete_two_occurrence_roster() -> None:
    source = (
        "fn main() {\n"
        "    let alpha = 11_i64;\n"
        "    let beta = 22_i64;\n"
        "    let focus = alpha + beta;\n"
        "    println!(\"{}\", focus);\n"
        "}\n"
    )

    roster = _project(source)
    roster.authenticate()

    assert roster.primary_span_bytes == b"focus"
    assert tuple(item.identifier for item in roster.occurrences) == ("alpha", "beta")
    assert tuple(item.relation for item in roster.occurrences) == (
        FARTHER_PRECEDING_DECLARATION,
        NEARER_PRECEDING_DECLARATION,
    )
    assert (
        roster.occurrences[0].distance_to_primary_bytes
        > roster.occurrences[1].distance_to_primary_bytes
    )
    for occurrence in roster.occurrences:
        occurrence.authenticate(
            source=source,
            token_frame=roster.token_frame,
            primary_byte_start=roster.primary_byte_start,
            primary_byte_end=roster.primary_byte_end,
        )
        assert (
            source.encode("ascii")[
                occurrence.identifier_byte_start : occurrence.identifier_byte_end
            ]
            == occurrence.identifier_bytes
        )
        assert occurrence.declaration_line_bytes.endswith(b";")
        assert occurrence.declaration_line_sha256 == sha256(
            occurrence.declaration_line_bytes
        ).hexdigest()


def test_relation_roots_ignore_names_padding_and_source_custody() -> None:
    first = _project(
        "fn main() {\n"
        "    let alpha = 1_i64;\n"
        "    let beta = 2_i64;\n"
        "    let focus = alpha + beta;\n"
        "}\n"
    )
    second = _project(
        "\n\nfn main() {\n"
        "    let renamed_far = 101_i64;\n"
        "\n"
        "    let renamed_near = 202_i64;\n"
        "    let focus = renamed_far + renamed_near;\n"
        "}\n"
    )

    assert tuple(item.relation_component_sha256 for item in first.occurrences) == tuple(
        item.relation_component_sha256 for item in second.occurrences
    )
    assert first.source_sha256 != second.source_sha256
    assert first.occurrences[0].sha256 != second.occurrences[0].sha256
    assert tuple(
        preceding_declaration_relation_component_sha256_v1(relation)
        for relation in PRECEDING_DECLARATION_RELATIONS
    ) == tuple(item.relation_component_sha256 for item in first.occurrences)


@pytest.mark.parametrize(
    "declarations",
    (
        (),
        ("    let only = 1_i64;\n",),
        (
            "    let first = 1_i64;\n",
            "    let second = 2_i64;\n",
            "    let third = 3_i64;\n",
        ),
    ),
)
def test_fails_closed_without_exactly_two_eligible_occurrences(
    declarations: tuple[str, ...],
) -> None:
    source = "fn main() {\n" + "".join(declarations) + "    let focus = 9_i64;\n}\n"

    with pytest.raises(
        PrecedingLetDeclarationOccurrenceError,
        match="exactly two occurrences",
    ):
        _project(source)


def test_excludes_the_declaration_matching_primary_identifier_bytes() -> None:
    source = (
        "fn main() {\n"
        "    let far_target = 1_i64;\n"
        "    let near_target = 2_i64;\n"
        "    let focus = far_target + near_target;\n"
        "}\n"
    )

    roster = _project(source)

    assert {item.identifier_bytes for item in roster.occurrences} == {
        b"far_target",
        b"near_target",
    }
    assert b"focus" not in {item.identifier_bytes for item in roster.occurrences}


@pytest.mark.parametrize(
    "bad_line",
    (
        "    let mut far_target = 1_i64;\n",
        "    let far_target = 1_i64\n",
        "    prefix(); let far_target = 1_i64;\n",
    ),
)
def test_rejects_malformed_or_unterminated_preceding_declarations(bad_line: str) -> None:
    source = (
        "fn main() {\n"
        + bad_line
        + "    let near_target = 2_i64;\n"
        + "    let focus = near_target;\n"
        + "}\n"
    )

    with pytest.raises(PrecedingLetDeclarationOccurrenceError):
        _project(source)


def test_forged_nested_occurrence_and_roster_custody_fail_authentication() -> None:
    source = (
        "fn main() {\n"
        "    let alpha = 1_i64;\n"
        "    let beta = 2_i64;\n"
        "    let focus = alpha + beta;\n"
        "}\n"
    )
    roster = _project(source)
    forged_occurrence = replace(roster.occurrences[0], identifier_bytes=b"omega")
    forged_roster = replace(
        roster,
        occurrences=(forged_occurrence, roster.occurrences[1]),
    )

    with pytest.raises(PrecedingLetDeclarationOccurrenceError):
        forged_occurrence.authenticate(
            source=source,
            token_frame=roster.token_frame,
            primary_byte_start=roster.primary_byte_start,
            primary_byte_end=roster.primary_byte_end,
        )
    with pytest.raises(PrecedingLetDeclarationOccurrenceError):
        forged_roster.authenticate()


def test_token_gap_byte_spans_remain_exact_across_comments_and_blank_lines() -> None:
    source = (
        "fn main() {\r\n"
        "    // let ghost = 0_i64;\r\n"
        "    let farther = 10_i64;\r\n"
        "\r\n"
        "    let nearer = 20_i64;\r\n"
        "    let focus = farther + nearer;\r\n"
        "    println!(\"let hidden = 3; {}\", focus);\r\n"
        "}\r\n"
    )

    roster = _project(source)
    frame = roster.token_frame
    reconstructed = "".join(
        (*(frame.gaps[index] + token for index, token in enumerate(frame.tokens)), frame.gaps[-1])
    )

    assert reconstructed == source
    for occurrence in roster.occurrences:
        token_index = occurrence.identifier_token_index
        assert frame.tokens[token_index].encode("ascii") == occurrence.identifier_bytes
        assert frame.kinds[token_index] == "IDENTIFIER"
        assert source.encode("ascii")[
            occurrence.declaration_line_byte_start : occurrence.declaration_line_byte_end
        ] == occurrence.declaration_line_bytes


def test_primary_span_must_be_one_exact_identifier_token() -> None:
    source = (
        "fn main() {\n"
        "    let alpha = 1_i64;\n"
        "    let beta = 2_i64;\n"
        "    let focus = alpha + beta;\n"
        "}\n"
    )
    start, end = _primary_span(source)

    with pytest.raises(
        PrecedingLetDeclarationOccurrenceError,
        match="exactly one identifier token",
    ):
        project_preceding_let_declaration_occurrences_v1(
            source,
            primary_byte_start=start,
            primary_byte_end=end - 1,
        )


def test_rejects_a_retained_declaration_line_terminated_only_by_eof() -> None:
    source = (
        "fn main() {\n"
        "    let farther = 1_i64;\n"
        "    let nearer = focus;"
    )
    start, end = _primary_span(source)

    with pytest.raises(
        PrecedingLetDeclarationOccurrenceError,
        match="not newline-terminated",
    ):
        project_preceding_let_declaration_occurrences_v1(
            source,
            primary_byte_start=start,
            primary_byte_end=end,
        )


def test_relation_component_rejects_noncanonical_labels() -> None:
    with pytest.raises(PrecedingLetDeclarationOccurrenceError):
        preceding_declaration_relation_component_sha256_v1("CORRECT_TARGET")
