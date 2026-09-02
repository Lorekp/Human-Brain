"""Fresh public cases for source-local identifier alpha binding.

Each case is derived mechanically from one frozen public coding-prior case.
Only the exact identifier lexemes ``solve``, ``arg0``, ``arg1``, and ``arg2``
change; behavior, source structure, and public task language otherwise remain
the same.  Fresh renamings remain private derivation/authentication data.  The
public case exposes the exact starter observation and a scoring-only expected
source, but no semantic symbol inventory.  This module constructs no executable
environment or private target.
"""

from __future__ import annotations

import json
import re
import tokenize
from dataclasses import dataclass
from hashlib import sha256
from io import StringIO

from experiments.r4_integrated.public_coding_prior_v1 import (
    PUBLIC_CODING_PRIOR_V1_SHA256,
    PublicCodingPriorCaseV1,
    public_coding_prior_v1,
)

SCHEMA = "source-local-alpha-binding-tasks-v1"
CASE_COUNT = 8

_SOURCE_SYMBOLS = ("solve", "arg0", "arg1", "arg2")
_IDENTIFIER_LEXEME = re.compile(
    r"(?<![A-Za-z0-9_])(?:solve|arg0|arg1|arg2)(?![A-Za-z0-9_])"
)
_CASE_ID = re.compile(r"alpha-binding-(\d{2})")
_STREAM_NAMESPACE = b"humanbrain-source-local-alpha-binding-stream-v1\0"

# Private, repository-fresh lexical replacements in ``_SOURCE_SYMBOLS`` order.
# They mechanically derive/authenticate cases and are never exposed as an
# ordered semantic inventory.  No replacement is reused between cases.
_FRESH_RENAMINGS = (
    ("cairn_wisp", "dovetail_ash", "elmstone_bay", "fablecrest_ivy"),
    ("garnet_hush", "hollowmere_jade", "ironleaf_kite", "juniper_lark"),
    ("kindred_moss", "lattice_north", "marble_oak", "nimbus_pine"),
    ("opal_quill", "prairie_rune", "quartz_sable", "river_thorn"),
    ("silver_umber", "tundra_vale", "velvet_wren", "willow_xylem"),
    ("yarrow_crest", "zephyr_dawn", "amber_finch", "bramble_glade"),
    ("cobalt_heath", "drift_isle", "ember_jolt", "flint_knoll"),
    ("grove_lumen", "harbor_mirth", "indigo_nest", "jasmine_orbit"),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _mapping(fresh_renaming: tuple[str, ...]) -> dict[str, str]:
    if (
        type(fresh_renaming) is not tuple
        or len(fresh_renaming) != len(_SOURCE_SYMBOLS)
        or len(set(fresh_renaming)) != len(fresh_renaming)
        or any(
            type(symbol) is not str or not symbol.isidentifier()
            for symbol in fresh_renaming
        )
    ):
        raise ValueError("alpha-binding renaming is not exact unique identifiers")
    return dict(zip(_SOURCE_SYMBOLS, fresh_renaming, strict=True))


def _replace_prompt_identifier_lexemes(text: str, replacements: dict[str, str]) -> str:
    if type(text) is not str or not text:
        raise TypeError("alpha-binding prompt must be one non-empty exact string")
    result = _IDENTIFIER_LEXEME.sub(lambda match: replacements[match.group(0)], text)
    if result == text or _IDENTIFIER_LEXEME.search(result) is not None:
        raise ValueError("alpha-binding prompt identifier replacement was incomplete")
    return result


def _source_tokens(source: str) -> tuple[tokenize.TokenInfo, ...]:
    if type(source) is not str or not source:
        raise TypeError("alpha-binding source must be one non-empty exact string")
    try:
        return tuple(tokenize.generate_tokens(StringIO(source).readline))
    except (IndentationError, tokenize.TokenError) as error:
        raise ValueError("alpha-binding source tokenization failed") from error


def _replace_source_identifier_lexemes(source: str, replacements: dict[str, str]) -> str:
    tokens = _source_tokens(source)
    seen: set[str] = set()
    replaced: list[tokenize.TokenInfo] = []
    for token in tokens:
        replacement = token.string
        if token.type == tokenize.NAME and token.string in replacements:
            seen.add(token.string)
            replacement = replacements[token.string]
        replaced.append(token._replace(string=replacement))
    if seen != set(_SOURCE_SYMBOLS):
        raise ValueError("alpha-binding source omitted a required identifier lexeme")
    result = tokenize.untokenize(replaced)
    if any(
        token.type == tokenize.NAME and token.string in _SOURCE_SYMBOLS
        for token in _source_tokens(result)
    ):
        raise ValueError("alpha-binding source retained an original identifier lexeme")
    return result


@dataclass(frozen=True, slots=True)
class AlphaBindingCaseV1:
    """One immutable public observation with scoring-only source custody."""

    case_id: str
    prompt: str
    starter_source: str
    expected_source: str

    def record(self) -> dict[str, object]:
        return {
            "case_id": self.case_id,
            "prompt": self.prompt,
            "starter_source": self.starter_source,
            "scoring_only_expected_source": self.expected_source,
        }

    def authenticate(self) -> None:
        match = _CASE_ID.fullmatch(self.case_id)
        if match is None:
            raise ValueError("alpha-binding case id is not canonical")
        index = int(match.group(1))
        if not 0 <= index < CASE_COUNT:
            raise ValueError("alpha-binding case id is outside the frozen stream")
        expected = _derive_case(public_coding_prior_v1()[index], index)
        if self != expected:
            raise ValueError("alpha-binding case content failed authentication")


def _derive_case(prior: PublicCodingPriorCaseV1, index: int) -> AlphaBindingCaseV1:
    if type(prior) is not PublicCodingPriorCaseV1 or not 0 <= index < CASE_COUNT:
        raise TypeError("alpha-binding derivation requires one frozen prior case and index")
    fresh_renaming = _FRESH_RENAMINGS[index]
    replacements = _mapping(fresh_renaming)
    prior_text = "\n".join(
        (prior.public_prompt, prior.starter_source, prior.successful_source)
    )
    if any(
        re.search(rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])", prior_text)
        for symbol in fresh_renaming
    ):
        raise ValueError("alpha-binding fresh identifier was already used by its prior case")
    return AlphaBindingCaseV1(
        case_id=f"alpha-binding-{index:02d}",
        prompt=_replace_prompt_identifier_lexemes(prior.public_prompt, replacements),
        starter_source=_replace_source_identifier_lexemes(
            prior.starter_source, replacements
        ),
        expected_source=_replace_source_identifier_lexemes(
            prior.successful_source, replacements
        ),
    )


def _stream_sha256(cases: tuple[AlphaBindingCaseV1, ...]) -> str:
    return sha256(
        _STREAM_NAMESPACE
        + _canonical_json(
            {
                "cases": tuple(case.record() for case in cases),
                "public_coding_prior_sha256": PUBLIC_CODING_PRIOR_V1_SHA256,
                "schema": SCHEMA,
            }
        )
    ).hexdigest()


_PRIOR_CASES = public_coding_prior_v1()
if len(_PRIOR_CASES) != CASE_COUNT:
    raise ValueError("alpha-binding scaffold requires exactly eight frozen prior cases")
_CASES = tuple(_derive_case(prior, index) for index, prior in enumerate(_PRIOR_CASES))
if len({symbol for renaming in _FRESH_RENAMINGS for symbol in renaming}) != 4 * CASE_COUNT:
    raise ValueError("alpha-binding fresh identifiers must be unique across the stream")

ALPHA_BINDING_STREAM_SHA256 = _stream_sha256(_CASES)


def source_local_alpha_binding_cases_v1() -> tuple[AlphaBindingCaseV1, ...]:
    """Return the authenticated immutable public alpha-binding stream."""

    for case in _CASES:
        case.authenticate()
    if _stream_sha256(_CASES) != ALPHA_BINDING_STREAM_SHA256:
        raise ValueError("alpha-binding stream content failed authentication")
    return _CASES


__all__ = [
    "ALPHA_BINDING_STREAM_SHA256",
    "CASE_COUNT",
    "SCHEMA",
    "AlphaBindingCaseV1",
    "source_local_alpha_binding_cases_v1",
]
