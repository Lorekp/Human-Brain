"""Frozen public micro-gate for factorized construction and fresh binding.

This fixture was authored before its successor mechanism.  It uses a tiny
arbitrary output language, not Python or V3 task families.  Six public lessons
show two cue-to-output factors in different pairings.  Three scoring cases use
the held-out pairings and fresh input symbols, so exact whole-episode replay
cannot solve them.

Gate contract:

* treatment must compose both learned factors and emit a well-formed output
  whose newly introduced local symbol is fresh and reused consistently;
* ``whole_episode_only`` must STOP on every held-out pairing;
* ``binding_off`` must still emit the correct factorized placeholder form but
  must score zero exact bound outputs.

A runner may expose all public lessons, but only ``public_observation()`` from a
scoring case before all treatment and ablation decisions are frozen.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256

SCHEMA = "factorized-output-binding-fixture-v1"
WHOLE_EPISODE_ONLY = "whole_episode_only"
BINDING_OFF = "binding_off"
ABLATIONS = (WHOLE_EPISODE_ONLY, BINDING_OFF)

INPUT_SLOT = "<@input>"
LOCAL_SLOT = "<@local>"

_CUE_A_TO_TOKEN = {"ulna": "drift", "vori": "weave", "xema": "hinge"}
_CUE_B_TO_TOKEN = {"bex": "amber", "cud": "indigo", "dor": "silver"}
_IDENTIFIER = r"[A-Za-z_][A-Za-z0-9_]*"
_OUTPUT = re.compile(
    rf"let (?P<local>{_IDENTIFIER}) = (?P<factor_a>{_IDENTIFIER})"
    rf"\((?P<input>{_IDENTIFIER})\); send (?P<factor_b>{_IDENTIFIER})"
    r"\((?P=local)\)"
)


def _program(factor_a: str, factor_b: str, input_symbol: str, local: str) -> str:
    return f"let {local} = {factor_a}({input_symbol}); send {factor_b}({local})"


@dataclass(frozen=True, slots=True)
class PublicLessonV1:
    """One public cue-pair demonstration."""

    lesson_id: str
    cue_a: str
    cue_b: str
    input_symbol: str
    demonstrated_output: str

    def record(self) -> dict[str, str]:
        return {
            "lesson_id": self.lesson_id,
            "cue_a": self.cue_a,
            "cue_b": self.cue_b,
            "input_symbol": self.input_symbol,
            "demonstrated_output": self.demonstrated_output,
        }


@dataclass(frozen=True, slots=True)
class FreshCompositionCaseV1:
    """One held-out factor pairing with scoring-only target custody."""

    case_id: str
    cue_a: str
    cue_b: str
    input_symbol: str
    scoring_only_factor_a: str
    scoring_only_factor_b: str

    def public_observation(self) -> dict[str, str]:
        return {
            "case_id": self.case_id,
            "cue_a": self.cue_a,
            "cue_b": self.cue_b,
            "input_symbol": self.input_symbol,
        }

    def binding_off_output(self) -> str:
        return _program(
            self.scoring_only_factor_a,
            self.scoring_only_factor_b,
            INPUT_SLOT,
            LOCAL_SLOT,
        )

    def accepts_treatment_output(self, output: str) -> bool:
        match = _OUTPUT.fullmatch(output)
        if match is None:
            return False
        local = match.group("local")
        return (
            match.group("factor_a") == self.scoring_only_factor_a
            and match.group("factor_b") == self.scoring_only_factor_b
            and match.group("input") == self.input_symbol
            and local not in _NONFRESH_SYMBOLS
            and local != self.input_symbol
        )

    def scoring_record(self) -> dict[str, object]:
        return {
            **self.public_observation(),
            "scoring_only_factor_a": self.scoring_only_factor_a,
            "scoring_only_factor_b": self.scoring_only_factor_b,
            "binding_off_output": self.binding_off_output(),
        }


_TRAIN_ROWS = (
    ("lesson-00", "ulna", "bex", "seed_elm", "link_ash"),
    ("lesson-01", "ulna", "dor", "seed_fir", "link_birch"),
    ("lesson-02", "vori", "bex", "seed_gale", "link_cedar"),
    ("lesson-03", "vori", "cud", "seed_hill", "link_drift"),
    ("lesson-04", "xema", "cud", "seed_iris", "link_frost"),
    ("lesson-05", "xema", "dor", "seed_jade", "link_glade"),
)

PUBLIC_LESSONS = tuple(
    PublicLessonV1(
        lesson_id=lesson_id,
        cue_a=cue_a,
        cue_b=cue_b,
        input_symbol=input_symbol,
        demonstrated_output=_program(
            _CUE_A_TO_TOKEN[cue_a],
            _CUE_B_TO_TOKEN[cue_b],
            input_symbol,
            local,
        ),
    )
    for lesson_id, cue_a, cue_b, input_symbol, local in _TRAIN_ROWS
)

FRESH_CASES = tuple(
    FreshCompositionCaseV1(
        case_id=f"fresh-composition-{index:02d}",
        cue_a=cue_a,
        cue_b=cue_b,
        input_symbol=input_symbol,
        scoring_only_factor_a=_CUE_A_TO_TOKEN[cue_a],
        scoring_only_factor_b=_CUE_B_TO_TOKEN[cue_b],
    )
    for index, (cue_a, cue_b, input_symbol) in enumerate(
        (
            ("ulna", "cud", "novel_quartz"),
            ("vori", "dor", "novel_raven"),
            ("xema", "bex", "novel_thistle"),
        )
    )
)

_NONFRESH_SYMBOLS = frozenset(
    {
        "let",
        "send",
        *_CUE_A_TO_TOKEN,
        *_CUE_A_TO_TOKEN.values(),
        *_CUE_B_TO_TOKEN,
        *_CUE_B_TO_TOKEN.values(),
        *(row[3] for row in _TRAIN_ROWS),
        *(row[4] for row in _TRAIN_ROWS),
        *(case.input_symbol for case in FRESH_CASES),
    }
)

_TRAIN_PAIRS = {(lesson.cue_a, lesson.cue_b) for lesson in PUBLIC_LESSONS}
_FRESH_PAIRS = {(case.cue_a, case.cue_b) for case in FRESH_CASES}
if len(PUBLIC_LESSONS) != 6 or len(FRESH_CASES) != 3:
    raise ValueError("factorized binding fixture cardinality changed")
if _TRAIN_PAIRS & _FRESH_PAIRS:
    raise ValueError("fresh factor pair leaked into public lessons")
if {case.cue_a for case in FRESH_CASES} != set(_CUE_A_TO_TOKEN):
    raise ValueError("fresh cases do not cover every learned cue-a factor")
if {case.cue_b for case in FRESH_CASES} != set(_CUE_B_TO_TOKEN):
    raise ValueError("fresh cases do not cover every learned cue-b factor")

_CUSTODY = {
    "ablations": ABLATIONS,
    "fresh_cases": tuple(case.scoring_record() for case in FRESH_CASES),
    "public_lessons": tuple(lesson.record() for lesson in PUBLIC_LESSONS),
    "schema": SCHEMA,
}
FACTOR_OUTPUT_BINDING_FIXTURE_SHA256 = sha256(
    json.dumps(
        _CUSTODY,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
).hexdigest()


__all__ = [
    "ABLATIONS",
    "BINDING_OFF",
    "FACTOR_OUTPUT_BINDING_FIXTURE_SHA256",
    "FRESH_CASES",
    "INPUT_SLOT",
    "LOCAL_SLOT",
    "PUBLIC_LESSONS",
    "SCHEMA",
    "WHOLE_EPISODE_ONLY",
    "FreshCompositionCaseV1",
    "PublicLessonV1",
]
