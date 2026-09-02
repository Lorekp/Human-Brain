"""Frozen arbitrary-token fixture for learned variable-cardinality emission.

This fixture was authored before the mechanism that it is intended to test.
Eight public lessons repeat one monotone copy span at demonstrated arity three,
using disjoint renamed symbols inside one opaque prefix/separator/suffix form.
Fresh scoring cases change the cardinality to one, two, four, and five and use
only novel symbols.

Gate contract:

* treatment emits every observed inventory symbol exactly once, in order,
  between the learned fixed boundaries and separators;
* ``fixed_arity_loop_off`` uses the same scorer and must fail every fresh case;
* ``binding_off`` emits the complete variable-length form with one explicit
  index placeholder per inventory position;
* an unknown cue or an empty inventory produces an explicit neuronal STOP;
* every item continuation and the final exit/STOP remain physical cell
  decisions, rather than a host loop or decoded length counter.

Only ``public_observation()`` may be shown before all treatment and control
decisions are frozen.  Scoring-only target strings never enter that record.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256

SCHEMA = "variable-cardinality-inventory-fixture-v1"
FIXED_ARITY_LOOP_OFF = "fixed_arity_loop_off"
BINDING_OFF = "binding_off"
ABLATIONS = (FIXED_ARITY_LOOP_OFF, BINDING_OFF)

DEMONSTRATED_ARITY = 3
FRESH_CARDINALITIES = (1, 2, 4, 5)
INVENTORY_PLACEHOLDER = "<@inventory:{index}>"

_LEARNED_CUE = "nivo"
_UNKNOWN_CUE = "ulvek"
_PREFIX = "qev^"
_SEPARATOR = "~|~"
_SUFFIX = "^mur"
_TOKEN = re.compile(r"[a-z][a-z0-9_]*")
_STOP = re.compile(r"<stop>([^<>]+)</stop>", re.DOTALL)

PHYSICAL_DECISION_CONTRACT = (
    "one physical continue-or-exit cell decision after every emitted item",
    "one physical exit decision terminates after the final bound item",
    "one physical STOP decision handles unknown cue or empty inventory",
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _validate_inventory(inventory: tuple[str, ...], *, allow_empty: bool = False) -> None:
    if not isinstance(inventory, tuple) or not all(
        isinstance(symbol, str) and _TOKEN.fullmatch(symbol) is not None
        for symbol in inventory
    ):
        raise ValueError("inventory must contain only opaque atomic tokens")
    if not allow_empty and not inventory:
        raise ValueError("inventory must be non-empty")
    if len(set(inventory)) != len(inventory):
        raise ValueError("inventory symbols must be unique")


def _emit(items: tuple[str, ...]) -> str:
    return f"{_PREFIX}{_SEPARATOR.join(items)}{_SUFFIX}"


def _binding_off(items: tuple[str, ...]) -> str:
    placeholders = tuple(
        INVENTORY_PLACEHOLDER.format(index=index) for index in range(len(items))
    )
    return _emit(placeholders)


@dataclass(frozen=True, slots=True)
class PublicCardinalityLessonV1:
    """One public renamed demonstration at the sole demonstrated arity."""

    lesson_id: str
    cue: str
    inventory: tuple[str, ...]
    demonstrated_output: str
    lesson_sha256: str

    def record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "lesson_id": self.lesson_id,
            "cue": self.cue,
            "inventory": self.inventory,
            "demonstrated_output": self.demonstrated_output,
        }

    def authenticate(self) -> None:
        _validate_inventory(self.inventory)
        if (
            not self.lesson_id
            or self.cue != _LEARNED_CUE
            or len(self.inventory) != DEMONSTRATED_ARITY
            or self.demonstrated_output != _emit(self.inventory)
            or self.lesson_sha256
            != _digest(b"variable-cardinality-public-lesson-v1", self.record())
        ):
            raise ValueError("public cardinality lesson changed")


@dataclass(frozen=True, slots=True)
class FreshCardinalityCaseV1:
    """One changed-length scoring case with sealed output custody."""

    case_id: str
    cue: str
    inventory: tuple[str, ...]
    observation_sha256: str
    case_sha256: str

    def _public_record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "case_id": self.case_id,
            "cue": self.cue,
            "inventory": self.inventory,
        }

    def public_observation(self) -> dict[str, object]:
        return {
            **self._public_record(),
            "observation_sha256": self.observation_sha256,
        }

    def expected_output(self) -> str:
        """Return scoring-only treatment custody; never a public observation."""

        return _emit(self.inventory)

    def binding_off_output(self) -> str:
        """Return scoring-only placeholder custody for the binding ablation."""

        return _binding_off(self.inventory)

    def accepts_treatment_output(self, output: str) -> bool:
        return isinstance(output, str) and output == self.expected_output()

    def accepts_binding_off_output(self, output: str) -> bool:
        return isinstance(output, str) and output == self.binding_off_output()

    def scoring_record(self) -> dict[str, object]:
        return {
            **self.public_observation(),
            "expected_output": self.expected_output(),
            "binding_off_output": self.binding_off_output(),
        }

    def authenticate(self) -> None:
        _validate_inventory(self.inventory)
        expected_observation = _digest(
            b"variable-cardinality-fresh-observation-v1",
            self._public_record(),
        )
        expected_case = _digest(
            b"variable-cardinality-fresh-case-v1",
            {
                **self._public_record(),
                "observation_sha256": expected_observation,
                "expected_output": self.expected_output(),
                "binding_off_output": self.binding_off_output(),
            },
        )
        if (
            not self.case_id
            or self.cue != _LEARNED_CUE
            or len(self.inventory) not in FRESH_CARDINALITIES
            or len(self.inventory) == DEMONSTRATED_ARITY
            or self.observation_sha256 != expected_observation
            or self.case_sha256 != expected_case
        ):
            raise ValueError("fresh cardinality case changed")


@dataclass(frozen=True, slots=True)
class CardinalityStopProbeV1:
    """One public uncertainty probe with scoring-only STOP custody."""

    probe_id: str
    cue: str
    inventory: tuple[str, ...]
    observation_sha256: str
    probe_sha256: str

    def _public_record(self) -> dict[str, object]:
        return {
            "schema": SCHEMA,
            "probe_id": self.probe_id,
            "cue": self.cue,
            "inventory": self.inventory,
        }

    def public_observation(self) -> dict[str, object]:
        return {
            **self._public_record(),
            "observation_sha256": self.observation_sha256,
        }

    def accepts_output(self, output: str) -> bool:
        return isinstance(output, str) and _STOP.fullmatch(output.strip()) is not None

    def scoring_record(self) -> dict[str, object]:
        return {
            **self.public_observation(),
            "expected_behavior": "explicit_nonempty_stop",
        }

    def authenticate(self) -> None:
        _validate_inventory(self.inventory, allow_empty=True)
        unknown = self.cue == _UNKNOWN_CUE and bool(self.inventory)
        empty = self.cue == _LEARNED_CUE and not self.inventory
        expected_observation = _digest(
            b"variable-cardinality-stop-observation-v1",
            self._public_record(),
        )
        expected_probe = _digest(
            b"variable-cardinality-stop-probe-v1",
            {
                **self._public_record(),
                "observation_sha256": expected_observation,
                "expected_behavior": "explicit_nonempty_stop",
            },
        )
        if (
            not self.probe_id
            or not (unknown or empty)
            or self.observation_sha256 != expected_observation
            or self.probe_sha256 != expected_probe
        ):
            raise ValueError("cardinality STOP probe changed")


def _lesson(
    lesson_id: str,
    inventory: tuple[str, ...],
) -> PublicCardinalityLessonV1:
    record = {
        "schema": SCHEMA,
        "lesson_id": lesson_id,
        "cue": _LEARNED_CUE,
        "inventory": inventory,
        "demonstrated_output": _emit(inventory),
    }
    lesson = PublicCardinalityLessonV1(
        lesson_id=lesson_id,
        cue=_LEARNED_CUE,
        inventory=inventory,
        demonstrated_output=_emit(inventory),
        lesson_sha256=_digest(b"variable-cardinality-public-lesson-v1", record),
    )
    lesson.authenticate()
    return lesson


def _fresh_case(
    case_id: str,
    inventory: tuple[str, ...],
) -> FreshCardinalityCaseV1:
    public = {
        "schema": SCHEMA,
        "case_id": case_id,
        "cue": _LEARNED_CUE,
        "inventory": inventory,
    }
    observation_sha256 = _digest(
        b"variable-cardinality-fresh-observation-v1",
        public,
    )
    case = FreshCardinalityCaseV1(
        case_id=case_id,
        cue=_LEARNED_CUE,
        inventory=inventory,
        observation_sha256=observation_sha256,
        case_sha256=_digest(
            b"variable-cardinality-fresh-case-v1",
            {
                **public,
                "observation_sha256": observation_sha256,
                "expected_output": _emit(inventory),
                "binding_off_output": _binding_off(inventory),
            },
        ),
    )
    case.authenticate()
    return case


def _stop_probe(
    probe_id: str,
    cue: str,
    inventory: tuple[str, ...],
) -> CardinalityStopProbeV1:
    public = {
        "schema": SCHEMA,
        "probe_id": probe_id,
        "cue": cue,
        "inventory": inventory,
    }
    observation_sha256 = _digest(
        b"variable-cardinality-stop-observation-v1",
        public,
    )
    probe = CardinalityStopProbeV1(
        probe_id=probe_id,
        cue=cue,
        inventory=inventory,
        observation_sha256=observation_sha256,
        probe_sha256=_digest(
            b"variable-cardinality-stop-probe-v1",
            {
                **public,
                "observation_sha256": observation_sha256,
                "expected_behavior": "explicit_nonempty_stop",
            },
        ),
    )
    probe.authenticate()
    return probe


PUBLIC_LESSONS = tuple(
    _lesson(f"lesson-{index:02d}", inventory)
    for index, inventory in enumerate(
        (
            ("afon7", "bexu2", "cira9"),
            ("dovu4", "eska8", "feni1"),
            ("goru6", "haxi3", "ivon5"),
            ("jema2", "kori7", "luxa4"),
            ("mavi8", "nesu1", "opra6"),
            ("peli3", "quno9", "ravi5"),
            ("sotu4", "tima7", "uwen2"),
            ("vexa6", "wori1", "xanu8"),
        )
    )
)

FRESH_CASES = (
    _fresh_case("case-00", ("yebu4",)),
    _fresh_case("case-01", ("zori7", "abek3")),
    _fresh_case(
        "case-02",
        ("cedu8", "daxi2", "epon5", "fura9"),
    ),
    _fresh_case(
        "case-03",
        ("gemi1", "huro6", "ilas4", "jovi8", "kemu3"),
    ),
)

STOP_PROBES = (
    _stop_probe("probe-00", _UNKNOWN_CUE, ("lira5", "moku2")),
    _stop_probe("probe-01", _LEARNED_CUE, ()),
)


def _authenticate_fixture() -> None:
    if len(PUBLIC_LESSONS) != 8:
        raise ValueError("public loop evidence count changed")
    if tuple(len(case.inventory) for case in FRESH_CASES) != FRESH_CARDINALITIES:
        raise ValueError("fresh cardinality ladder changed")
    for lesson in PUBLIC_LESSONS:
        lesson.authenticate()
    for case in FRESH_CASES:
        case.authenticate()
    for probe in STOP_PROBES:
        probe.authenticate()

    train_symbols = {
        symbol for lesson in PUBLIC_LESSONS for symbol in lesson.inventory
    }
    fresh_symbols = {symbol for case in FRESH_CASES for symbol in case.inventory}
    probe_symbols = {symbol for probe in STOP_PROBES for symbol in probe.inventory}
    if len(train_symbols) != len(PUBLIC_LESSONS) * DEMONSTRATED_ARITY:
        raise ValueError("public lessons reuse a renamed inventory symbol")
    if train_symbols & fresh_symbols or train_symbols & probe_symbols:
        raise ValueError("fresh or probe inventory leaked into public lessons")
    if fresh_symbols & probe_symbols:
        raise ValueError("STOP probe symbol leaked into fresh scoring")

    forbidden = {
        "expected_output",
        "binding_off_output",
        "expected_behavior",
        "case_sha256",
        "probe_sha256",
    }
    observations = (
        *(case.public_observation() for case in FRESH_CASES),
        *(probe.public_observation() for probe in STOP_PROBES),
    )
    if any(forbidden & set(observation) for observation in observations):
        raise ValueError("scoring custody leaked into a public observation")


_authenticate_fixture()

_CUSTODY = {
    "schema": SCHEMA,
    "ablations": ABLATIONS,
    "demonstrated_arity": DEMONSTRATED_ARITY,
    "fresh_cardinalities": FRESH_CARDINALITIES,
    "physical_decision_contract": PHYSICAL_DECISION_CONTRACT,
    "public_lessons": tuple(lesson.record() for lesson in PUBLIC_LESSONS),
    "fresh_cases": tuple(case.scoring_record() for case in FRESH_CASES),
    "stop_probes": tuple(probe.scoring_record() for probe in STOP_PROBES),
}


def _fixture_content_sha256() -> str:
    return _digest(b"variable-cardinality-inventory-fixture-v1", _CUSTODY)


VARIABLE_CARDINALITY_FIXTURE_SHA256 = (
    "656424b1e47347636c0b6638ba9b799b9aac8db5a3fef8df489645818acc67e7"
)
if _fixture_content_sha256() != VARIABLE_CARDINALITY_FIXTURE_SHA256:
    raise ValueError("variable-cardinality fixture differs from its frozen root")


def authenticate_fixture() -> None:
    """Re-authenticate all records and the frozen content root."""

    _authenticate_fixture()
    if _fixture_content_sha256() != VARIABLE_CARDINALITY_FIXTURE_SHA256:
        raise ValueError("variable-cardinality fixture content changed")


__all__ = [
    "ABLATIONS",
    "BINDING_OFF",
    "DEMONSTRATED_ARITY",
    "FIXED_ARITY_LOOP_OFF",
    "FRESH_CARDINALITIES",
    "FRESH_CASES",
    "INVENTORY_PLACEHOLDER",
    "PHYSICAL_DECISION_CONTRACT",
    "PUBLIC_LESSONS",
    "SCHEMA",
    "STOP_PROBES",
    "VARIABLE_CARDINALITY_FIXTURE_SHA256",
    "CardinalityStopProbeV1",
    "FreshCardinalityCaseV1",
    "PublicCardinalityLessonV1",
    "authenticate_fixture",
]
