"""Pure closed-response scoring for the exposed KS2 2026 Amelia subset.

This implements the serialization in official-reading-next-candidate.md, not a
learner or an exam run. Item IDs are the fixture's strings. Only score_response
reads the separate official key. Request construction/validation never read it;
this is a data-routing boundary, not a filesystem security sandbox.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FIXTURE = Path(__file__).with_name("ks2_2026_amelia")
ITEM_ORDER = ("25", "27", "30", "36")
SOURCE_HASHES = {
    "questions.json": "757f5d4c60d257f73c05c2a2cce0728d40295bae939c0d0f99759387215565c3",
    "rubric.json": "ea21cf26af23358ee42123b2db421702ce17fc2e08d305c1fbb947bbf3f4a067",
    "passage.txt": "6c3dc3242592ff149942fd8c5906a0c8218b38a73afcb9a660a103f496be691d",
    "passage_structure.json": "67a1664bcd73fe435c73a74188dbbb5f88ca90923f0edec792689a494bfe5283",
    "route-map.png": "a9a4b97c339861460b7fd15429d9c71bac112b9b9987b55f5cdf5aadf77826e1",
}
ATTRIBUTION = (
    "Contains material developed by the Standards and Testing Agency for 2026 "
    "national curriculum assessments and licensed under Open Government Licence v3.0"
)
FORMATS = {
    "single_option_checkbox": "selected_options: exactly one integer index from 1 through 4.",
    "three_row_year_table": "years: three integers or null blanks, in printed row order.",
    "five_event_ordering_with_supplied_entry": (
        "ranks: four integers or null blanks for printed rows 1, 3, 4, 5. "
        "Complete writable ranks must permute 2 through 5; the supplied entry stays unchanged."
    ),
}


def _read(name: str, fixture: Path) -> bytes:
    raw = (fixture / name).read_bytes()
    if hashlib.sha256(raw).hexdigest() != SOURCE_HASHES[name]:
        raise ValueError(f"Frozen source hash mismatch: {name}")
    return raw


def _question(item: str, fixture: Path) -> dict:
    if type(item) is not str or item not in ITEM_ORDER:
        raise ValueError("Requested item must be one of the original string IDs")
    return next(
        row for row in json.loads(_read("questions.json", fixture))["items"] if row["id"] == item
    )


def learner_request(item: str, *, fixture: Path = FIXTURE) -> dict:
    """Return original material and generic format instructions, without a key.

    The caller must actually present map_png or disclose a text-only condition.
    This function does not provide a structured response producer/model binding.
    """
    question = _question(item, fixture)
    return {
        "passage": _read("passage.txt", fixture).decode("utf-8"),
        "structure": json.loads(_read("passage_structure.json", fixture)),
        "map_png": _read("route-map.png", fixture),
        "question": question,
        "response_format": (
            "One final UTF-8 JSON object with exactly item (the original string ID), "
            'status ("answer" or "stop"), and response. STOP requires response:null. '
            "An answer requires an object containing only " + FORMATS[question["format"]]
        ),
        "attribution": ATTRIBUTION,
    }


def _unique_fields(pairs: list) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_field")
        result[key] = value
    return result


def _nonfinite(value: str) -> None:
    raise ValueError("nonfinite_json_constant")


def validate_response(
    item: str, raw: bytes | None, *, late: bool = False, fixture: Path = FIXTURE
) -> dict:
    """Validate only shape/custody; values retain their original printed order.

    Delivery timing is supplied by the future runner, never inferred from an
    answer. Caller type/configuration errors raise; malformed emitted bytes score
    zero. A table containing a null blank is incomplete and cannot earn a mark.
    """
    question = _question(item, fixture)
    if raw is not None and type(raw) is not bytes:
        raise TypeError("Response must be UTF-8 bytes or None")
    if type(late) is not bool:
        raise TypeError("late must be a bool from the delivery recorder")
    result = {
        "item": item,
        "outcome": "invalid",
        "reason": None,
        "response_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
        "values": None,
    }

    def finish(outcome: str, reason: str) -> dict:
        return {**result, "outcome": outcome, "reason": reason}

    if late:
        return finish("late", "late_response")
    if raw is None:
        return finish("missing", "no_response")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return finish("invalid", "invalid_utf8")
    if not text.strip():
        return finish("blank", "blank_response")
    try:
        parsed = json.loads(text, object_pairs_hook=_unique_fields, parse_constant=_nonfinite)
    except (ValueError, RecursionError) as error:
        reason = (
            str(error)
            if str(error) in {"duplicate_field", "nonfinite_json_constant"}
            else "invalid_json"
        )
        return finish("invalid", reason)
    if type(parsed) is not dict or set(parsed) != {"item", "status", "response"}:
        return finish("invalid", "envelope_fields")
    if type(parsed["item"]) is not str or parsed["item"] != item:
        return finish("invalid", "wrong_item")
    if parsed["status"] == "stop":
        return (
            finish("stop", "explicit_stop")
            if parsed["response"] is None
            else finish("invalid", "stop_requires_null")
        )
    if parsed["status"] != "answer":
        return finish("invalid", "invalid_status")
    field, length = {
        "25": ("selected_options", 1),
        "27": ("selected_options", 1),
        "30": ("years", 3),
        "36": ("ranks", 4),
    }[item]
    response = parsed["response"]
    if type(response) is not dict or set(response) != {field}:
        return finish("invalid", "response_fields")
    values = response[field]
    if type(values) is not list or len(values) != length:
        return finish("invalid", "response_length")
    if any(
        type(value) is not int and not (value is None and item in {"30", "36"}) for value in values
    ):
        return finish("invalid", "response_value_type")
    if item in {"25", "27"} and values[0] not in range(1, 5):
        return finish("invalid", "option_range")
    if item == "36":
        filled = [value for value in values if value is not None]
        if any(value not in range(2, 6) for value in filled) or len(set(filled)) != len(filled):
            return finish("invalid", "invalid_ranks")
        writable = iter(values)
        values = [
            row["supplied_value"] if row["supplied_value"] is not None else next(writable)
            for row in question["rows"]
        ]
    result["values"] = values
    return (
        finish("blank", "incomplete_response") if None in values else finish("valid", "valid_shape")
    )


def score_response(
    item: str, raw: bytes | None, *, late: bool = False, fixture: Path = FIXTURE
) -> dict:
    """Evaluator-only official comparison; all four items are indivisible marks."""
    validation = validate_response(item, raw, late=late, fixture=fixture)
    rubric = json.loads(_read("rubric.json", fixture))
    key = next(row for row in rubric["items"] if row["id"] == item)
    result = {
        **validation,
        "marks": 0,
        "available_marks": key["marks"],
        "question_sha256": SOURCE_HASHES["questions.json"],
        "rubric_sha256": SOURCE_HASHES["rubric.json"],
    }
    if validation["outcome"] == "valid":
        if item in {"25", "27"}:
            expected = [key["correct_option_1based"]]
        elif item == "30":
            expected = key["correct_years_in_printed_row_order"]
        else:
            expected = key["correct_numbers_in_printed_row_order"]
        correct = validation["values"] == expected
        result.update(
            marks=int(correct),
            outcome="correct" if correct else "wrong",
            reason="official_key_match" if correct else "official_key_mismatch",
        )
    return result
