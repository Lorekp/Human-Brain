"""Mechanical response/key QA only: no learner or exam execution."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "ks2_2026_scoring", ROOT / "experiments/reading_exam_development/ks2_2026_scoring.py"
)
scoring = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scoring)


def answer(item, **response):
    return json.dumps({"item": item, "status": "answer", "response": response}).encode()


@pytest.mark.parametrize(
    ("item", "response"),
    [
        ("25", {"selected_options": [4]}),
        ("27", {"selected_options": [2]}),
        ("30", {"years": [1923, 1928, 1932]}),
        ("36", {"ranks": [5, 2, 4, 3]}),
    ],
)
def test_exact_official_complete_answers(item, response):
    result = scoring.score_response(item, answer(item, **response))
    assert (result["marks"], result["available_marks"], result["outcome"]) == (1, 1, "correct")
    if item == "36":
        assert result["values"] == [5, 1, 2, 4, 3]


@pytest.mark.parametrize(
    ("item", "response", "outcome"),
    [
        ("25", {"selected_options": [1]}, "wrong"),
        ("27", {"selected_options": [4]}, "wrong"),
        ("30", {"years": [1923, 1928, 0]}, "wrong"),
        ("30", {"years": [1923, 1928, None]}, "blank"),
        ("30", {"years": [1932, 1928, 1923]}, "wrong"),
        ("36", {"ranks": [5, 2, 3, 4]}, "wrong"),
        ("36", {"ranks": [5, 2, 4, None]}, "blank"),
        ("36", {"ranks": [None] * 4}, "blank"),
    ],
)
def test_no_partial_credit_or_reordering(item, response, outcome):
    result = scoring.score_response(item, answer(item, **response))
    assert result["marks"] == 0
    assert result["outcome"] == outcome


@pytest.mark.parametrize(
    ("item", "response", "reason"),
    [
        ("25", {"selected_options": [4, 4]}, "response_length"),
        ("25", {"selected_options": [4, 1]}, "response_length"),
        ("25", {"selected_options": []}, "response_length"),
        ("25", {"selected_options": [0]}, "option_range"),
        ("25", {"selected_options": [5]}, "option_range"),
        ("25", {"selected_options": [True]}, "response_value_type"),
        ("25", {"selected_options": [4.0]}, "response_value_type"),
        ("25", {"selected_options": ["4"]}, "response_value_type"),
        ("25", {"selected_options": [None]}, "response_value_type"),
        ("25", {"selected_options": [4], "comment": "extra"}, "response_fields"),
        ("30", {"years": [1923, 1928]}, "response_length"),
        ("30", {"years": [1923, 1928, "1932"]}, "response_value_type"),
        ("30", {"years": "1923 1928 1932"}, "response_length"),
        ("36", {"ranks": [5, 1, 2, 4, 3]}, "response_length"),
        ("36", {"ranks": [5, 2, 3, 3]}, "invalid_ranks"),
        ("36", {"ranks": [5, 2, 4, 1]}, "invalid_ranks"),
        ("36", {"ranks": [6, 2, 4, 3]}, "invalid_ranks"),
        ("36", {"ranks": [5, 2, 4, True]}, "response_value_type"),
    ],
)
def test_invalid_shapes_are_zero(item, response, reason):
    result = scoring.score_response(item, answer(item, **response))
    assert (result["marks"], result["outcome"], result["reason"]) == (0, "invalid", reason)


@pytest.mark.parametrize(
    ("raw", "reason"),
    [
        (b'{"item":"25","item":"25","status":"answer","response":{}}', "duplicate_field"),
        (
            b'{"item":"25","status":"answer","response":{"selected_options":[4],"selected_options":[4]}}',
            "duplicate_field",
        ),
        (
            b'{"item":"25","status":"answer","response":{"selected_options":[NaN]}}',
            "nonfinite_json_constant",
        ),
        (b"[]", "envelope_fields"),
        (b"\xff", "invalid_utf8"),
        (b"The answer is 4", "invalid_json"),
        (b'{"item":25,"status":"answer","response":{"selected_options":[4]}}', "wrong_item"),
        (answer("27", selected_options=[4]), "wrong_item"),
        (b'{"item":"25","status":"answer","response":null}', "response_fields"),
        (
            b'{"item":"25","status":"stop","response":{"selected_options":[4]}}',
            "stop_requires_null",
        ),
        (b'{"item":"25","status":"ANSWER","response":null}', "invalid_status"),
        (b'{"item":"25","status":"stop","response":null,"marks":1}', "envelope_fields"),
    ],
)
def test_invalid_serialization(raw, reason):
    result = scoring.score_response("25", raw)
    assert (result["marks"], result["outcome"], result["reason"]) == (0, "invalid", reason)


def test_delivery_and_abstention_are_distinct():
    stop = b'{"item":"25","status":"stop","response":null}'
    for raw, late, outcome in [
        (None, False, "missing"),
        (b" \n", False, "blank"),
        (stop, False, "stop"),
        (answer("25", selected_options=[4]), True, "late"),
    ]:
        result = scoring.score_response("25", raw, late=late)
        assert (result["marks"], result["outcome"]) == (0, outcome)
    with pytest.raises(TypeError):
        scoring.score_response("25", stop.decode())
    with pytest.raises(ValueError):
        scoring.score_response("99", stop)


def test_learner_request_and_validation_never_read_key(monkeypatch):
    original = Path.read_bytes

    def guarded(path):
        assert path.name != "rubric.json", "Learner boundary attempted to read rubric"
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    questions = json.loads(original(scoring.FIXTURE / "questions.json"))["items"]
    for item, question in zip(scoring.ITEM_ORDER, questions, strict=True):
        request = scoring.learner_request(item)
        assert request["question"] == question
        assert request["map_png"] == original(scoring.FIXTURE / "route-map.png")
        assert request["passage"] == original(scoring.FIXTURE / "passage.txt").decode("utf-8")
        assert request["attribution"] == scoring.ATTRIBUTION
        assert set(request) == {
            "passage",
            "structure",
            "map_png",
            "question",
            "response_format",
            "attribution",
        }
        assert scoring.validate_response(item, None)["outcome"] == "missing"


def test_frozen_question_and_key_custody_fail_closed(monkeypatch):
    original = Path.read_bytes
    for changed in ("questions.json", "rubric.json"):
        monkeypatch.setattr(
            Path,
            "read_bytes",
            lambda p, changed=changed: original(p) + (b" " if p.name == changed else b""),
        )
        with pytest.raises(ValueError, match="Frozen source hash mismatch"):
            scoring.score_response("25", answer("25", selected_options=[4]))
