from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from experiments.r4_integrated import variable_cardinality_inventory_fixture_v1 as fixture

_FIXTURE_SOURCE_SHA256 = (
    "6b73f64371ce957328a0fb5d6cd08decc212ad1b6f04a99aa0a8e9bf24565fee"
)


def test_fixture_content_and_source_are_frozen() -> None:
    fixture.authenticate_fixture()
    source = Path(fixture.__file__).resolve().read_bytes()
    assert sha256(source).hexdigest() == _FIXTURE_SOURCE_SHA256
    assert fixture.VARIABLE_CARDINALITY_FIXTURE_SHA256 == (
        "656424b1e47347636c0b6638ba9b799b9aac8db5a3fef8df489645818acc67e7"
    )


def test_public_lessons_repeat_only_the_demonstrated_arity() -> None:
    assert len(fixture.PUBLIC_LESSONS) == 8
    assert {
        len(lesson.inventory) for lesson in fixture.PUBLIC_LESSONS
    } == {fixture.DEMONSTRATED_ARITY}
    symbols = [
        symbol for lesson in fixture.PUBLIC_LESSONS for symbol in lesson.inventory
    ]
    assert len(symbols) == len(set(symbols))
    for lesson in fixture.PUBLIC_LESSONS:
        lesson.authenticate()
        assert all(
            lesson.demonstrated_output.count(symbol) == 1
            for symbol in lesson.inventory
        )


def test_fresh_public_observations_seal_targets_and_change_cardinality() -> None:
    assert tuple(len(case.inventory) for case in fixture.FRESH_CASES) == (1, 2, 4, 5)
    forbidden = {
        "binding_off_output",
        "case_sha256",
        "expected_behavior",
        "expected_output",
    }
    for case in fixture.FRESH_CASES:
        observation = case.public_observation()
        assert not forbidden & set(observation)
        assert case.expected_output() not in repr(observation)
        assert case.binding_off_output() not in repr(observation)
        assert case.accepts_treatment_output(case.expected_output())
        assert not case.accepts_treatment_output(case.binding_off_output())
        assert case.accepts_binding_off_output(case.binding_off_output())

        expected = case.expected_output()
        positions = [expected.index(symbol) for symbol in case.inventory]
        assert positions == sorted(positions)
        assert all(expected.count(symbol) == 1 for symbol in case.inventory)
        wrong_items = (
            tuple(reversed(case.inventory))
            if len(case.inventory) > 1
            else (case.inventory[0], case.inventory[0])
        )
        wrong_output = expected.replace(
            "~|~".join(case.inventory),
            "~|~".join(wrong_items),
        )
        assert not case.accepts_treatment_output(wrong_output)


def test_unknown_and_empty_observations_require_explicit_stop() -> None:
    assert len(fixture.STOP_PROBES) == 2
    for probe in fixture.STOP_PROBES:
        observation = probe.public_observation()
        assert "expected_behavior" not in observation
        assert "probe_sha256" not in observation
        assert "stop" not in str(observation).lower()
        assert probe.accepts_output("<stop>insufficient evidence</stop>")
        assert not probe.accepts_output("qev^value^mur")
