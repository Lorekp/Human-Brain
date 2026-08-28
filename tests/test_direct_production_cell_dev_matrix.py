from experiments.r3_pilot.direct_production_cell_dev_matrix import (
    run_development_matrix,
    visible_compositions,
    visible_teachers,
)


def test_visible_matrix_separates_replay_from_six_compositional_holdouts() -> None:
    report = run_development_matrix()

    assert report.teacher_total == len(visible_teachers()) == 6
    assert report.composition_total == len(visible_compositions()) == 6
    assert report.teacher_exact == sum(result.exact for result in report.teacher_replays)
    assert report.composition_exact == sum(result.exact for result in report.compositions)
    assert {case.tokens for case in visible_teachers()}.isdisjoint(
        case.tokens for case in visible_compositions()
    )
    assert all(result.root_allowed_currents_pa for result in report.compositions)
    assert all(
        (result.first_mismatch is None) == result.exact
        for result in report.compositions
        if result.failure is None
    )
    assert report.elapsed_seconds < 15.0

    print(
        f"teacher={report.teacher_exact}/{report.teacher_total} "
        f"composition={report.composition_exact}/{report.composition_total} "
        f"seconds={report.elapsed_seconds:.3f} "
        f"teacher_outputs="
        f"{[(result.name, result.actual_tokens) for result in report.teacher_replays]} "
        f"composition_outputs="
        f"{[(result.name, result.actual_tokens, result.first_mismatch) for result in report.compositions]}"
    )
