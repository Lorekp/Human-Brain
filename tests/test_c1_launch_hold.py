from __future__ import annotations

import sys
from pathlib import Path


RUNNER_DIR = Path(__file__).resolve().parents[1] / "experiments" / "r3_pilot"
if str(RUNNER_DIR) not in sys.path:
    sys.path.insert(0, str(RUNNER_DIR))

import run_c1_cohort as c1


def test_long_c1_entry_point_is_mechanically_blocked(
    tmp_path: Path,
    capsys,
) -> None:
    output_dir = tmp_path / "must-not-exist"
    code = c1.main(["--output-dir", str(output_dir)])

    assert code == c1.EXIT_STOP
    assert not output_dir.exists()
    assert "HARD NO-GO" in capsys.readouterr().err
    assert "r3-c1-prelaunch-audit.md" in c1.C1_LAUNCH_BLOCK_REASON
