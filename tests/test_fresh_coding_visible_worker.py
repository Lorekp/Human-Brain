from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "experiments" / "r3_pilot" / "fresh_coding_visible_worker.py"
LIMITS = {
    "cpu_seconds": 2,
    "memory_bytes": 256 * 1024 * 1024,
    "output_bytes": 1024 * 1024,
    "processes": 1,
}


def _run(source: str, cases: list[dict[str, object]]) -> subprocess.CompletedProcess[bytes]:
    payload = {
        "source": source,
        "entry_point": "solve",
        "cases": cases,
        "limits": LIMITS,
    }
    return subprocess.run(
        [sys.executable, "-I", "-S", str(WORKER)],
        cwd=ROOT,
        input=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=5,
    )


def test_worker_emits_only_compact_case_results_for_pass_failure_and_exception() -> None:
    completed = _run(
        "def solve(value, delta, should_fail):\n"
        "    if should_fail:\n"
        "        raise ValueError('negative value')\n"
        "    return value + delta\n",
        [
            {"args": [2, 1, False], "expected": 3},
            {"args": [4, 1, False], "expected": 9},
            {"args": [-1, 1, True], "expected": 0},
        ],
    )

    assert completed.returncode == 0
    assert completed.stderr == b""
    assert completed.stdout == (
        b'{"cases":[{"actual":3,"args":[2,1,false],"expected":3,"passed":true},'
        b'{"actual":5,"args":[4,1,false],"expected":9,"passed":false},'
        b'{"args":[-1,1,true],"exception":"ValueError: negative value",'
        b'"expected":0,"passed":false}],"passed":false}'
    )


def test_worker_uses_type_sensitive_value_comparison() -> None:
    completed = _run(
        "def solve():\n    return 1\n",
        [{"args": [], "expected": True}],
    )

    result = json.loads(completed.stdout)
    assert result == {
        "cases": [{"args": [], "expected": True, "actual": 1, "passed": False}],
        "passed": False,
    }


def test_worker_rejects_imports_and_unsafe_builtins() -> None:
    imported = _run(
        "import os\n\ndef solve(value):\n    return value\n",
        [{"args": [1], "expected": 1}],
    )
    opened = _run(
        "def solve(path):\n    return open(path).read()\n",
        [{"args": ["secret.txt"], "expected": ""}],
    )

    assert json.loads(imported.stdout) == {
        "cases": [
            {
                "args": [1],
                "expected": 1,
                "exception": "ImportError: __import__ not found",
                "passed": False,
            }
        ],
        "passed": False,
    }
    assert json.loads(opened.stdout) == {
        "cases": [
            {
                "args": ["secret.txt"],
                "expected": "",
                "exception": "NameError: name 'open' is not defined",
                "passed": False,
            }
        ],
        "passed": False,
    }
    assert imported.stderr == opened.stderr == b""


def test_worker_records_non_json_actual_as_an_exception() -> None:
    completed = _run(
        "def solve(value):\n    return set([value])\n",
        [{"args": [3], "expected": [3]}],
    )

    result = json.loads(completed.stdout)
    assert result["passed"] is False
    assert result["cases"] == [
        {
            "args": [3],
            "expected": [3],
            "exception": "TypeError: Object of type set is not JSON serializable",
            "passed": False,
        }
    ]
