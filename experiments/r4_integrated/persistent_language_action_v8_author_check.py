"""Ordinary dataset QA: execute 19 distinct authored answers and 8 broken starts.

This is not a learner run. No task construction, historical program replay,
checkpoint decode or model call occurs. The frozen manifest remains unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

MANIFEST_SHA = "66b0bd96be3b9e97c4eb5f9812c2b5977f294fc5900d55109445c0a303e522d0"


def run(manifest_path: Path, output: Path):
    started = time.perf_counter()
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != MANIFEST_SHA:
        raise ValueError("authored checks require the frozen V8 manifest")
    manifest = json.loads(raw)
    cases, seen = [], set()
    for row in manifest["episodes"]:
        if row["kind"] != "edit" or row["phase"] == "retention":
            continue
        files = row["grade"]["expected_after_files"]
        identity = tuple((item["path"], item["content_hex"]) for item in files)
        if identity not in seen:
            cases.append((row, "canonical", files))
            seen.add(identity)
        if row["phase"] == "recovery":
            cases.append((row, "broken", row["files"]))
    if len(seen) != 19 or len(cases) != 27:
        raise ValueError("selected author witness scope changed")
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for row, kind, files in cases:
        directory = output / (row["episode_id"] + "-" + kind)
        directory.mkdir()
        for item in files:
            if item["path"] not in ("app.py", "main.py"):
                raise ValueError("author witness path is outside the selected two-file scope")
            (directory / item["path"]).write_bytes(bytes.fromhex(item["content_hex"]))
        command = [sys.executable, "-E", "-s", "-B", str(directory / "main.py")]
        tick = time.perf_counter()
        child = subprocess.run(command, cwd=directory, capture_output=True, timeout=5, check=False)
        elapsed = time.perf_counter() - tick
        intact = all(
            (directory / item["path"]).read_bytes() == bytes.fromhex(item["content_hex"])
            for item in files
        )
        passed = intact and (
            (
                child.returncode == 0
                and child.stdout == bytes.fromhex(row["grade"]["expected_stdout_hex"])
                and child.stderr == b""
            )
            if kind == "canonical"
            else (child.returncode != 0 and b"AssertionError" in child.stderr)
        )
        result = {
            "episode_id": row["episode_id"],
            "kind": kind,
            "command": command,
            "returncode": child.returncode,
            "stdout_hex": child.stdout.hex(),
            "stderr_hex": child.stderr.hex(),
            "elapsed_seconds": elapsed,
            "files_intact": intact,
            "passed": passed,
        }
        (directory / "receipt.json").write_text(
            json.dumps(result, indent=2) + "\n", encoding="utf-8"
        )
        results.append(result)
    report = {
        "schema": "persistent-language-action-author-check-v8",
        "manifest_sha256": MANIFEST_SHA,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "passed": all(row["passed"] for row in results),
        "cases": results,
        "children": len(results),
        "canonical": 19,
        "broken": 8,
        "models": 0,
        "learning_offers": 0,
        "checkpoint_decodes": 0,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, sort_keys=True))
    if not report["passed"]:
        raise RuntimeError("authored exercise witness failed; retained actual results")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.manifest.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
