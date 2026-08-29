import ast
import json
from hashlib import sha256
from pathlib import Path

from experiments.r3_pilot import public_typed_horizon2_successor_fresh_k3 as fresh


def test_visible_pass_is_authenticated_before_fresh_target_construction() -> None:
    visible = fresh._authenticate_visible()
    assert visible["verdict"] == "PASS_HORIZON2_SUCCESSOR_VISIBLE"
    assert visible["passed_tasks"] == visible["total_tasks"] == 24
    assert sha256(fresh._VISIBLE_PATH.read_bytes()).hexdigest() == fresh._VISIBLE_SHA256

    source = Path(fresh.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    main_source = ast.get_source_segment(source, main)
    assert main_source is not None
    assert main_source.index("_authenticate_visible()") < main_source.index(
        "train_public_successor()"
    ) < main_source.index("_fresh_specs()") < main_source.index(
        "_acquire_spend_marker("
    ) < main_source.index("_run_fresh(")


def test_spent_fresh_artifact_guard_never_runs_boundary(monkeypatch) -> None:
    class SpentArtifact:
        @staticmethod
        def exists() -> bool:
            return True

        @staticmethod
        def read_text(*, encoding: str) -> str:
            assert encoding == "utf-8"
            return json.dumps({"verdict": "BANKED"})

        def __str__(self) -> str:
            return "spent.json"

    monkeypatch.setattr(fresh, "_ARTIFACT_PATH", SpentArtifact())
    monkeypatch.setattr(
        fresh,
        "_authenticate_visible",
        lambda: (_ for _ in ()).throw(AssertionError("guard authenticated visible")),
    )
    monkeypatch.setattr(
        fresh,
        "_run_fresh",
        lambda *args: (_ for _ in ()).throw(AssertionError("guard ran fresh")),
    )
    assert fresh.main() == 0


def test_spend_marker_is_exclusive_and_attests_target(tmp_path, monkeypatch) -> None:
    marker_path = tmp_path / "spend.json"
    monkeypatch.setattr(fresh, "_SPEND_MARKER_PATH", marker_path)
    arguments = (-14, -8, -2, 1, 6, 11)
    commitment = fresh._fresh_commitment(arguments, fresh._fresh_specs())

    assert fresh._acquire_spend_marker(commitment)
    assert not fresh._acquire_spend_marker(commitment)

    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["schema"] == "public-typed-horizon2-successor-fresh-k3-spend-v1"
    assert marker["planner_source_sha256"] == fresh._PLANNER_SOURCE_SHA256
    assert marker["fresh_commitment"]["arguments"] == list(arguments)
    assert marker["fresh_commitment_sha256"] == sha256(
        fresh.planner._canonical_json(commitment)
    ).hexdigest()


def test_spend_marker_guard_never_authenticates_or_runs(monkeypatch) -> None:
    class MissingArtifact:
        @staticmethod
        def exists() -> bool:
            return False

    class ExistingMarker:
        @staticmethod
        def exists() -> bool:
            return True

        def __str__(self) -> str:
            return "spent-marker.json"

    monkeypatch.setattr(fresh, "_ARTIFACT_PATH", MissingArtifact())
    monkeypatch.setattr(fresh, "_SPEND_MARKER_PATH", ExistingMarker())
    monkeypatch.setattr(
        fresh,
        "_authenticate_visible",
        lambda: (_ for _ in ()).throw(AssertionError("marker guard authenticated visible")),
    )
    monkeypatch.setattr(
        fresh,
        "_run_fresh",
        lambda *args: (_ for _ in ()).throw(AssertionError("marker guard ran fresh")),
    )
    assert fresh.main() == 2
