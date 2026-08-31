from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = (
    ROOT / "experiments" / "r3_pilot" / "python_docs_heapq_causal_gate_v1.manifest.json"
)
RESULT_PATH = (
    ROOT / "experiments" / "r3_pilot" / "python_docs_heapq_causal_gate_v1.result.json"
)
MANIFEST_NAMESPACE = "humanbrain-python-docs-heapq-causal-gate-manifest-v1"
EXPECTED_MANIFEST_SHA256 = (
    "b3420802d1f773b49ecb9680a3019cdff8411a741cffecac4dce8e5073edc4cd"
)
EXPECTED_RESULT_RECEIPT_SHA256 = (
    "019ac75d46f9dde4eaafa638aa4aa067350dc83ce04382476941451399bbe186"
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_file(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _root(namespace: str, value: object) -> str:
    return sha256(namespace.encode("ascii") + b"\0" + _canonical_json(value)).hexdigest()


def test_spent_heapq_evidence_authenticates_without_current_runtime() -> None:
    manifest_bytes = MANIFEST_PATH.read_bytes()
    result_bytes = RESULT_PATH.read_bytes()
    manifest = json.loads(manifest_bytes)
    result = json.loads(result_bytes)

    assert manifest_bytes == _canonical_file(manifest)
    assert result_bytes == _canonical_file(result)

    manifest_payload = dict(manifest)
    manifest_sha256 = manifest_payload.pop("manifest_sha256")
    assert manifest_sha256 == EXPECTED_MANIFEST_SHA256
    assert _root(MANIFEST_NAMESPACE, manifest_payload) == manifest_sha256

    result_payload = dict(result)
    result_receipt = result_payload.pop("receipt_sha256")
    assert result_receipt == EXPECTED_RESULT_RECEIPT_SHA256
    assert (
        _root(manifest["execution"]["result_receipt_namespace"], result_payload)
        == result_receipt
    )

    assert result["schema"] == "humanbrain.python-docs-heapq-causal-gate-result.v1"
    assert result["scope"] == manifest["scope"]
    assert result["manifest_sha256"] == manifest_sha256
    assert result["passed"] is True
    assert all(record["passed"] is True for record in result["controls"].values())
    assert result["memory"]["config_sha256"] == (
        "ca4ce6e32a4f8adc8ffb0243ab5a424da98ef619838dea2145ddc61dd9a3bdd2"
    )
    assert result["memory"]["checkpoint_sha256"] == (
        "6056f16c928c08bd9701dd15e3fecf70ed74544179978664267d3f552488a981"
    )
    assert result["main_recall"]["candidate_set_sha256"] == manifest[
        "candidate_set_sha256"
    ]
    assert result["counts"] == {
        "candidate_execution_count": 1,
        "candidate_materialization_count": 1,
        "control_recall_count": 5,
        "main_recall_count": 1,
    }
