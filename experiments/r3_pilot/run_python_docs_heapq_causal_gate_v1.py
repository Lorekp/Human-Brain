"""Run the frozen Python-docs heapq knowledge-action causal gate once.

Importing this module, authenticating its manifest, and preparing frozen
acquisition memories do not perform recall or execute candidate code.  The
scored entry point requires an explicit confirmation token and an unused result
path.  Oracle inputs are materialized only after the main and control recalls.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Mapping

from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from humanbrain import knowledge_cell_projection
from tools.build_python_docs_knowledge_corpus import build_from_cache


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = Path(__file__).with_name(
    "python_docs_heapq_causal_gate_v1.manifest.json"
)
CONFIRM_ONCE = "RUN_FROZEN_HEAPQ_CAUSAL_GATE_V1_ONCE"
MANIFEST_NAMESPACE = "humanbrain-python-docs-heapq-causal-gate-manifest-v1"
CASE_INPUT_NAMESPACE = "humanbrain-python-docs-heapq-case-inputs-v1"
ORACLE_NAMESPACE = "humanbrain-python-docs-heapq-oracle-v1"
RESULT_SCHEMA = "humanbrain.python-docs-heapq-causal-gate-result.v1"

EXPECTED_MANIFEST_SHA256 = (
    "b3420802d1f773b49ecb9680a3019cdff8411a741cffecac4dce8e5073edc4cd"
)
EXPECTED_BASE_COMMIT = "16ee37949689ca656c8d5186d1e1ec99534c24d3"
FROZEN_TASK_TEXT = (
    "Return the requested number of largest entries, ordered from largest downward."
)
FROZEN_ADMITTED_GROUPS = (
    (
        "326a9d9cc6ee317ad03272119fa0496b0cdc90f34a639d7e5788fd4384be806c",
        "3beb18c9816ea14aed097f079dba17ee7b37708f15c9de71af408056eaf6b830",
    ),
    (
        "5a0f28854ede563ff73826c67b28c42ad707ddd3b862779a47ed6568027c53e0",
        "4c99b81ebeef4a2b69c9e08eb8166357c5ecaa877d846ef6e7045d58ae13f3da",
    ),
)
FROZEN_SIGNATURE_ONLY_GROUPS = (
    ("326a9d9cc6ee317ad03272119fa0496b0cdc90f34a639d7e5788fd4384be806c",),
    ("5a0f28854ede563ff73826c67b28c42ad707ddd3b862779a47ed6568027c53e0",),
)
FROZEN_CONFLICT_GROUPS = (
    (
        "326a9d9cc6ee317ad03272119fa0496b0cdc90f34a639d7e5788fd4384be806c",
        "3beb18c9816ea14aed097f079dba17ee7b37708f15c9de71af408056eaf6b830",
        "5a0f28854ede563ff73826c67b28c42ad707ddd3b862779a47ed6568027c53e0",
        "4c99b81ebeef4a2b69c9e08eb8166357c5ecaa877d846ef6e7045d58ae13f3da",
    ),
)
FROZEN_CANDIDATES = (
    (
        "select_largest",
        "import heapq\n\ndef select_high(data, limit):\n"
        "    return heapq.nlargest(limit, data)\n",
    ),
    (
        "select_smallest",
        "import heapq\n\ndef select_high(data, limit):\n"
        "    return heapq.nsmallest(limit, data)\n",
    ),
)
FROZEN_CASE_INPUTS = (
    ([7, 1, 9, 3], 2),
    ([-5, -1, -3], 2),
    ([4, 4, 2], 2),
)
FROZEN_ORACLE = ([9, 7], [-1, -3], [4, 4])
FROZEN_IMPLEMENTATION_FILES = (
    (
        "data/knowledge/python-3.14.7-docs.lock.json",
        "c5e8f906437e9b1fe82a1fbe5289b5f630eea49fcfbc74e03cb7ca857461620c",
    ),
    (
        "experiments/g0/common_event_cells_v1.py",
        "f425c07c44942ff7cd9672a506f616d9e6f9640c52340b481a81ad7b1603eaeb",
    ),
    (
        "experiments/r3_pilot/common_knowledge_action_memory_v1.py",
        "49e9e55e9d98f80d7bf295acb906824062074124d9f69c60ca52ca7440dd9ba7",
    ),
    (
        "experiments/r3_pilot/common_rooted_structured_event_graph_v1.py",
        "7754a3c80b205beefb86e48ac9ebe3ee86408097354f8de96069ef0fab74070c",
    ),
    (
        "src/humanbrain/knowledge_cell_projection.py",
        "3a3d5ef05e9453fbbdf4df1a08df4470568a9c9889913a31c7bb7ce0662a5ab5",
    ),
    (
        "src/humanbrain/knowledge_corpus.py",
        "f7ec7ee1183535ecf5d98a807bd5cf0e395558ec33962ade3dd99f96d14d05c2",
    ),
    (
        "tools/build_python_docs_knowledge_corpus.py",
        "9d40c2fb42e5a788ddffc314ad2e66bd250c6cdd55d9bf1f90d0115a1ecc290f",
    ),
)
_SHA256 = re.compile(r"[0-9a-f]{64}")


class HeapqCausalGateError(RuntimeError):
    """The frozen gate or one of its authenticated inputs drifted."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _canonical_manifest(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _root(namespace: str, value: object) -> str:
    return sha256(namespace.encode("ascii") + b"\0" + _canonical_json(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _nested_groups(value: object) -> tuple[tuple[str, ...], ...]:
    if type(value) is not list or any(type(group) is not list for group in value):
        raise HeapqCausalGateError("chunk groups must be canonical nested lists")
    groups = tuple(tuple(group) for group in value)
    if any(any(type(item) is not str for item in group) for group in groups):
        raise HeapqCausalGateError("chunk group ids must be exact strings")
    return groups


def _authenticate_sha_fields(value: object) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise HeapqCausalGateError("manifest keys must be strings")
            if key.endswith("_sha256") and (
                type(item) is not str or _SHA256.fullmatch(item) is None
            ):
                raise HeapqCausalGateError(f"malformed root: {key}")
            _authenticate_sha_fields(item)
    elif type(value) is list:
        for item in value:
            _authenticate_sha_fields(item)


def _candidate_set_sha256(candidate_records: list[dict[str, Any]]) -> str:
    ordered = sorted(
        candidate_records,
        key=lambda record: (record["symbol"], record["candidate_id"]),
    )
    roots = tuple(record["projection_sha256"] for record in ordered)
    return _root("common-knowledge-action-candidate-set-v1", roots)


def _authenticate_manifest(manifest: dict[str, Any]) -> None:
    expected_keys = {
        "base_commit",
        "candidate_set_sha256",
        "candidates",
        "claim_boundary",
        "controls",
        "corpus",
        "execution",
        "freshness",
        "implementation",
        "manifest_sha256",
        "memory",
        "projection",
        "schema",
        "scope",
        "task",
    }
    if type(manifest) is not dict or set(manifest) != expected_keys:
        raise HeapqCausalGateError("manifest top-level schema changed")
    _authenticate_sha_fields(manifest)
    if (
        manifest["schema"] != "humanbrain.python-docs-heapq-causal-gate.v1"
        or manifest["scope"] != "PYTHON_DOCS_HEAPQ_CAUSAL_GATE_V1"
        or manifest["base_commit"] != EXPECTED_BASE_COMMIT
        or manifest["implementation"]["knowledge_bridge_commit"]
        != EXPECTED_BASE_COMMIT
        or manifest["task"]["text"] != FROZEN_TASK_TEXT
        or _nested_groups(manifest["projection"]["admitted_chunk_groups"])
        != FROZEN_ADMITTED_GROUPS
        or _nested_groups(
            manifest["controls"]["signature_only"]["admitted_chunk_groups"]
        )
        != FROZEN_SIGNATURE_ONLY_GROUPS
        or _nested_groups(manifest["controls"]["conflict"]["admitted_chunk_groups"])
        != FROZEN_CONFLICT_GROUPS
        or manifest["corpus"]["split"] != "acquire"
        or manifest["memory"]["fit_split"] != "acquire"
        or manifest["memory"]["fit_task_input_count"] != 0
        or manifest["memory"]["fit_candidate_input_count"] != 0
        or manifest["memory"]["raw_text_retention_count"] != 0
        or manifest["freshness"]
        != {
            "curation_disclosure": (
                "human-curated-known-acquire-documentation-api-pair-and-task; "
                "not blind or randomly sampled"
            ),
            "fit_candidate_input_count": 0,
            "fit_task_input_count": 0,
            "prior_candidate_execution_count": 0,
            "prior_candidate_materialization_count": 0,
            "prior_scored_recall_count": 0,
            "status": "pre-registered-unspent-at-manifest-freeze",
        }
    ):
        raise HeapqCausalGateError("frozen gate chronology or inputs changed")

    implementation_files = tuple(
        (record["path"], record["sha256"])
        for record in manifest["implementation"]["files"]
    )
    candidate_sources = tuple(
        (record["candidate_id"], record["source"])
        for record in manifest["candidates"]
    )
    case_inputs = tuple(
        (record[0], record[1]) for record in manifest["execution"]["case_inputs"]
    )
    oracle = tuple(manifest["execution"]["oracle"])
    if (
        implementation_files != FROZEN_IMPLEMENTATION_FILES
        or candidate_sources != FROZEN_CANDIDATES
        or case_inputs != FROZEN_CASE_INPUTS
        or oracle != FROZEN_ORACLE
        or manifest["execution"]["function_name"] != "select_high"
        or manifest["execution"]["result_receipt_namespace"]
        != "humanbrain-python-docs-heapq-causal-gate-result-v1"
    ):
        raise HeapqCausalGateError("frozen implementation, candidates, or oracle changed")

    if manifest["candidate_set_sha256"] != _candidate_set_sha256(
        manifest["candidates"]
    ):
        raise HeapqCausalGateError("candidate-set root changed")
    if manifest["execution"]["case_inputs_sha256"] != _root(
        CASE_INPUT_NAMESPACE, manifest["execution"]["case_inputs"]
    ):
        raise HeapqCausalGateError("case-input root changed")
    if manifest["execution"]["oracle_sha256"] != _root(
        ORACLE_NAMESPACE, manifest["execution"]["oracle"]
    ):
        raise HeapqCausalGateError("oracle root changed")

    manifest_payload = dict(manifest)
    observed_manifest_sha256 = manifest_payload.pop("manifest_sha256")
    computed_manifest_sha256 = _root(MANIFEST_NAMESPACE, manifest_payload)
    if (
        observed_manifest_sha256 != computed_manifest_sha256
        or observed_manifest_sha256 != EXPECTED_MANIFEST_SHA256
    ):
        raise HeapqCausalGateError("manifest root changed")


def load_manifest(
    path: str | Path = DEFAULT_MANIFEST_PATH,
) -> dict[str, Any]:
    manifest_path = Path(path)
    raw = manifest_path.read_bytes()
    try:
        manifest = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HeapqCausalGateError("gate manifest is not valid JSON") from exc
    if type(manifest) is not dict or raw != _canonical_manifest(manifest):
        raise HeapqCausalGateError("gate manifest is not canonical JSON")
    _authenticate_manifest(manifest)
    return manifest


def _authenticate_runtime_files(manifest: Mapping[str, Any]) -> None:
    for record in manifest["implementation"]["files"]:
        path = ROOT / record["path"]
        if not path.is_file() or _file_sha256(path) != record["sha256"]:
            raise HeapqCausalGateError(
                f"frozen implementation input changed: {record['path']}"
            )


def authenticate_preregistration(
    manifest: Mapping[str, Any],
) -> dict[str, object]:
    """Authenticate static roots without fitting, recalling, or executing."""

    if type(manifest) is not dict:
        raise TypeError("exact manifest dictionary required")
    _authenticate_manifest(manifest)
    _authenticate_runtime_files(manifest)
    task = knowledge_action.project_knowledge_action_task_v1(FROZEN_TASK_TEXT)
    if (
        list(task.factors) != manifest["task"]["factors"]
        or task.sha256 != manifest["task"]["projection_sha256"]
    ):
        raise HeapqCausalGateError("task projection root changed")
    candidates = tuple(
        knowledge_action.KnowledgePatchCandidateV1.from_source(candidate_id, source)
        for candidate_id, source in FROZEN_CANDIDATES
    )
    for candidate, record in zip(candidates, manifest["candidates"], strict=True):
        projection = candidate.projection
        if {
            "ast_sha256": projection.ast_sha256,
            "candidate_id": projection.candidate_id,
            "projection_sha256": projection.sha256,
            "source": candidate.source,
            "source_sha256": projection.source_sha256,
            "symbol": projection.symbol,
        } != record:
            raise HeapqCausalGateError("candidate projection root changed")
    return {
        "manifest_sha256": manifest["manifest_sha256"],
        "task_sha256": task.sha256,
        "candidate_set_sha256": manifest["candidate_set_sha256"],
        "candidate_projection_sha256": tuple(
            value.projection.sha256 for value in candidates
        ),
    }


def _authenticate_projection(
    projection: knowledge_cell_projection.KnowledgeCellProjectionV1,
    spec: Mapping[str, Any],
    *,
    records_sha256: str,
) -> None:
    projection.authenticate()
    if (
        projection.records_sha256 != records_sha256
        or projection.split != "acquire"
        or projection.sha256 != spec["projection_sha256"]
        or projection.graph.sha256 != spec["graph_sha256"]
        or projection.producer_receipt.sha256 != spec["producer_receipt_sha256"]
    ):
        raise HeapqCausalGateError("acquire projection root changed")


def _authenticate_memory(
    memory: knowledge_action.FrozenCommonKnowledgeActionMemoryV1,
    *,
    checkpoint_sha256: str,
    config_sha256: str,
    expected_symbols: tuple[str, ...],
    plasticity_gate: bool,
) -> None:
    memory.authenticate()
    summary = memory.training_summary
    if (
        memory.checkpoint_sha256 != checkpoint_sha256
        or summary.config_sha256 != config_sha256
        or summary.plasticity_gate is not plasticity_gate
        or summary.factor_hash_collision_count != 0
        or memory.symbol_vocabulary != expected_symbols
    ):
        raise HeapqCausalGateError("frozen knowledge-action checkpoint changed")
    if not plasticity_gate and (
        summary.association_update_count != 0 or summary.associated_chunk_count != 0
    ):
        raise HeapqCausalGateError("plasticity-off checkpoint learned associations")


@dataclass(frozen=True, slots=True)
class PreparedHeapqGateV1:
    manifest: Mapping[str, Any]
    source_summary: Mapping[str, Any]
    main_projection: knowledge_cell_projection.KnowledgeCellProjectionV1
    main_memory: knowledge_action.FrozenCommonKnowledgeActionMemoryV1
    no_plasticity_memory: knowledge_action.FrozenCommonKnowledgeActionMemoryV1
    signature_only_memory: knowledge_action.FrozenCommonKnowledgeActionMemoryV1
    conflict_memory: knowledge_action.FrozenCommonKnowledgeActionMemoryV1
    task: knowledge_action.KnowledgeActionTaskProjectionV1
    candidates: tuple[knowledge_action.KnowledgePatchCandidateV1, ...]
    environment: knowledge_action.KnowledgePatchEnvironmentV1


def prepare_gate(manifest: dict[str, Any]) -> PreparedHeapqGateV1:
    """Fit and authenticate acquire-only states; never recall or execute."""

    authenticate_preregistration(manifest)
    corpus, source_summary = build_from_cache()
    corpus_spec = manifest["corpus"]
    expected_summary = {
        "adapter_version_sha256": corpus_spec["adapter_version_sha256"],
        "archive_sha256": corpus_spec["archive_sha256"],
        "benchmark_deny_manifest_sha256": corpus_spec[
            "benchmark_deny_manifest_sha256"
        ],
        "chunk_count": corpus_spec["chunk_count"],
        "chunk_splits": corpus_spec["split_chunk_counts"],
        "exact_dedup_cluster_count": corpus_spec["exact_dedup_cluster_count"],
        "file_inventory_sha256": corpus_spec["file_inventory_sha256"],
        "included_bytes": corpus_spec["included_bytes"],
        "included_files": corpus_spec["included_files"],
        "input_manifest_sha256": corpus_spec["input_manifest_sha256"],
        "lock_sha256": corpus_spec["lock_sha256"],
        "receipt_sha256": corpus_spec["receipt_sha256"],
        "records_sha256": corpus_spec["records_sha256"],
        "tokenish": corpus_spec["tokenish"],
    }
    if dict(source_summary) != expected_summary:
        raise HeapqCausalGateError("Python documentation corpus summary changed")
    receipt_sha256 = sha256(corpus.canonical_manifest_bytes()).hexdigest()
    if (
        corpus.receipt.records_sha256 != corpus_spec["records_sha256"]
        or receipt_sha256 != corpus_spec["receipt_sha256"]
    ):
        raise HeapqCausalGateError("Python documentation corpus roots changed")

    by_id = {chunk.chunk_id: chunk for chunk in corpus.chunks}
    observed_chunks: list[dict[str, object]] = []
    for frozen in corpus_spec["chunks"]:
        chunk = by_id.get(frozen["chunk_id"])
        if chunk is None:
            raise HeapqCausalGateError("frozen acquire chunk is missing")
        observed_chunks.append(
            {
                "chunk_id": chunk.chunk_id,
                "ordinal": chunk.ordinal,
                "path": chunk.path,
                "role": frozen["role"],
                "source_file_raw_sha256": chunk.source_file_raw_sha256,
                "source_id": chunk.source_id,
                "split": chunk.split,
                "text_sha256": sha256(chunk.text.encode("utf-8")).hexdigest(),
            }
        )
    if observed_chunks != corpus_spec["chunks"]:
        raise HeapqCausalGateError("frozen acquire chunk metadata changed")

    main_projection = knowledge_cell_projection.project_knowledge_corpus_to_cells_v1(
        corpus,
        admitted_chunk_groups=FROZEN_ADMITTED_GROUPS,
        split="acquire",
    )
    _authenticate_projection(
        main_projection,
        manifest["projection"],
        records_sha256=corpus_spec["records_sha256"],
    )
    main_memory = knowledge_action.fit_common_knowledge_action_memory_v1(
        main_projection
    )
    expected_symbols = tuple(sorted(record["symbol"] for record in manifest["candidates"]))
    _authenticate_memory(
        main_memory,
        checkpoint_sha256=manifest["memory"]["expected_checkpoint_sha256"],
        config_sha256=manifest["memory"]["config_sha256"],
        expected_symbols=expected_symbols,
        plasticity_gate=True,
    )

    no_plasticity_memory = knowledge_action.fit_common_knowledge_action_memory_v1(
        main_projection, plasticity_gate=False
    )
    _authenticate_memory(
        no_plasticity_memory,
        checkpoint_sha256=manifest["memory"][
            "expected_no_plasticity_checkpoint_sha256"
        ],
        config_sha256=manifest["memory"]["config_sha256"],
        expected_symbols=expected_symbols,
        plasticity_gate=False,
    )

    signature_spec = manifest["controls"]["signature_only"]
    signature_projection = knowledge_cell_projection.project_knowledge_corpus_to_cells_v1(
        corpus,
        admitted_chunk_groups=FROZEN_SIGNATURE_ONLY_GROUPS,
        split="acquire",
    )
    _authenticate_projection(
        signature_projection,
        signature_spec,
        records_sha256=corpus_spec["records_sha256"],
    )
    signature_memory = knowledge_action.fit_common_knowledge_action_memory_v1(
        signature_projection
    )
    _authenticate_memory(
        signature_memory,
        checkpoint_sha256=signature_spec["checkpoint_sha256"],
        config_sha256=manifest["memory"]["config_sha256"],
        expected_symbols=expected_symbols,
        plasticity_gate=True,
    )

    conflict_spec = manifest["controls"]["conflict"]
    conflict_projection = knowledge_cell_projection.project_knowledge_corpus_to_cells_v1(
        corpus,
        admitted_chunk_groups=FROZEN_CONFLICT_GROUPS,
        split="acquire",
    )
    _authenticate_projection(
        conflict_projection,
        conflict_spec,
        records_sha256=corpus_spec["records_sha256"],
    )
    conflict_memory = knowledge_action.fit_common_knowledge_action_memory_v1(
        conflict_projection
    )
    _authenticate_memory(
        conflict_memory,
        checkpoint_sha256=conflict_spec["checkpoint_sha256"],
        config_sha256=manifest["memory"]["config_sha256"],
        expected_symbols=expected_symbols,
        plasticity_gate=True,
    )

    task = knowledge_action.project_knowledge_action_task_v1(FROZEN_TASK_TEXT)
    candidates = tuple(
        knowledge_action.KnowledgePatchCandidateV1.from_source(candidate_id, source)
        for candidate_id, source in FROZEN_CANDIDATES
    )
    environment = knowledge_action.KnowledgePatchEnvironmentV1(main_memory, candidates)
    if (
        task.sha256 != manifest["task"]["projection_sha256"]
        or list(task.factors) != manifest["task"]["factors"]
        or environment.materialized_candidate_ids
        or environment.execution_count != 0
    ):
        raise HeapqCausalGateError("prepared task or unspent environment changed")
    return PreparedHeapqGateV1(
        manifest=manifest,
        source_summary=dict(source_summary),
        main_projection=main_projection,
        main_memory=main_memory,
        no_plasticity_memory=no_plasticity_memory,
        signature_only_memory=signature_memory,
        conflict_memory=conflict_memory,
        task=task,
        candidates=candidates,
        environment=environment,
    )


def _control_record(
    recall: knowledge_action.KnowledgeActionRecallV1,
    *,
    passed: bool,
) -> dict[str, object]:
    return {"passed": passed, "recall": recall.record()}


def _write_result_once(path: Path, result: Mapping[str, Any]) -> None:
    if not path.parent.is_dir():
        raise HeapqCausalGateError("result parent directory does not exist")
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_manifest(dict(result)).decode("utf-8"))
    except FileExistsError as exc:
        raise HeapqCausalGateError("frozen gate result path is already spent") from exc


def run_gate_once(
    *,
    manifest_path: str | Path = DEFAULT_MANIFEST_PATH,
    result_path: str | Path,
) -> dict[str, Any]:
    """Spend the gate once.  This is the only function that calls recall."""

    destination = Path(result_path)
    if destination.exists():
        raise HeapqCausalGateError("frozen gate result path is already spent")
    manifest = load_manifest(manifest_path)
    prepared = prepare_gate(manifest)
    projections = prepared.environment.candidate_projections

    main_recall = prepared.main_memory.recall(prepared.task, projections)
    cell_off_recall = prepared.main_memory.recall(
        prepared.task, projections, cell_gate=False
    )
    no_plasticity_recall = prepared.no_plasticity_memory.recall(
        prepared.task, projections
    )
    reverse_order_recall = prepared.main_memory.recall(
        prepared.task, tuple(reversed(projections))
    )
    signature_only_recall = prepared.signature_only_memory.recall(
        prepared.task, projections
    )
    conflict_recall = prepared.conflict_memory.recall(prepared.task, projections)

    control_specs = manifest["controls"]
    controls = {
        "candidate_order_reversal": _control_record(
            reverse_order_recall,
            passed=reverse_order_recall.sha256 == main_recall.sha256,
        ),
        "cell_off": _control_record(
            cell_off_recall,
            passed=(
                cell_off_recall.status == control_specs["cell_off"]["expected_status"]
                and cell_off_recall.selected_candidate_id is None
            ),
        ),
        "conflict": _control_record(
            conflict_recall,
            passed=(
                conflict_recall.status == control_specs["conflict"]["expected_status"]
                and conflict_recall.selected_candidate_id is None
            ),
        ),
        "no_plasticity": _control_record(
            no_plasticity_recall,
            passed=(
                no_plasticity_recall.status
                == control_specs["no_plasticity"]["expected_status"]
                and no_plasticity_recall.selected_candidate_id is None
            ),
        ),
        "signature_only": _control_record(
            signature_only_recall,
            passed=(
                signature_only_recall.status
                == control_specs["signature_only"]["expected_status"]
                and signature_only_recall.selected_candidate_id is None
            ),
        ),
    }
    controls_passed = all(record["passed"] for record in controls.values())
    if prepared.environment.materialized_candidate_ids or prepared.environment.execution_count:
        raise HeapqCausalGateError("a control materialized or executed a candidate")

    # Oracle-bearing values are constructed only after every recall is immutable.
    case_inputs = tuple(
        (record[0], record[1]) for record in manifest["execution"]["case_inputs"]
    )
    oracle = tuple(manifest["execution"]["oracle"])
    outputs: tuple[object, ...] = ()
    if main_recall.status == knowledge_action.RECALLED:
        outputs = prepared.environment.execute_selected(
            main_recall,
            function_name=manifest["execution"]["function_name"],
            cases=case_inputs,
        )
    oracle_match = outputs == oracle
    execution_contract_passed = (
        prepared.environment.execution_count
        == (1 if main_recall.status == knowledge_action.RECALLED else 0)
        and prepared.environment.materialized_candidate_ids
        == (
            (main_recall.selected_candidate_id,)
            if main_recall.status == knowledge_action.RECALLED
            else ()
        )
    )
    passed = (
        main_recall.status == knowledge_action.RECALLED
        and controls_passed
        and oracle_match
        and execution_contract_passed
    )

    payload: dict[str, Any] = {
        "claim_boundary": manifest["claim_boundary"],
        "controls": controls,
        "corpus": {
            "receipt_sha256": manifest["corpus"]["receipt_sha256"],
            "records_sha256": manifest["corpus"]["records_sha256"],
            "selected_decision_state": manifest["corpus"][
                "selected_decision_state"
            ],
        },
        "counts": {
            "candidate_execution_count": prepared.environment.execution_count,
            "candidate_materialization_count": len(
                prepared.environment.materialized_candidate_ids
            ),
            "control_recall_count": 5,
            "main_recall_count": 1,
        },
        "execution": {
            "case_inputs_sha256": manifest["execution"]["case_inputs_sha256"],
            "execution_contract_passed": execution_contract_passed,
            "materialized_candidate_ids": prepared.environment.materialized_candidate_ids,
            "oracle_match": oracle_match,
            "oracle_sha256": manifest["execution"]["oracle_sha256"],
            "outputs": outputs,
        },
        "main_recall": main_recall.record(),
        "manifest_sha256": manifest["manifest_sha256"],
        "memory": asdict(prepared.main_memory.training_summary),
        "passed": passed,
        "projection": {
            "graph_sha256": prepared.main_projection.graph.sha256,
            "producer_receipt_sha256": prepared.main_projection.producer_receipt.sha256,
            "projection_sha256": prepared.main_projection.sha256,
        },
        "schema": RESULT_SCHEMA,
        "scope": manifest["scope"],
        "task_sha256": prepared.task.sha256,
    }
    receipt_sha256 = _root(
        manifest["execution"]["result_receipt_namespace"], payload
    )
    result = {**payload, "receipt_sha256": receipt_sha256}
    _write_result_once(destination, result)
    return result


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST_PATH))
    parser.add_argument("--result", required=True)
    parser.add_argument("--confirm-once", required=True)
    arguments = parser.parse_args()
    if arguments.confirm_once != CONFIRM_ONCE:
        parser.error(f"--confirm-once must equal {CONFIRM_ONCE}")
    result = run_gate_once(
        manifest_path=arguments.manifest,
        result_path=arguments.result,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


__all__ = [
    "CONFIRM_ONCE",
    "DEFAULT_MANIFEST_PATH",
    "EXPECTED_MANIFEST_SHA256",
    "HeapqCausalGateError",
    "PreparedHeapqGateV1",
    "authenticate_preregistration",
    "load_manifest",
    "prepare_gate",
    "run_gate_once",
]


if __name__ == "__main__":
    _main()
