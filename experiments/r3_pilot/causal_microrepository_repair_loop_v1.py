"""Apply one recurrent cell-policy repair to one visible microrepository.

The module is an environment adapter around the committed V2 recurrent
fragment-relation policy.  ``prepare_microrepository_repair_v1`` authenticates
the repository source against that policy's learned root, reads visible inputs,
and freezes the policy frontier without reading desired outputs.  The returned
one-shot repair then reads the visible expected-output manifest, delegates the
only terminal decision to the frontier's physical semantic race, materializes
one unique source, patches one file, and executes one repository test command.

The test process's PASS/FAIL is converted to the existing generic
MATCH/MISMATCH feature and raced through the committed COMMIT/REVISE cells.
Parsing, filesystem I/O, process execution, and rollback remain explicit host
adapters.  This is evidence for one self-posed List[Int] -> Int microrepository
repair only, not arbitrary repository repair or benchmark capability.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import NamedTuple

from experiments.r3_pilot import causal_recurrent_fragment_relation_policy_v2 as recurrent
from experiments.r3_pilot import public_typed_output_cells as output_cells
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "CAUSAL_MICROREPOSITORY_REPAIR_LOOP_V1"
RECURRENT_POLICY_SOURCE_COMMIT = "d9e95d5"
PREPARED = f"{SCHEMA}_PREPARED"
COMMITTED = f"{SCHEMA}_COMMITTED"
STOP_SOURCE_DRIFT = f"STOP_{SCHEMA}_SOURCE_DRIFT"
STOP_ADAPTER_INPUT = f"STOP_{SCHEMA}_ADAPTER_INPUT"
STOP_POLICY_INVARIANT = f"STOP_{SCHEMA}_POLICY_INVARIANT"
STOP_PATCH_WRITE = f"STOP_{SCHEMA}_PATCH_WRITE"
STOP_TEST_FAILURE = f"STOP_{SCHEMA}_TEST_FAILURE"
STOP_CONSEQUENCE_POLICY = f"STOP_{SCHEMA}_CONSEQUENCE_POLICY"

EXPECTED_RELATION_COUNT = 2
EXPECTED_POLICY_ACTUATIONS = 2
MAX_REPOSITORY_CONSEQUENCE_ROOTS = 1
MAX_REPOSITORY_TESTS = 1
UNSELECTED_PATCH_SOURCES = 0
UNSELECTED_PATCH_WRITES = 0
UNSELECTED_REPOSITORY_TESTS = 0

CLAIM_BOUNDARY = (
    "one self-posed visible List[Int] -> Int microrepository repair using the "
    "committed recurrent two-relation cell policy; repository manifests, root "
    "source rendering, filesystem patching, process execution, and rollback "
    "remain host scaffolding"
)

_ATTESTATION_NAMESPACE = b"causal-microrepository-root-attestation-v1"
_PREPARED_NAMESPACE = b"causal-microrepository-repair-prepared-v1"
_RESULT_NAMESPACE = b"causal-microrepository-repair-result-v1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _bytes_sha256(value: bytes) -> str:
    return sha256(value).hexdigest()


class MicroRepositoryRepairSpecV1(NamedTuple):
    repo_root: Path
    source_relative_path: str
    case_manifest_relative_path: str
    expected_manifest_relative_path: str
    test_command: tuple[str, ...]
    root_program_sha256: str
    root_source_sha256: str
    test_timeout_seconds: float = 30.0


class RootSourceAttestationV1(NamedTuple):
    status: str
    policy_checkpoint_sha256: str
    root_program_sha256: str
    relation_receipt_sha256s: tuple[str, ...]
    expected_source_sha256: str
    observed_source_sha256: str
    sha256: str


class _MicroRepositoryAdapterSnapshotV1(NamedTuple):
    repo_root: Path
    source_path: Path
    case_path: Path
    expected_path: Path
    original_source: bytes
    attestation: RootSourceAttestationV1


class PreparedMicroRepositoryRepairStateV1(NamedTuple):
    status: str
    policy_checkpoint_sha256: str
    root_attestation_sha256: str
    case_count: int
    frontier_sha256: str | None
    relation_receipt_sha256s: tuple[str, ...]
    desired_reads_during_prepare: int
    source_materializations_during_prepare: int
    patch_writes_during_prepare: int
    repository_tests_during_prepare: int
    root_rollouts: int
    affordance_races: int
    policy_races: int
    effect_races: int
    selected_policy_actuations: int
    committed_edits: int
    revisions: int
    unselected_splice_executions: int
    sha256: str


class MicroRepositoryRepairWorkV1(NamedTuple):
    source_reads: int
    case_manifest_reads: int
    desired_manifest_reads: int
    root_source_renderings: int
    terminal_selection_races: int
    source_materializations: int
    selected_policy_actuations: int
    selected_patch_sources: int
    unselected_patch_sources: int
    patched_file_count: int
    patch_writes: int
    unselected_patch_writes: int
    repository_consequence_roots: int
    repository_tests: int
    unselected_repository_tests: int
    consequence_policy_races: int
    unselected_splice_executions: int
    commits: int
    revisions: int
    reverts: int


class MicroRepositoryRepairResultV1(NamedTuple):
    status: str
    prepared_sha256: str
    root_attestation_sha256: str
    selection_status: str | None
    winner_key: str | None
    exact_proof_count: int
    applied_relation_receipt_sha256s: tuple[str, ...]
    source_before_sha256: str
    source_after_sha256: str
    selection_race_sha256: str | None
    consequence_action_key: str | None
    consequence_race_sha256: str | None
    repository_test_exit_code: int | None
    repository_test_stdout_sha256: str | None
    repository_test_stderr_sha256: str | None
    work: MicroRepositoryRepairWorkV1
    sha256: str


def _child_path(root: Path, relative_path: str) -> Path:
    if not relative_path or Path(relative_path).is_absolute():
        raise ValueError("microrepository paths must be non-empty and relative")
    child = (root / relative_path).resolve()
    try:
        child.relative_to(root)
    except ValueError as exc:
        raise ValueError("microrepository path escaped its root") from exc
    return child


def _read_case_manifest(path: Path) -> tuple[tuple[list[int]], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or tuple(payload) != ("cases",):
        raise ValueError("case manifest must contain only a cases field")
    raw_cases = payload["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("case manifest must contain at least one case")
    cases: list[tuple[list[int]]] = []
    for raw_case in raw_cases:
        if not isinstance(raw_case, list) or any(
            isinstance(value, bool) or not isinstance(value, int) for value in raw_case
        ):
            raise ValueError("each visible case must be one List[Int]")
        cases.append((list(raw_case),))
    return tuple(cases)


def _read_expected_manifest(
    path: Path,
    *,
    case_count: int,
) -> tuple[output_cells.TypedOutputEncoding, ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or tuple(payload) != ("expected",):
        raise ValueError("expected manifest must contain only an expected field")
    raw_expected = payload["expected"]
    if (
        not isinstance(raw_expected, list)
        or len(raw_expected) != case_count
        or any(isinstance(value, bool) or not isinstance(value, int) for value in raw_expected)
    ):
        raise ValueError("visible expected outputs must be one Int per case")
    return tuple(
        output_cells.TypedOutputCellBank.encode(
            value,
            declared_type=output_cells.OutputType.INT,
        )
        for value in raw_expected
    )


def _root_source_attestation(
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    spec: MicroRepositoryRepairSpecV1,
    observed_source: bytes,
) -> RootSourceAttestationV1:
    policy.authenticate()
    receipts = policy.relation_receipts
    valid_receipts = (
        len(receipts) == EXPECTED_RELATION_COUNT
        and len({item.receipt_sha256 for item in receipts}) == EXPECTED_RELATION_COUNT
        and len({item.base_program_sha256 for item in receipts}) == 1
        and len({item.base_program_tokens for item in receipts}) == 1
    )
    if not valid_receipts:
        status = STOP_POLICY_INVARIANT
        root_program_sha256 = ""
    else:
        root_program_sha256 = receipts[0].base_program_sha256
        status = (
            PREPARED
            if root_program_sha256 == spec.root_program_sha256
            and _bytes_sha256(observed_source) == spec.root_source_sha256
            else STOP_SOURCE_DRIFT
        )
    receipt_sha256s = tuple(item.receipt_sha256 for item in receipts)
    expected_source_sha256 = spec.root_source_sha256
    observed_source_sha256 = _bytes_sha256(observed_source)
    attestation_sha256 = _digest(
        _ATTESTATION_NAMESPACE,
        {
            "status": status,
            "policy": policy.checkpoint_sha256,
            "root_program": root_program_sha256,
            "receipts": receipt_sha256s,
            "expected_source": expected_source_sha256,
            "observed_source": observed_source_sha256,
        },
    )
    return RootSourceAttestationV1(
        status=status,
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        root_program_sha256=root_program_sha256,
        relation_receipt_sha256s=receipt_sha256s,
        expected_source_sha256=expected_source_sha256,
        observed_source_sha256=observed_source_sha256,
        sha256=attestation_sha256,
    )


def _snapshot_microrepository_repair_v1(
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    spec: MicroRepositoryRepairSpecV1,
) -> _MicroRepositoryAdapterSnapshotV1:
    """Resolve and attest one repository with exactly one source read."""

    repo_root = spec.repo_root.resolve(strict=True)
    if not repo_root.is_dir() or not spec.test_command:
        raise ValueError("microrepository root and test command are required")
    source_path = _child_path(repo_root, spec.source_relative_path)
    case_path = _child_path(repo_root, spec.case_manifest_relative_path)
    expected_path = _child_path(repo_root, spec.expected_manifest_relative_path)
    try:
        original_source = source_path.read_bytes()
    except OSError:
        original_source = b""
    attestation = _root_source_attestation(policy, spec, original_source)
    return _MicroRepositoryAdapterSnapshotV1(
        repo_root=repo_root,
        source_path=source_path,
        case_path=case_path,
        expected_path=expected_path,
        original_source=original_source,
        attestation=attestation,
    )


def _prepared_state(
    *,
    status: str,
    policy_checkpoint_sha256: str,
    attestation: RootSourceAttestationV1,
    case_count: int,
    frontier: recurrent.RecurrentFragmentRelationFrontierV2 | None,
) -> PreparedMicroRepositoryRepairStateV1:
    frontier_state = None if frontier is None else frontier.state
    selected_actuations = 0 if frontier_state is None else frontier_state.work.selected_actuations
    root_rollouts = 0 if frontier_state is None else frontier_state.work.root_rollouts
    affordance_races = 0 if frontier_state is None else frontier_state.work.affordance_races
    policy_races = 0 if frontier_state is None else frontier_state.work.policy_races
    effect_races = 0 if frontier_state is None else frontier_state.work.effect_races
    committed_edits = 0 if frontier_state is None else frontier_state.work.committed_edits
    revisions = 0 if frontier_state is None else frontier_state.work.revisions
    unselected_splices = (
        0 if frontier_state is None else frontier_state.work.unselected_splice_executions
    )
    frontier_sha256 = None if frontier_state is None else frontier_state.sha256
    relation_sha256s = (
        () if frontier_state is None else frontier_state.applied_relation_receipt_sha256s
    )
    payload = {
        "status": status,
        "policy": policy_checkpoint_sha256,
        "attestation": attestation.sha256,
        "cases": case_count,
        "frontier": frontier_sha256,
        "relations": relation_sha256s,
        "desired_reads": 0,
        "source_materializations": 0,
        "patch_writes": 0,
        "repository_tests": 0,
        "root_rollouts": root_rollouts,
        "affordance_races": affordance_races,
        "policy_races": policy_races,
        "effect_races": effect_races,
        "selected_actuations": selected_actuations,
        "committed_edits": committed_edits,
        "revisions": revisions,
        "unselected_splices": unselected_splices,
    }
    return PreparedMicroRepositoryRepairStateV1(
        status=status,
        policy_checkpoint_sha256=policy_checkpoint_sha256,
        root_attestation_sha256=attestation.sha256,
        case_count=case_count,
        frontier_sha256=frontier_sha256,
        relation_receipt_sha256s=relation_sha256s,
        desired_reads_during_prepare=0,
        source_materializations_during_prepare=0,
        patch_writes_during_prepare=0,
        repository_tests_during_prepare=0,
        root_rollouts=root_rollouts,
        affordance_races=affordance_races,
        policy_races=policy_races,
        effect_races=effect_races,
        selected_policy_actuations=selected_actuations,
        committed_edits=committed_edits,
        revisions=revisions,
        unselected_splice_executions=unselected_splices,
        sha256=_digest(_PREPARED_NAMESPACE, payload),
    )


def _work(
    *,
    source_reads: int,
    case_manifest_reads: int,
    desired_manifest_reads: int,
    selected_policy_actuations: int,
    terminal_selection_races: int = 0,
    source_materializations: int = 0,
    selected_patch_sources: int = 0,
    patched_file_count: int = 0,
    patch_writes: int = 0,
    repository_consequence_roots: int = 0,
    repository_tests: int = 0,
    consequence_policy_races: int = 0,
    commits: int = 0,
    revisions: int = 0,
    reverts: int = 0,
) -> MicroRepositoryRepairWorkV1:
    return MicroRepositoryRepairWorkV1(
        source_reads=source_reads,
        case_manifest_reads=case_manifest_reads,
        desired_manifest_reads=desired_manifest_reads,
        root_source_renderings=0,
        terminal_selection_races=terminal_selection_races,
        source_materializations=source_materializations,
        selected_policy_actuations=selected_policy_actuations,
        selected_patch_sources=selected_patch_sources,
        unselected_patch_sources=UNSELECTED_PATCH_SOURCES,
        patched_file_count=patched_file_count,
        patch_writes=patch_writes,
        unselected_patch_writes=UNSELECTED_PATCH_WRITES,
        repository_consequence_roots=repository_consequence_roots,
        repository_tests=repository_tests,
        unselected_repository_tests=UNSELECTED_REPOSITORY_TESTS,
        consequence_policy_races=consequence_policy_races,
        unselected_splice_executions=0,
        commits=commits,
        revisions=revisions,
        reverts=reverts,
    )


@dataclass(slots=True)
class PreparedMicroRepositoryRepairV1:
    """One-shot desired-free frontier plus explicit repository adapters."""

    _policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2
    _spec: MicroRepositoryRepairSpecV1
    _repo_root: Path
    _source_path: Path
    _expected_path: Path
    _original_source: bytes
    _attestation: RootSourceAttestationV1
    _frontier: recurrent.RecurrentFragmentRelationFrontierV2 | None
    state: PreparedMicroRepositoryRepairStateV1
    _source_reads: int
    _case_manifest_reads: int
    _consumed: bool = False

    def _result(
        self,
        *,
        status: str,
        work: MicroRepositoryRepairWorkV1,
        source_after_sha256: str,
        selection: recurrent.RecurrentFragmentSelection | None = None,
        consequence_action_key: str | None = None,
        consequence_race_sha256: str | None = None,
        repository_test_exit_code: int | None = None,
        repository_test_stdout_sha256: str | None = None,
        repository_test_stderr_sha256: str | None = None,
    ) -> MicroRepositoryRepairResultV1:
        selection_status = None if selection is None else selection.status
        witness = None if selection is None else selection.witness
        receipt_sha256s = () if witness is None else witness.applied_relation_receipt_sha256s
        selection_race_sha256 = (
            None if selection is None or selection.race is None else selection.race.sha256
        )
        winner_key = None if selection is None else selection.winner_key
        exact_proof_count = 0 if selection is None else selection.exact_proof_count
        payload = {
            "status": status,
            "prepared": self.state.sha256,
            "attestation": self._attestation.sha256,
            "selection_status": selection_status,
            "winner": winner_key,
            "exact_proof_count": exact_proof_count,
            "relations": receipt_sha256s,
            "source_before": self._attestation.observed_source_sha256,
            "source_after": source_after_sha256,
            "selection_race": selection_race_sha256,
            "consequence_action": consequence_action_key,
            "consequence_race": consequence_race_sha256,
            "test_exit": repository_test_exit_code,
            "test_stdout": repository_test_stdout_sha256,
            "test_stderr": repository_test_stderr_sha256,
            "work": work._asdict(),
        }
        return MicroRepositoryRepairResultV1(
            status=status,
            prepared_sha256=self.state.sha256,
            root_attestation_sha256=self._attestation.sha256,
            selection_status=selection_status,
            winner_key=winner_key,
            exact_proof_count=exact_proof_count,
            applied_relation_receipt_sha256s=receipt_sha256s,
            source_before_sha256=self._attestation.observed_source_sha256,
            source_after_sha256=source_after_sha256,
            selection_race_sha256=selection_race_sha256,
            consequence_action_key=consequence_action_key,
            consequence_race_sha256=consequence_race_sha256,
            repository_test_exit_code=repository_test_exit_code,
            repository_test_stdout_sha256=repository_test_stdout_sha256,
            repository_test_stderr_sha256=repository_test_stderr_sha256,
            work=work,
            sha256=_digest(_RESULT_NAMESPACE, payload),
        )

    def repair(self) -> MicroRepositoryRepairResultV1:
        """Read desired once, select once, patch once, and test once."""
        if self._consumed:
            raise RuntimeError("prepared microrepository repair is one-shot")
        self._consumed = True
        selected_actuations = self.state.selected_policy_actuations
        if self.state.status != PREPARED or self._frontier is None:
            return self._result(
                status=self.state.status,
                source_after_sha256=self._attestation.observed_source_sha256,
                work=_work(
                    source_reads=self._source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=0,
                    selected_policy_actuations=selected_actuations,
                ),
            )

        try:
            current_source = self._source_path.read_bytes()
        except OSError:
            current_source = b""
        source_reads = self._source_reads + 1
        current_source_sha256 = _bytes_sha256(current_source)
        if current_source != self._original_source:
            return self._result(
                status=STOP_SOURCE_DRIFT,
                source_after_sha256=current_source_sha256,
                work=_work(
                    source_reads=source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=0,
                    selected_policy_actuations=selected_actuations,
                ),
            )

        try:
            desired_outputs = _read_expected_manifest(
                self._expected_path,
                case_count=self.state.case_count,
            )
        except OSError, UnicodeError, json.JSONDecodeError, ValueError:
            return self._result(
                status=STOP_ADAPTER_INPUT,
                source_after_sha256=current_source_sha256,
                work=_work(
                    source_reads=source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=1,
                    selected_policy_actuations=selected_actuations,
                ),
            )

        selection = self._frontier.select(desired_outputs)
        selection_races = int(selection.race is not None)
        source_materializations = self._frontier.source_materialization_count
        if selection.status != recurrent.SELECTED or selection.witness is None:
            return self._result(
                status=selection.status,
                selection=selection,
                source_after_sha256=current_source_sha256,
                work=_work(
                    source_reads=source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=1,
                    terminal_selection_races=selection_races,
                    source_materializations=source_materializations,
                    selected_policy_actuations=selected_actuations,
                ),
            )

        witness = selection.witness
        expected_receipts = {item.receipt_sha256 for item in self._policy.relation_receipts}
        observed_receipts = set(witness.applied_relation_receipt_sha256s)
        if (
            selection.exact_proof_count != 1
            or selection.materialized_witnesses != 1
            or source_materializations != 1
            or len(witness.applied_relation_receipt_sha256s) != EXPECTED_RELATION_COUNT
            or len(observed_receipts) != EXPECTED_RELATION_COUNT
            or observed_receipts != expected_receipts
            or selected_actuations != EXPECTED_POLICY_ACTUATIONS
        ):
            return self._result(
                status=STOP_POLICY_INVARIANT,
                selection=selection,
                source_after_sha256=current_source_sha256,
                work=_work(
                    source_reads=source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=1,
                    terminal_selection_races=selection_races,
                    source_materializations=source_materializations,
                    selected_policy_actuations=selected_actuations,
                ),
            )

        selected_source = witness.source.encode("utf-8")
        selected_source_sha256 = _bytes_sha256(selected_source)
        try:
            self._source_path.write_bytes(selected_source)
        except OSError:
            return self._result(
                status=STOP_PATCH_WRITE,
                selection=selection,
                source_after_sha256=current_source_sha256,
                work=_work(
                    source_reads=source_reads,
                    case_manifest_reads=self._case_manifest_reads,
                    desired_manifest_reads=1,
                    terminal_selection_races=selection_races,
                    source_materializations=source_materializations,
                    selected_policy_actuations=selected_actuations,
                    selected_patch_sources=1,
                ),
            )

        test_exit_code: int
        test_stdout: bytes
        test_stderr: bytes
        try:
            completed = subprocess.run(
                self._spec.test_command,
                cwd=self._repo_root,
                capture_output=True,
                check=False,
                timeout=self._spec.test_timeout_seconds,
            )
            test_exit_code = int(completed.returncode)
            test_stdout = completed.stdout
            test_stderr = completed.stderr
        except subprocess.TimeoutExpired as exc:
            test_exit_code = -1
            test_stdout = exc.stdout or b""
            test_stderr = exc.stderr or b""
        except OSError as exc:
            test_exit_code = -1
            test_stdout = b""
            test_stderr = str(exc).encode("utf-8", errors="replace")

        feature = (
            recurrent.FEATURE_EFFECT_MATCH
            if test_exit_code == 0
            else recurrent.FEATURE_EFFECT_MISMATCH
        )
        # d9e95d5 exposes no external consequence hook.  This is the sole,
        # localized private seam: the authenticated frozen arrays remain the
        # authority for the repository PASS/FAIL -> COMMIT/REVISE decision.
        action, consequence_race = recurrent._policy_race(
            self._policy._policy_excitatory,
            self._policy._policy_inhibitory,
            feature,
        )
        consequence_action_key = recurrent.ACTION_KEYS[action]
        common_work = {
            "source_reads": source_reads,
            "case_manifest_reads": self._case_manifest_reads,
            "desired_manifest_reads": 1,
            "terminal_selection_races": selection_races,
            "source_materializations": source_materializations,
            "selected_policy_actuations": selected_actuations,
            "selected_patch_sources": 1,
            "patched_file_count": 1,
            "patch_writes": 1,
            "repository_consequence_roots": 1,
            "repository_tests": 1,
            "consequence_policy_races": 1,
        }
        stdout_sha256 = _bytes_sha256(test_stdout)
        stderr_sha256 = _bytes_sha256(test_stderr)
        if test_exit_code == 0 and action == recurrent.ACTION_COMMIT:
            return self._result(
                status=COMMITTED,
                selection=selection,
                source_after_sha256=selected_source_sha256,
                consequence_action_key=consequence_action_key,
                consequence_race_sha256=consequence_race.sha256,
                repository_test_exit_code=test_exit_code,
                repository_test_stdout_sha256=stdout_sha256,
                repository_test_stderr_sha256=stderr_sha256,
                work=_work(**common_work, commits=1),
            )

        self._source_path.write_bytes(self._original_source)
        if test_exit_code != 0 and action == recurrent.ACTION_REVISE:
            status = STOP_TEST_FAILURE
        else:
            status = STOP_CONSEQUENCE_POLICY
        return self._result(
            status=status,
            selection=selection,
            source_after_sha256=self._attestation.observed_source_sha256,
            consequence_action_key=consequence_action_key,
            consequence_race_sha256=consequence_race.sha256,
            repository_test_exit_code=test_exit_code,
            repository_test_stdout_sha256=stdout_sha256,
            repository_test_stderr_sha256=stderr_sha256,
            work=_work(
                **common_work,
                revisions=int(action == recurrent.ACTION_REVISE),
                reverts=1,
            ),
        )


def _finalize_microrepository_repair_from_snapshot_v1(
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    spec: MicroRepositoryRepairSpecV1,
    cases: tuple[tuple[object, ...], ...],
    frontier: recurrent.RecurrentFragmentRelationFrontierV2 | None,
    snapshot: _MicroRepositoryAdapterSnapshotV1,
    source_reads: int = 1,
    case_manifest_reads: int = 1,
) -> PreparedMicroRepositoryRepairV1:
    """Finalize one attested adapter snapshot around a frozen frontier."""

    if type(policy) is not recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2:
        raise TypeError("exact frozen recurrent relation policy required")
    if type(cases) is not tuple or any(type(case) is not tuple for case in cases):
        raise TypeError("visible typed cases must be one frozen tuple")
    if source_reads != 1 or case_manifest_reads not in {0, 1}:
        raise ValueError("frontier preparation read counts changed")
    if frontier is not None and type(frontier) is not recurrent.RecurrentFragmentRelationFrontierV2:
        raise TypeError("exact recurrent relation frontier required")
    if type(snapshot) is not _MicroRepositoryAdapterSnapshotV1:
        raise TypeError("exact microrepository adapter snapshot required")
    policy.authenticate()
    if snapshot.attestation.policy_checkpoint_sha256 != policy.checkpoint_sha256:
        raise RuntimeError(STOP_POLICY_INVARIANT)
    repo_root = snapshot.repo_root
    source_path = snapshot.source_path
    expected_path = snapshot.expected_path
    original_source = snapshot.original_source
    attestation = snapshot.attestation

    prepared_frontier = frontier
    if attestation.status != PREPARED:
        status = attestation.status
        prepared_frontier = None
        case_count = 0
    elif not cases:
        status = STOP_ADAPTER_INPUT
        prepared_frontier = None
        case_count = 0
    elif frontier is None:
        status = STOP_POLICY_INVARIANT
        case_count = len(cases)
    else:
        frontier._authenticate()
        build_state = frontier.state
        status = frontier.status
        case_count = len(cases)
        learned_receipts = {item.receipt_sha256 for item in policy.relation_receipts}
        applied_receipts = set(build_state.applied_relation_receipt_sha256s)
        if status == recurrent.READY and (
            frontier._policy is not policy
            or frontier._cases != cases
            or build_state.desired_reads_during_build != 0
            or build_state.source_materializations_during_build != 0
            or build_state.terminal_program_count != 2
            or build_state.terminal_program_sha256s[0] != attestation.root_program_sha256
            or build_state.work.selected_actuations != EXPECTED_POLICY_ACTUATIONS
            or build_state.work.committed_edits != EXPECTED_POLICY_ACTUATIONS
            or build_state.work.revisions != 0
            or build_state.work.unselected_splice_executions != 0
            or len(build_state.applied_relation_receipt_sha256s) != EXPECTED_RELATION_COUNT
            or len(applied_receipts) != EXPECTED_RELATION_COUNT
            or applied_receipts != learned_receipts
        ):
            status = STOP_POLICY_INVARIANT
        elif status == recurrent.READY:
            status = PREPARED

    state = _prepared_state(
        status=status,
        policy_checkpoint_sha256=policy.checkpoint_sha256,
        attestation=attestation,
        case_count=case_count,
        frontier=prepared_frontier,
    )
    return PreparedMicroRepositoryRepairV1(
        policy,
        spec,
        repo_root,
        source_path,
        expected_path,
        original_source,
        attestation,
        prepared_frontier,
        state,
        source_reads,
        case_manifest_reads,
    )


def prepare_microrepository_repair_v1(
    policy: recurrent.FrozenCausalRecurrentFragmentRelationPolicyV2,
    signature: TypedSignature,
    spec: MicroRepositoryRepairSpecV1,
    *,
    relation_cell_gate: bool = True,
    policy_cell_gate: bool = True,
    actuation_gate: bool = True,
) -> PreparedMicroRepositoryRepairV1:
    """Authenticate and freeze a desired-free repair frontier."""

    snapshot = _snapshot_microrepository_repair_v1(policy, spec)
    if snapshot.attestation.status != PREPARED:
        return _finalize_microrepository_repair_from_snapshot_v1(
            policy,
            spec,
            (),
            frontier=None,
            snapshot=snapshot,
            source_reads=1,
            case_manifest_reads=0,
        )
    try:
        cases = _read_case_manifest(snapshot.case_path)
    except OSError, UnicodeError, json.JSONDecodeError, ValueError:
        return _finalize_microrepository_repair_from_snapshot_v1(
            policy,
            spec,
            (),
            frontier=None,
            snapshot=snapshot,
            source_reads=1,
            case_manifest_reads=1,
        )

    frontier = policy.build(
        signature,
        cases,
        relation_cell_gate=relation_cell_gate,
        policy_cell_gate=policy_cell_gate,
        actuation_gate=actuation_gate,
    )
    return _finalize_microrepository_repair_from_snapshot_v1(
        policy,
        spec,
        cases,
        frontier=frontier,
        snapshot=snapshot,
        source_reads=1,
        case_manifest_reads=1,
    )


__all__ = [
    "CLAIM_BOUNDARY",
    "COMMITTED",
    "EXPECTED_POLICY_ACTUATIONS",
    "EXPECTED_RELATION_COUNT",
    "MAX_REPOSITORY_CONSEQUENCE_ROOTS",
    "MAX_REPOSITORY_TESTS",
    "PREPARED",
    "RECURRENT_POLICY_SOURCE_COMMIT",
    "SCHEMA",
    "STOP_ADAPTER_INPUT",
    "STOP_CONSEQUENCE_POLICY",
    "STOP_PATCH_WRITE",
    "STOP_POLICY_INVARIANT",
    "STOP_SOURCE_DRIFT",
    "STOP_TEST_FAILURE",
    "MicroRepositoryRepairResultV1",
    "MicroRepositoryRepairSpecV1",
    "PreparedMicroRepositoryRepairStateV1",
    "PreparedMicroRepositoryRepairV1",
    "RootSourceAttestationV1",
    "prepare_microrepository_repair_v1",
]
