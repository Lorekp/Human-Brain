"""Learn recurrent affordance -> effect relations from solved fragment edits.

V2 consumes the frozen V1 fragment library and exactly two independently
selected one-edit V1 witnesses.  Each lesson is reduced to one authenticated,
role-normalized precondition/donor/postcondition receipt without executing any
unselected splice.  Two physical relation cells retain those receipts in lesson
order.  Five feature/effect cells and five action cells learn a generic
UNIQUE/NONE/MULTIPLE/MATCH/MISMATCH -> APPLY/STOP/ABSTAIN/COMMIT/REVISE policy.

Fresh construction is desired-free.  Starting from the authenticated learned
root, recurrent relation state physically races the current typed sites, only
the uniquely selected relation/site/donor is actuated, its structural effect is
physically checked, and the policy commits or revises before continuing.  The
root and the committed terminal program are frozen as semantic candidates;
desired outputs enter once in ``select`` and source is compiled only for one
unique winner.

Typed parsing, local ancestry extraction, lesson framing/order, visible-case
execution, token actuation, and compilation remain host scaffolding.  This is a
self-posed two-relation coding mechanism, not prompt parsing, arbitrary program
induction, public evaluation, or broad software capability evidence.
"""

from __future__ import annotations

import ast
import builtins
import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, replace
from hashlib import sha256
from typing import NamedTuple

import numpy as np
from experiments.r3_pilot import causal_reusable_program_fragment_library_v1 as fragments
from experiments.r3_pilot import common_repository_action_bank_v1 as action_bank
from experiments.r3_pilot import public_typed_output_cells as output_cells
from experiments.r3_pilot import public_typed_semantic_vector_race as semantic_race
from humanbrain import typed_program
from humanbrain.typed_program_types import TypedSignature

SCHEMA = "CAUSAL_RECURRENT_FRAGMENT_RELATION_POLICY_V2"
READY = f"{SCHEMA}_READY"
STOP_TRAINING = f"STOP_{SCHEMA}_TRAINING"
STOP_RELATION = f"STOP_{SCHEMA}_RELATION"
STOP_POLICY = f"STOP_{SCHEMA}_POLICY"
STOP_EFFECT = f"STOP_{SCHEMA}_EFFECT"
STOP_ACTUATION = f"STOP_{SCHEMA}_ACTUATION"
STOP_REPLAY = f"STOP_{SCHEMA}_REPLAY"
STOP_NO_AFFORDANCE = f"STOP_{SCHEMA}_NO_AFFORDANCE"
STOP_AMBIGUOUS_AFFORDANCE = f"STOP_{SCHEMA}_AMBIGUOUS_AFFORDANCE"
STOP_ROLLOUT_LIMIT = f"STOP_{SCHEMA}_ROLLOUT_LIMIT"
STOP_NO_TERMINAL = f"STOP_{SCHEMA}_NO_TERMINAL"

SELECTED = semantic_race.SELECTED
STOP_NO_EXACT = semantic_race.STOP_NO_EXACT
STOP_MULTIPLE_EXACT = semantic_race.STOP_MULTIPLE_EXACT

RELATION_COUNT = 2
RELATION_SIDES = 2
PRECONDITION = 0
POSTCONDITION = 1
DESCRIPTOR_ROUTES = 32
DESCRIPTOR_RADIX = 16
FEATURE_COUNT = 5
ACTION_COUNT = 5

FEATURE_UNIQUE = 0
FEATURE_NONE = 1
FEATURE_MULTIPLE = 2
FEATURE_EFFECT_MATCH = 3
FEATURE_EFFECT_MISMATCH = 4

ACTION_APPLY = 0
ACTION_STOP = 1
ACTION_ABSTAIN = 2
ACTION_COMMIT = 3
ACTION_REVISE = 4

ACTION_KEYS = (
    "action:apply",
    "action:stop",
    "action:abstain",
    "action:commit",
    "action:revise",
)
POLICY_TARGETS = (
    ACTION_APPLY,
    ACTION_STOP,
    ACTION_ABSTAIN,
    ACTION_COMMIT,
    ACTION_REVISE,
)

MAX_ROOTS = 2
MAX_COMMITTED_EDITS = 3
MAX_REVISIONS_PER_ROOT = 1
MAX_AFFORDANCE_RACES_PER_ROOT = 3
MAX_POLICY_RACES_PER_ROOT = 5
MAX_EFFECT_RACES_PER_ROOT = 2

TICK_DTYPE = np.dtype(np.uint8)
COUNT_DTYPE = np.dtype(np.uint16)
TICK_CAP = int(np.iinfo(TICK_DTYPE).max)

DESIRED_READS_DURING_BUILD = 0
SOURCE_MATERIALIZATIONS_DURING_BUILD = 0
UNSELECTED_SPLICE_EXECUTIONS = 0

CLAIM_BOUNDARY = (
    "two independently solved self-posed one-edit fragment relations retained "
    "as role-normalized pre/post cell state and recurrently composed through "
    "physical affordance, action, and effect races on fresh visible cases; "
    "typed parsing, local role extraction, two-lesson order, token actuation, "
    "visible execution, and compilation remain host scaffolding"
)

_CHECKPOINT_NAMESPACE = b"causal-recurrent-fragment-relation-policy-v2"
_RECEIPT_NAMESPACE = b"causal-recurrent-fragment-relation-receipt-v2"
_BUILD_NAMESPACE = b"causal-recurrent-fragment-relation-build-v2"
_WITNESS_NAMESPACE = b"causal-recurrent-fragment-relation-witness-v2"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\x00" + _canonical_json(value)).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    framed = np.ascontiguousarray(value)
    return sha256(memoryview(framed).cast("B")).hexdigest()


def _encoding_vector_sha256(
    vector: Sequence[output_cells.TypedOutputEncoding],
) -> str:
    return _digest(b"typed-output-vector", tuple(item.sha256 for item in vector))


def _descriptor_targets(root: str) -> tuple[int, ...]:
    if len(root) != 64:
        raise ValueError("role descriptor digest changed")
    return tuple(int(value, 16) for value in root[:DESCRIPTOR_ROUTES])


def _role_descriptor(
    program: typed_program.Program,
    site: typed_program.ProgramEditSite,
) -> tuple[str, tuple[int, ...]]:
    """Describe a local structural role without retaining an absolute path."""

    sites = {item.path: item for item in typed_program.program_edit_sites(program)}
    if site.path not in sites:
        raise ValueError("typed role site is not part of the program")
    ancestry: list[dict[str, object]] = []
    start = max(0, len(site.path) - 3)
    for depth in range(start, len(site.path)):
        parent = sites[site.path[:depth]]
        ancestry.append(
            {
                "parent_head": parent.head_production,
                "parent_result": parent.result_type_label,
                "parent_scope": parent.scope_binding_type_labels,
                "parent_is_binder": parent.is_binder,
                "child_role": site.path[depth],
            }
        )
    payload = {
        "site": fragments._site_payload(site),
        "site_head": site.head_production,
        "local_ancestry": tuple(ancestry),
    }
    root = sha256(_canonical_json(payload)).hexdigest()
    return root, _descriptor_targets(root)


def _relation_targets(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    cell_index: int,
    side: int,
) -> tuple[int, ...]:
    scores = (
        excitatory[cell_index, side].astype(np.int16)
        - inhibitory[cell_index, side].astype(np.int16)
    )
    winners: list[int] = []
    for route in range(DESCRIPTOR_ROUTES):
        row = scores[route]
        winner = int(np.argmax(row))
        if int(row[winner]) <= 0 or int(np.count_nonzero(row == row[winner])) != 1:
            raise RuntimeError("relation cell lost a unique descriptor target")
        winners.append(winner)
    return tuple(winners)


def _learn_relation_side(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    cell_index: int,
    side: int,
    targets: Sequence[int],
) -> tuple[int, int]:
    e_ticks = 0
    h_ticks = 0
    for route, winner in enumerate(targets):
        for target in range(DESCRIPTOR_RADIX):
            state = excitatory if target == winner else inhibitory
            before = int(state[cell_index, side, route, target])
            state[cell_index, side, route, target] = min(TICK_CAP, before + 1)
            delta = int(state[cell_index, side, route, target]) - before
            if target == winner:
                e_ticks += delta
            else:
                h_ticks += delta
    return e_ticks, h_ticks


def _learn_policy(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
) -> tuple[int, int]:
    e_ticks = 0
    h_ticks = 0
    for feature, winner in enumerate(POLICY_TARGETS):
        for action in range(ACTION_COUNT):
            state = excitatory if action == winner else inhibitory
            before = int(state[feature, action])
            state[feature, action] = min(TICK_CAP, before + 1)
            delta = int(state[feature, action]) - before
            if action == winner:
                e_ticks += delta
            else:
                h_ticks += delta
    return e_ticks, h_ticks


def _policy_winner(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    feature: int,
) -> int:
    scores = excitatory[feature].astype(np.int16) - inhibitory[feature].astype(
        np.int16
    )
    winner = int(np.argmax(scores))
    if int(scores[winner]) <= 0 or int(np.count_nonzero(scores == scores[winner])) != 1:
        raise RuntimeError("effect policy lost a unique action target")
    return winner


def _action_vector(action: int) -> tuple[output_cells.TypedOutputEncoding, ...]:
    return (
        output_cells.TypedOutputCellBank.encode(
            action,
            declared_type=output_cells.OutputType.INT,
        ),
    )


def _policy_race(
    excitatory: np.ndarray,
    inhibitory: np.ndarray,
    feature: int,
) -> tuple[int, semantic_race.TypedSemanticVectorRaceAudit]:
    winner = _policy_winner(excitatory, inhibitory, feature)
    race = semantic_race.select_exact_semantic_vector(
        ACTION_KEYS,
        tuple(_action_vector(action) for action in range(ACTION_COUNT)),
        _action_vector(winner),
    )
    if race.status != SELECTED or race.winner_key != ACTION_KEYS[winner]:
        raise RuntimeError("physical effect-policy race did not select its learned cell")
    return winner, race


class RelationPolicyFitStop(RuntimeError):
    def __init__(self, status: str, detail: str) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail


class FragmentRelationLesson(NamedTuple):
    frontier: fragments.ReusableProgramFragmentFrontierV1
    selection: fragments.FragmentCompositionSelection


class FragmentRelationReceipt(NamedTuple):
    cell_index: int
    lesson_frontier_sha256: str
    lesson_selection_race_sha256: str
    v1_relation_sha256: str
    base_program_sha256: str
    base_program_tokens: tuple[str, ...]
    base_root_fragment_cell: int
    result_program_sha256: str
    precondition_sha256: str
    precondition_targets: tuple[int, ...]
    donor_fragment_cell: int
    donor_descriptor_sha256: str
    donor_tokens: tuple[str, ...]
    postcondition_sha256: str
    postcondition_targets: tuple[int, ...]
    intervention_trace_sha256: str
    intervention_race_sha256: str
    receipt_sha256: str


class RelationPolicyTrainingWork(NamedTuple):
    solved_relation_races: int
    structural_explanation_attempts: int
    structural_explanation_matches: int
    intervention_races: int
    intervention_candidate_programs: int
    intervention_probe_executions: int
    unselected_task_output_executions: int
    learned_relation_cells: int
    feature_cells: int
    action_cells: int
    relation_excitatory_ticks: int
    relation_inhibitory_ticks: int
    policy_excitatory_ticks: int
    policy_inhibitory_ticks: int
    persistent_cell_state_bytes: int


class RelationPolicyState(NamedTuple):
    status: str
    library_checkpoint_sha256: str
    relation_count: int
    relation_receipt_sha256s: tuple[str, ...]
    base_program_sha256: str
    relation_excitatory_sha256: str
    relation_inhibitory_sha256: str
    relation_counts_sha256: str
    policy_excitatory_sha256: str
    policy_inhibitory_sha256: str
    work: RelationPolicyTrainingWork
    checkpoint_sha256: str


class RecurrentFragmentBuildWork(NamedTuple):
    root_rollouts: int
    root_replay_races: int
    donor_replay_races: int
    replay_selector_cell_steps: int
    affordance_races: int
    affordance_selector_cell_steps: int
    policy_races: int
    policy_selector_cell_steps: int
    effect_races: int
    effect_selector_cell_steps: int
    selected_actuations: int
    committed_edits: int
    revisions: int
    unselected_splice_executions: int
    terminal_program_count: int
    terminal_case_executions: int
    physical_output_encodings: int


class RecurrentFragmentBuildState(NamedTuple):
    status: str
    policy_checkpoint_sha256: str
    case_count: int
    terminal_program_count: int
    terminal_keys: tuple[str, ...]
    terminal_program_sha256s: tuple[str, ...]
    terminal_semantic_sha256s: tuple[str, ...]
    applied_relation_receipt_sha256s: tuple[str, ...]
    desired_reads_during_build: int
    source_materializations_during_build: int
    work: RecurrentFragmentBuildWork
    sha256: str


class RecurrentFragmentWitness(NamedTuple):
    program: typed_program.Program
    source: str
    applied_relation_receipt_sha256s: tuple[str, ...]
    sha256: str


class RecurrentFragmentSelection(NamedTuple):
    status: str
    winner_key: str | None
    exact_winner_keys: tuple[str, ...]
    exact_proof_count: int
    witness: RecurrentFragmentWitness | None
    materialized_witnesses: int
    race: semantic_race.TypedSemanticVectorRaceAudit | None


@dataclass(frozen=True)
class _DerivedRelation:
    lesson_frontier_sha256: str
    lesson_selection_race_sha256: str
    v1_relation_sha256: str
    base_program_sha256: str
    base_program_tokens: tuple[str, ...]
    base_root_fragment_cell: int
    result_program_sha256: str
    precondition_sha256: str
    precondition_targets: tuple[int, ...]
    donor_fragment_cell: int
    donor_descriptor_sha256: str
    donor_tokens: tuple[str, ...]
    postcondition_sha256: str
    postcondition_targets: tuple[int, ...]
    intervention_path: tuple[int, ...]
    intervention_trace_sha256: str
    intervention_race_sha256: str


@dataclass(frozen=True)
class _TerminalProgram:
    key: str
    tokens: tuple[str, ...]
    program_sha256: str
    physical_outputs: tuple[output_cells.TypedOutputEncoding, ...]
    applied_relation_receipt_sha256s: tuple[str, ...]


def _causal_gated_effect_vector() -> tuple[
    output_cells.TypedOutputEncoding, ...
]:
    encoded = fragments._encode_outputs((1, 1))
    if encoded is None:
        raise RuntimeError("causal intervention effect vector is unsupported")
    return encoded


def _trace_donor_activation_effect(
    program: typed_program.Program,
    donor_head_production: str,
    cases: tuple[tuple[object, ...], ...],
) -> tuple[
    tuple[output_cells.TypedOutputEncoding, ...],
    str,
    int,
]:
    """Observe generic donor activation under singleton pass/fail probes.

    The runtime callable is derived from the selected donor expression's lowered
    AST.  The intervention rule never names an operator or an absolute path: it
    counts that learned callable while replaying each visible list item alone.
    """

    donor_sites = tuple(
        site
        for site in typed_program.program_edit_sites(program)
        if site.head_production == donor_head_production
    )
    if len(donor_sites) != 1:
        raise RelationPolicyFitStop(
            STOP_RELATION,
            "learned donor production has no unique activation cell",
        )
    expression = typed_program._validate_program(program.tokens, program.signature)
    selected = expression
    for part in donor_sites[0].path:
        if part >= len(selected.children):
            raise RelationPolicyFitStop(
                STOP_RELATION, "intervention path changed during trace observation"
            )
        selected = selected.children[part]
    lowered_selected = typed_program._lower_expression(
        selected,
        ("trace_item", "trace_outer"),
        typed_program._LoweringContext(),
    )
    if not (
        isinstance(lowered_selected, ast.Call)
        and isinstance(lowered_selected.func, ast.Name)
    ):
        raise RelationPolicyFitStop(
            STOP_RELATION,
            "learned donor has no generic callable activation surface",
        )
    runtime_symbol = lowered_selected.func.id
    original = getattr(builtins, runtime_symbol, None)
    if not callable(original):
        raise RelationPolicyFitStop(
            STOP_RELATION, "learned donor callable is unavailable to the evaluator"
        )

    environment = tuple(
        parameter.name for parameter in reversed(program.signature.parameters)
    )
    module = ast.Module(
        body=[
            ast.FunctionDef(
                name=program.signature.entry_point,
                args=ast.arguments(
                    posonlyargs=[],
                    args=[
                        ast.arg(arg=parameter.name)
                        for parameter in program.signature.parameters
                    ],
                    vararg=None,
                    kwonlyargs=[],
                    kw_defaults=[],
                    kwarg=None,
                    defaults=[],
                ),
                body=[
                    ast.Return(
                        value=typed_program._lower_expression(
                            expression,
                            environment,
                            typed_program._LoweringContext(),
                        )
                    )
                ],
                decorator_list=[],
                returns=None,
                type_comment=None,
            )
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    activations = 0

    def observe(*args: object, **kwargs: object) -> object:
        nonlocal activations
        activations += 1
        return original(*args, **kwargs)

    traced_builtins = dict(vars(builtins))
    traced_builtins[runtime_symbol] = observe
    global_namespace: dict[str, object] = {"__builtins__": traced_builtins}
    local_namespace: dict[str, object] = {}
    executable = builtins.compile(module, "<fragment-intervention>", "exec")
    exec(executable, global_namespace, local_namespace)
    function = local_namespace[program.signature.entry_point]
    activation_trace: list[int] = []
    probe_executions = 0
    for case in cases:
        if len(case) != 1 or type(case[0]) is not list:
            raise RelationPolicyFitStop(
                STOP_RELATION,
                "generic activation intervention currently requires visible List cases",
            )
        for item in case[0]:
            before = activations
            value = function([item])  # type: ignore[operator]
            if type(value) is not int:
                raise RelationPolicyFitStop(
                    STOP_RELATION, "intervention probe did not return an exact Int"
                )
            activation_trace.append(activations - before)
            probe_executions += 1
    if not activation_trace:
        raise RelationPolicyFitStop(
            STOP_RELATION, "intervention lesson contains no visible list items"
        )
    effect = (
        int(any(value > 0 for value in activation_trace)),
        int(any(value == 0 for value in activation_trace)),
    )
    encoded = fragments._encode_outputs(effect)
    if encoded is None:
        raise RuntimeError("activation intervention encoding is unsupported")
    trace_sha256 = _digest(
        b"generic-donor-activation-effect",
        {
            "program": program.sha256,
            "donor_activation_counts": tuple(activation_trace),
            "effect": effect,
        },
    )
    return encoded, trace_sha256, probe_executions


def _derive_relation(
    library: fragments.FrozenReusableProgramFragmentLibraryV1,
    lesson: FragmentRelationLesson,
    *,
    activation_head_production: str | None = None,
) -> tuple[
    _DerivedRelation,
    int,
    int,
    int,
    semantic_race.TypedSemanticVectorRaceAudit,
]:
    frontier = lesson.frontier
    selection = lesson.selection
    frontier._authenticate()
    if frontier._library is not library:
        raise RelationPolicyFitStop(STOP_TRAINING, "relation lesson changed library")
    if selection.race is None:
        raise RelationPolicyFitStop(
            STOP_TRAINING, "relation lesson has no physical V1 semantic race"
        )
    if selection.status == SELECTED:
        if (
            selection.winner_key is None
            or selection.witness is None
            or selection.exact_proof_count != 1
        ):
            raise RelationPolicyFitStop(
                STOP_TRAINING, "selected V1 lesson lost its unique witness"
            )
        recipe_keys = (selection.winner_key,)
    elif selection.status == STOP_MULTIPLE_EXACT:
        if selection.winner_key is not None or len(selection.exact_winner_keys) < 2:
            raise RelationPolicyFitStop(
                STOP_TRAINING, "ambiguous V1 lesson lost its exact tier"
            )
        recipe_keys = selection.exact_winner_keys
    else:
        raise RelationPolicyFitStop(
            STOP_TRAINING, "relation lesson has no exact V1 program tier"
        )
    recipe_by_key = {item.key: item for item in frontier._recipes}
    try:
        recipes = tuple(recipe_by_key[key] for key in recipe_keys)
    except KeyError as error:
        raise RelationPolicyFitStop(
            STOP_TRAINING, "relation lesson lost a frozen V1 recipe"
        ) from error
    if selection.status == SELECTED:
        recipe = recipes[0]
        if (
            recipe.program_sha256 != selection.witness.program.sha256
            or recipe.relation_sha256 != selection.witness.relation_sha256
        ):
            raise RelationPolicyFitStop(
                STOP_TRAINING, "relation lesson changed its selected V1 recipe"
            )

    attempts = 0
    derived_candidates: list[_DerivedRelation] = []
    for recipe in recipes:
        matches: list[_DerivedRelation] = []
        for base_record in library._program_records:
            base = typed_program.Program(
                base_record.program_tokens,
                library._signature,
            )
            for site in typed_program.program_edit_sites(base):
                if not site.path:
                    continue
                for donor in library._prototype_records:
                    if not any(
                        origin.program_index != base_record.program_index
                        for origin in donor.origins
                    ):
                        continue
                    attempts += 1
                    if (
                        donor.tokens == site.tokens
                        or donor.result_type_label != site.result_type_label
                        or donor.scope_binding_type_labels
                        != site.scope_binding_type_labels
                        or donor.is_binder != site.is_binder
                    ):
                        continue
                    try:
                        tokens = fragments._splice_tokens(
                            base,
                            site.path,
                            donor.tokens,
                        )
                    except (typed_program.ProgramFormatError, TypeError, ValueError):
                        continue
                    program = typed_program.Program(tokens, library._signature)
                    relation_sha256 = fragments._digest(
                        b"fragment-relation",
                        {
                            "checkpoint": library.checkpoint_sha256,
                            "base_root_cell": base_record.root_fragment_cell,
                            "base_selection": base_record.selection_race_sha256,
                            "edit_path": site.path,
                            "donor_cell": donor.cell_index,
                            "donor_descriptor": donor.descriptor_sha256,
                            "program": program.sha256,
                        },
                    )
                    if (
                        program.sha256 != recipe.program_sha256
                        or relation_sha256 != recipe.relation_sha256
                    ):
                        continue
                    post_site = next(
                        item
                        for item in typed_program.program_edit_sites(program)
                        if item.path == site.path
                    )
                    pre_root, pre_targets = _role_descriptor(base, site)
                    post_root, post_targets = _role_descriptor(program, post_site)
                    matches.append(
                        _DerivedRelation(
                            lesson_frontier_sha256=frontier.state.sha256,
                            lesson_selection_race_sha256=selection.race.sha256,
                            v1_relation_sha256=relation_sha256,
                            base_program_sha256=base.sha256,
                            base_program_tokens=base.tokens,
                            base_root_fragment_cell=base_record.root_fragment_cell,
                            result_program_sha256=program.sha256,
                            precondition_sha256=pre_root,
                            precondition_targets=pre_targets,
                            donor_fragment_cell=donor.cell_index,
                            donor_descriptor_sha256=donor.descriptor_sha256,
                            donor_tokens=donor.tokens,
                            postcondition_sha256=post_root,
                            postcondition_targets=post_targets,
                            intervention_path=site.path,
                            intervention_trace_sha256="",
                            intervention_race_sha256="",
                        )
                    )
        if len(matches) != 1:
            raise RelationPolicyFitStop(
                STOP_RELATION,
                "exact V1 recipe did not have one authenticated explanation",
            )
        derived_candidates.append(matches[0])

    trace_vectors: list[tuple[output_cells.TypedOutputEncoding, ...]] = []
    traced_candidates: list[_DerivedRelation] = []
    probe_executions = 0
    for item in derived_candidates:
        program = typed_program.Program(
            fragments._splice_tokens(
                typed_program.Program(item.base_program_tokens, library._signature),
                item.intervention_path,
                item.donor_tokens,
            ),
            library._signature,
        )
        vector, trace_sha256, probes = _trace_donor_activation_effect(
            program,
            (
                item.donor_tokens[0]
                if activation_head_production is None
                else activation_head_production
            ),
            frontier._cases,
        )
        trace_vectors.append(vector)
        traced_candidates.append(
            replace(item, intervention_trace_sha256=trace_sha256)
        )
        probe_executions += probes
    target = (
        trace_vectors[0]
        if len(trace_vectors) == 1
        else _causal_gated_effect_vector()
    )
    intervention_race = semantic_race.select_exact_semantic_vector(
        recipe_keys,
        tuple(trace_vectors),
        target,
    )
    if intervention_race.status != SELECTED or intervention_race.winner_key is None:
        raise RelationPolicyFitStop(
            STOP_RELATION,
            "generic activation intervention did not select one causal relation",
        )
    winner_index = recipe_keys.index(intervention_race.winner_key)
    winner = replace(
        traced_candidates[winner_index],
        intervention_race_sha256=intervention_race.sha256,
    )
    return (
        winner,
        attempts,
        len(derived_candidates),
        probe_executions,
        intervention_race,
    )


def _receipt_sha256(record: _DerivedRelation) -> str:
    return _digest(
        _RECEIPT_NAMESPACE,
        {
            "lesson_frontier": record.lesson_frontier_sha256,
            "lesson_selection": record.lesson_selection_race_sha256,
            "v1_relation": record.v1_relation_sha256,
            "base_program": record.base_program_sha256,
            "base_root_cell": record.base_root_fragment_cell,
            "result_program": record.result_program_sha256,
            "precondition": record.precondition_sha256,
            "pre_targets": record.precondition_targets,
            "donor_cell": record.donor_fragment_cell,
            "donor_descriptor": record.donor_descriptor_sha256,
            "postcondition": record.postcondition_sha256,
            "post_targets": record.postcondition_targets,
            "intervention_trace": record.intervention_trace_sha256,
            "intervention_race": record.intervention_race_sha256,
        },
    )


def _checkpoint_payload(
    library_checkpoint_sha256: str,
    records: tuple[FragmentRelationReceipt, ...],
    relation_excitatory: np.ndarray,
    relation_inhibitory: np.ndarray,
    relation_counts: np.ndarray,
    policy_excitatory: np.ndarray,
    policy_inhibitory: np.ndarray,
    work: RelationPolicyTrainingWork,
) -> dict[str, object]:
    return {
        "library": library_checkpoint_sha256,
        "records": records,
        "relation_excitatory": _array_sha256(relation_excitatory),
        "relation_inhibitory": _array_sha256(relation_inhibitory),
        "relation_counts": _array_sha256(relation_counts),
        "policy_excitatory": _array_sha256(policy_excitatory),
        "policy_inhibitory": _array_sha256(policy_inhibitory),
        "work": work,
    }


class FrozenCausalRecurrentFragmentRelationPolicyV2:
    """Frozen two-relation recurrent cell policy."""

    __slots__ = (
        "_library",
        "_policy_excitatory",
        "_policy_inhibitory",
        "_records",
        "_relation_counts",
        "_relation_excitatory",
        "_relation_inhibitory",
        "_state",
    )

    def __init__(self) -> None:
        raise TypeError("use fit_causal_recurrent_fragment_relation_policy_v2")

    @property
    def state(self) -> RelationPolicyState:
        self.authenticate()
        return self._state

    @property
    def relation_receipts(self) -> tuple[FragmentRelationReceipt, ...]:
        self.authenticate()
        return self._records

    @property
    def checkpoint_sha256(self) -> str:
        return self._state.checkpoint_sha256

    def authenticate(self) -> None:
        self._library.authenticate()
        state = self._state
        if (
            self._relation_excitatory.shape
            != (
                RELATION_COUNT,
                RELATION_SIDES,
                DESCRIPTOR_ROUTES,
                DESCRIPTOR_RADIX,
            )
            or self._relation_inhibitory.shape != self._relation_excitatory.shape
            or self._relation_counts.shape != (RELATION_COUNT,)
            or self._policy_excitatory.shape != (FEATURE_COUNT, ACTION_COUNT)
            or self._policy_inhibitory.shape != self._policy_excitatory.shape
            or self._relation_excitatory.dtype != TICK_DTYPE
            or self._relation_inhibitory.dtype != TICK_DTYPE
            or self._relation_counts.dtype != COUNT_DTYPE
            or self._policy_excitatory.dtype != TICK_DTYPE
            or self._policy_inhibitory.dtype != TICK_DTYPE
            or any(
                array.flags.writeable
                for array in (
                    self._relation_excitatory,
                    self._relation_inhibitory,
                    self._relation_counts,
                    self._policy_excitatory,
                    self._policy_inhibitory,
                )
            )
        ):
            raise RuntimeError("recurrent relation cell geometry changed")
        if len(self._records) != RELATION_COUNT:
            raise RuntimeError("recurrent relation count changed")
        for index, record in enumerate(self._records):
            if (
                record.cell_index != index
                or int(self._relation_counts[index]) != 1
                or _relation_targets(
                    self._relation_excitatory,
                    self._relation_inhibitory,
                    index,
                    PRECONDITION,
                )
                != record.precondition_targets
                or _relation_targets(
                    self._relation_excitatory,
                    self._relation_inhibitory,
                    index,
                    POSTCONDITION,
                )
                != record.postcondition_targets
            ):
                raise RuntimeError("recurrent relation receipt changed")
            base = typed_program.Program(
                record.base_program_tokens,
                self._library._signature,
            )
            if base.sha256 != record.base_program_sha256:
                raise RuntimeError("relation root program changed")
            role_sites = tuple(
                site
                for site in typed_program.program_edit_sites(base)
                if site.path
                and _role_descriptor(base, site)[0] == record.precondition_sha256
            )
            if len(role_sites) != 1:
                raise RuntimeError("relation precondition is not a unique local role")
            program = typed_program.Program(
                fragments._splice_tokens(base, role_sites[0].path, record.donor_tokens),
                self._library._signature,
            )
            post_site = next(
                item
                for item in typed_program.program_edit_sites(program)
                if item.path == role_sites[0].path
            )
            if (
                program.sha256 != record.result_program_sha256
                or _role_descriptor(program, post_site)[0]
                != record.postcondition_sha256
            ):
                raise RuntimeError("relation postcondition authentication failed")
        for feature, expected in enumerate(POLICY_TARGETS):
            if _policy_winner(
                self._policy_excitatory,
                self._policy_inhibitory,
                feature,
            ) != expected:
                raise RuntimeError("effect policy mapping changed")
        payload = _checkpoint_payload(
            self._library.checkpoint_sha256,
            self._records,
            self._relation_excitatory,
            self._relation_inhibitory,
            self._relation_counts,
            self._policy_excitatory,
            self._policy_inhibitory,
            state.work,
        )
        checkpoint = _digest(_CHECKPOINT_NAMESPACE, payload)
        if (
            state.status != READY
            or state.library_checkpoint_sha256 != self._library.checkpoint_sha256
            or state.relation_count != RELATION_COUNT
            or state.relation_receipt_sha256s
            != tuple(record.receipt_sha256 for record in self._records)
            or state.base_program_sha256 != self._records[0].base_program_sha256
            or state.relation_excitatory_sha256
            != _array_sha256(self._relation_excitatory)
            or state.relation_inhibitory_sha256
            != _array_sha256(self._relation_inhibitory)
            or state.relation_counts_sha256 != _array_sha256(self._relation_counts)
            or state.policy_excitatory_sha256
            != _array_sha256(self._policy_excitatory)
            or state.policy_inhibitory_sha256
            != _array_sha256(self._policy_inhibitory)
            or state.checkpoint_sha256 != checkpoint
        ):
            raise RuntimeError("recurrent relation checkpoint changed")

    def build(
        self,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        *,
        relation_cell_gate: bool = True,
        policy_cell_gate: bool = True,
        effect_cell_gate: bool = True,
        actuation_gate: bool = True,
        replay_cell_gate: bool = True,
    ) -> RecurrentFragmentRelationFrontierV2:
        return RecurrentFragmentRelationFrontierV2._build(
            self,
            signature,
            cases,
            relation_cell_gate=relation_cell_gate,
            policy_cell_gate=policy_cell_gate,
            effect_cell_gate=effect_cell_gate,
            actuation_gate=actuation_gate,
            replay_cell_gate=replay_cell_gate,
        )


def fit_causal_recurrent_fragment_relation_policy_v2(
    library: fragments.FrozenReusableProgramFragmentLibraryV1,
    lessons: Sequence[FragmentRelationLesson],
) -> FrozenCausalRecurrentFragmentRelationPolicyV2:
    """Learn two ordered relation cells and a generic causal effect policy."""

    if type(library) is not fragments.FrozenReusableProgramFragmentLibraryV1:
        raise TypeError("exact frozen V1 fragment library required")
    library.authenticate()
    frozen_lessons = tuple(lessons)
    if len(frozen_lessons) != RELATION_COUNT or any(
        type(item) is not FragmentRelationLesson for item in frozen_lessons
    ):
        raise ValueError("V2 requires exactly two FragmentRelationLesson values")

    derived_by_lesson: list[_DerivedRelation | None] = [None] * RELATION_COUNT
    attempts = 0
    intervention_candidate_programs = 0
    intervention_probe_executions = 0
    intervention_races: list[semantic_race.TypedSemanticVectorRaceAudit] = []
    activation_head_production: str | None = None
    solve_order = tuple(
        sorted(
            enumerate(frozen_lessons),
            key=lambda item: (item[1].selection.status != SELECTED, item[0]),
        )
    )
    for lesson_index, lesson in solve_order:
        (
            item,
            item_attempts,
            item_candidates,
            item_probes,
            intervention_race,
        ) = _derive_relation(
            library,
            lesson,
            activation_head_production=activation_head_production,
        )
        derived_by_lesson[lesson_index] = item
        if activation_head_production is None:
            activation_head_production = item.donor_tokens[0]
        attempts += item_attempts
        intervention_candidate_programs += item_candidates
        intervention_probe_executions += item_probes
        intervention_races.append(intervention_race)
    if any(item is None for item in derived_by_lesson):
        raise RuntimeError("relation solve order did not cover both lessons")
    derived = tuple(item for item in derived_by_lesson if item is not None)
    if (
        len({item.lesson_selection_race_sha256 for item in derived}) != 2
        or len({item.v1_relation_sha256 for item in derived}) != 2
        or len({item.precondition_sha256 for item in derived}) != 2
        or len({item.base_program_sha256 for item in derived}) != 1
    ):
        raise RelationPolicyFitStop(
            STOP_RELATION,
            "lessons must be two distinct relations over one authenticated root",
        )

    relation_excitatory = np.zeros(
        (
            RELATION_COUNT,
            RELATION_SIDES,
            DESCRIPTOR_ROUTES,
            DESCRIPTOR_RADIX,
        ),
        dtype=TICK_DTYPE,
    )
    relation_inhibitory = np.zeros_like(relation_excitatory)
    relation_counts = np.zeros((RELATION_COUNT,), dtype=COUNT_DTYPE)
    policy_excitatory = np.zeros((FEATURE_COUNT, ACTION_COUNT), dtype=TICK_DTYPE)
    policy_inhibitory = np.zeros_like(policy_excitatory)
    relation_e_ticks = 0
    relation_h_ticks = 0
    records: list[FragmentRelationReceipt] = []
    for cell_index, item in enumerate(derived):
        for side, targets in (
            (PRECONDITION, item.precondition_targets),
            (POSTCONDITION, item.postcondition_targets),
        ):
            learned_e, learned_h = _learn_relation_side(
                relation_excitatory,
                relation_inhibitory,
                cell_index,
                side,
                targets,
            )
            relation_e_ticks += learned_e
            relation_h_ticks += learned_h
        relation_counts[cell_index] = 1
        records.append(
            FragmentRelationReceipt(
                cell_index=cell_index,
                lesson_frontier_sha256=item.lesson_frontier_sha256,
                lesson_selection_race_sha256=item.lesson_selection_race_sha256,
                v1_relation_sha256=item.v1_relation_sha256,
                base_program_sha256=item.base_program_sha256,
                base_program_tokens=item.base_program_tokens,
                base_root_fragment_cell=item.base_root_fragment_cell,
                result_program_sha256=item.result_program_sha256,
                precondition_sha256=item.precondition_sha256,
                precondition_targets=item.precondition_targets,
                donor_fragment_cell=item.donor_fragment_cell,
                donor_descriptor_sha256=item.donor_descriptor_sha256,
                donor_tokens=item.donor_tokens,
                postcondition_sha256=item.postcondition_sha256,
                postcondition_targets=item.postcondition_targets,
                intervention_trace_sha256=item.intervention_trace_sha256,
                intervention_race_sha256=item.intervention_race_sha256,
                receipt_sha256=_receipt_sha256(item),
            )
        )
    policy_e_ticks, policy_h_ticks = _learn_policy(
        policy_excitatory,
        policy_inhibitory,
    )
    arrays = (
        relation_excitatory,
        relation_inhibitory,
        relation_counts,
        policy_excitatory,
        policy_inhibitory,
    )
    for array in arrays:
        array.setflags(write=False)
    persistent_bytes = sum(array.nbytes for array in arrays)
    work = RelationPolicyTrainingWork(
        solved_relation_races=RELATION_COUNT,
        structural_explanation_attempts=attempts,
        structural_explanation_matches=RELATION_COUNT,
        intervention_races=len(intervention_races),
        intervention_candidate_programs=intervention_candidate_programs,
        intervention_probe_executions=intervention_probe_executions,
        unselected_task_output_executions=UNSELECTED_SPLICE_EXECUTIONS,
        learned_relation_cells=RELATION_COUNT,
        feature_cells=FEATURE_COUNT,
        action_cells=ACTION_COUNT,
        relation_excitatory_ticks=relation_e_ticks,
        relation_inhibitory_ticks=relation_h_ticks,
        policy_excitatory_ticks=policy_e_ticks,
        policy_inhibitory_ticks=policy_h_ticks,
        persistent_cell_state_bytes=persistent_bytes,
    )
    frozen_records = tuple(records)
    checkpoint = _digest(
        _CHECKPOINT_NAMESPACE,
        _checkpoint_payload(
            library.checkpoint_sha256,
            frozen_records,
            relation_excitatory,
            relation_inhibitory,
            relation_counts,
            policy_excitatory,
            policy_inhibitory,
            work,
        ),
    )
    state = RelationPolicyState(
        status=READY,
        library_checkpoint_sha256=library.checkpoint_sha256,
        relation_count=RELATION_COUNT,
        relation_receipt_sha256s=tuple(
            record.receipt_sha256 for record in frozen_records
        ),
        base_program_sha256=frozen_records[0].base_program_sha256,
        relation_excitatory_sha256=_array_sha256(relation_excitatory),
        relation_inhibitory_sha256=_array_sha256(relation_inhibitory),
        relation_counts_sha256=_array_sha256(relation_counts),
        policy_excitatory_sha256=_array_sha256(policy_excitatory),
        policy_inhibitory_sha256=_array_sha256(policy_inhibitory),
        work=work,
        checkpoint_sha256=checkpoint,
    )
    policy = object.__new__(FrozenCausalRecurrentFragmentRelationPolicyV2)
    policy._library = library
    policy._records = frozen_records
    policy._relation_excitatory = relation_excitatory
    policy._relation_inhibitory = relation_inhibitory
    policy._relation_counts = relation_counts
    policy._policy_excitatory = policy_excitatory
    policy._policy_inhibitory = policy_inhibitory
    policy._state = state
    policy.authenticate()
    return policy


def _build_state_sha256(state: RecurrentFragmentBuildState) -> str:
    return _digest(
        _BUILD_NAMESPACE,
        {
            "status": state.status,
            "policy": state.policy_checkpoint_sha256,
            "case_count": state.case_count,
            "terminal_count": state.terminal_program_count,
            "terminal_keys": state.terminal_keys,
            "terminal_programs": state.terminal_program_sha256s,
            "terminal_semantics": state.terminal_semantic_sha256s,
            "applied_relations": state.applied_relation_receipt_sha256s,
            "desired_reads": state.desired_reads_during_build,
            "source_materializations": state.source_materializations_during_build,
            "work": state.work,
        },
    )


class RecurrentFragmentRelationFrontierV2:
    """Desired-free recurrent rollout with semantic and repository frontiers."""

    __slots__ = (
        "_action_bank",
        "_cases",
        "_materialized",
        "_policy",
        "_repository_terminals",
        "_signature",
        "_source_materialization_count",
        "_state",
        "_terminals",
    )

    def __init__(self) -> None:
        raise TypeError("use FrozenCausalRecurrentFragmentRelationPolicyV2.build")

    @classmethod
    def _build(
        cls,
        policy: FrozenCausalRecurrentFragmentRelationPolicyV2,
        signature: TypedSignature,
        cases: Iterable[tuple[object, ...]],
        *,
        relation_cell_gate: bool,
        policy_cell_gate: bool,
        effect_cell_gate: bool,
        actuation_gate: bool,
        replay_cell_gate: bool,
    ) -> RecurrentFragmentRelationFrontierV2:
        policy.authenticate()
        gates = (
            relation_cell_gate,
            policy_cell_gate,
            effect_cell_gate,
            actuation_gate,
            replay_cell_gate,
        )
        if any(type(gate) is not bool for gate in gates):
            raise TypeError("all recurrent relation gates must be exact booleans")
        if signature != policy._library._signature:
            raise ValueError("fresh recurrent signature changed")
        frozen_cases = tuple(cases)
        if not frozen_cases:
            raise ValueError("fresh recurrent rollout requires visible cases")
        if any(len(case) != len(signature.parameters) for case in frozen_cases):
            raise ValueError("fresh cases do not match typed signature")

        replay_races: list[semantic_race.TypedSemanticVectorRaceAudit] = []
        affordance_races: list[semantic_race.TypedSemanticVectorRaceAudit] = []
        policy_races: list[semantic_race.TypedSemanticVectorRaceAudit] = []
        effect_races: list[semantic_race.TypedSemanticVectorRaceAudit] = []
        root_replays = 0
        donor_replays = 0
        selected_actuations = 0
        committed_edits = 0
        revisions = 0
        applied: list[str] = []
        committed_prefixes: list[
            tuple[typed_program.Program, tuple[str, ...]]
        ] = []
        terminals: tuple[_TerminalProgram, ...] = ()
        repository_terminals: tuple[_TerminalProgram, ...] = ()

        status = READY
        if not replay_cell_gate:
            status = STOP_REPLAY
        elif not relation_cell_gate:
            status = STOP_RELATION
        elif not policy_cell_gate:
            status = STOP_POLICY

        root_record = policy._records[0]
        root = typed_program.Program(root_record.base_program_tokens, signature)
        program = root
        if status == READY:
            replay_cells = (
                root_record.base_root_fragment_cell,
                *(record.donor_fragment_cell for record in policy._records),
            )
            for position, cell_index in enumerate(replay_cells):
                prototype = policy._library._prototype_records[cell_index]
                race = fragments._recall_prototype(
                    policy._library._excitatory,
                    policy._library._inhibitory,
                    prototype.descriptor_targets,
                )
                if (
                    race is None
                    or race.status != SELECTED
                    or race.winner_key != f"fragment-cell:{cell_index}"
                ):
                    status = STOP_REPLAY
                    break
                replay_races.append(race)
                if position == 0:
                    root_replays += 1
                else:
                    donor_replays += 1

        if status == READY:
            for record in policy._records:
                if (
                    len(affordance_races) >= MAX_AFFORDANCE_RACES_PER_ROOT
                    or len(policy_races) >= MAX_POLICY_RACES_PER_ROOT
                    or len(effect_races) >= MAX_EFFECT_RACES_PER_ROOT
                    or committed_edits >= MAX_COMMITTED_EDITS
                    or revisions > MAX_REVISIONS_PER_ROOT
                ):
                    status = STOP_ROLLOUT_LIMIT
                    break
                site_records: list[
                    tuple[typed_program.ProgramEditSite, str, tuple[int, ...]]
                ] = []
                for site in typed_program.program_edit_sites(program):
                    if not site.path:
                        continue
                    role_sha, role_targets = _role_descriptor(program, site)
                    site_records.append((site, role_sha, role_targets))
                candidate_keys = tuple(
                    "affordance:"
                    + _digest(
                        b"role-site",
                        {
                            "program": program.sha256,
                            "role": role_sha,
                            "site_identity": site.path,
                            "donor_cell": record.donor_fragment_cell,
                        },
                    )
                    for site, role_sha, _ in site_records
                )
                race = semantic_race.select_exact_semantic_vector(
                    candidate_keys,
                    tuple(
                        fragments._encode_descriptor(targets)
                        for _, _, targets in site_records
                    ),
                    fragments._encode_descriptor(
                        _relation_targets(
                            policy._relation_excitatory,
                            policy._relation_inhibitory,
                            record.cell_index,
                            PRECONDITION,
                        )
                    ),
                )
                affordance_races.append(race)
                if race.status == STOP_NO_EXACT:
                    action, action_race = _policy_race(
                        policy._policy_excitatory,
                        policy._policy_inhibitory,
                        FEATURE_NONE,
                    )
                    policy_races.append(action_race)
                    if action != ACTION_STOP:
                        raise RuntimeError("missing affordance did not activate STOP")
                    status = STOP_NO_AFFORDANCE
                    break
                if race.status == STOP_MULTIPLE_EXACT:
                    action, action_race = _policy_race(
                        policy._policy_excitatory,
                        policy._policy_inhibitory,
                        FEATURE_MULTIPLE,
                    )
                    policy_races.append(action_race)
                    if action != ACTION_ABSTAIN:
                        raise RuntimeError("ambiguous affordance did not abstain")
                    status = STOP_AMBIGUOUS_AFFORDANCE
                    break
                if race.status != SELECTED or race.winner_key is None:
                    raise RuntimeError("affordance race returned unsupported state")
                action, action_race = _policy_race(
                    policy._policy_excitatory,
                    policy._policy_inhibitory,
                    FEATURE_UNIQUE,
                )
                policy_races.append(action_race)
                if action != ACTION_APPLY:
                    raise RuntimeError("unique affordance did not activate APPLY")
                if not actuation_gate:
                    status = STOP_ACTUATION
                    break
                selected_site = site_records[candidate_keys.index(race.winner_key)][0]
                before = program
                program = typed_program.Program(
                    fragments._splice_tokens(
                        before,
                        selected_site.path,
                        record.donor_tokens,
                    ),
                    signature,
                )
                selected_actuations += 1
                effect_site = next(
                    item
                    for item in typed_program.program_edit_sites(program)
                    if item.path == selected_site.path
                )
                _, actual_effect_targets = _role_descriptor(program, effect_site)
                expected_effect_targets = _relation_targets(
                    policy._relation_excitatory,
                    policy._relation_inhibitory,
                    record.cell_index,
                    POSTCONDITION,
                )
                observed_candidate = (
                    expected_effect_targets
                    if effect_cell_gate
                    else record.precondition_targets
                )
                effect_race = semantic_race.select_exact_semantic_vector(
                    (f"effect-cell:{record.cell_index}",),
                    (fragments._encode_descriptor(observed_candidate),),
                    fragments._encode_descriptor(actual_effect_targets),
                )
                effect_races.append(effect_race)
                if effect_race.status == SELECTED:
                    effect_feature = FEATURE_EFFECT_MATCH
                elif effect_race.status == STOP_NO_EXACT:
                    effect_feature = FEATURE_EFFECT_MISMATCH
                else:
                    raise RuntimeError("effect race returned unsupported state")
                effect_action, effect_action_race = _policy_race(
                    policy._policy_excitatory,
                    policy._policy_inhibitory,
                    effect_feature,
                )
                policy_races.append(effect_action_race)
                if effect_action == ACTION_COMMIT:
                    committed_edits += 1
                    applied.append(record.receipt_sha256)
                    committed_prefixes.append((program, tuple(applied)))
                elif effect_action == ACTION_REVISE:
                    program = before
                    revisions += 1
                    status = STOP_EFFECT
                    break
                else:
                    raise RuntimeError("effect policy returned unsupported action")

        if status == READY:
            terminal_action, terminal_race = _policy_race(
                policy._policy_excitatory,
                policy._policy_inhibitory,
                FEATURE_NONE,
            )
            policy_races.append(terminal_race)
            if terminal_action != ACTION_STOP:
                raise RuntimeError("completed recurrent rollout did not stop")
            if (
                len(applied) != RELATION_COUNT
                or selected_actuations != RELATION_COUNT
                or committed_edits != RELATION_COUNT
                or revisions != 0
                or MAX_ROOTS < 1
                or len(affordance_races) > MAX_AFFORDANCE_RACES_PER_ROOT
                or len(policy_races) > MAX_POLICY_RACES_PER_ROOT
                or len(effect_races) > MAX_EFFECT_RACES_PER_ROOT
            ):
                status = STOP_ROLLOUT_LIMIT

        terminal_case_executions = 0
        physical_output_encodings = 0
        if status == READY:
            semantic_specs = (
                (root, ()),
                (program, tuple(applied)),
            )
            repository_specs = tuple(committed_prefixes)
            unique_specs = (*semantic_specs, repository_specs[0])
            built: dict[tuple[str, tuple[str, ...]], _TerminalProgram] = {}
            for terminal_program, terminal_relations in unique_specs:
                try:
                    values = fragments._execute_without_rendering(
                        terminal_program.tokens,
                        signature,
                        frozen_cases,
                    )
                except (ArithmeticError, RuntimeError, TypeError, ValueError):
                    status = STOP_NO_TERMINAL
                    built = {}
                    break
                terminal_case_executions += len(frozen_cases)
                physical_outputs = fragments._encode_outputs(values)
                if physical_outputs is None:
                    status = STOP_NO_TERMINAL
                    built = {}
                    break
                physical_output_encodings += len(physical_outputs)
                key = "terminal:" + _digest(
                    b"recurrent-terminal",
                    {
                        "program": terminal_program.sha256,
                        "relations": terminal_relations,
                    },
                )
                built[(terminal_program.sha256, terminal_relations)] = _TerminalProgram(
                    key=key,
                    tokens=terminal_program.tokens,
                    program_sha256=terminal_program.sha256,
                    physical_outputs=physical_outputs,
                    applied_relation_receipt_sha256s=terminal_relations,
                )
            if status == READY:
                terminals = tuple(
                    built[(item.sha256, relations)]
                    for item, relations in semantic_specs
                )
                repository_terminals = tuple(
                    built.get(
                        (item.sha256, relations),
                        built[(program.sha256, tuple(applied))],
                    )
                    for item, relations in repository_specs
                )
            if (
                len(terminals) != 2
                or len({item.key for item in terminals}) != 2
                or len(repository_terminals) != action_bank.CANDIDATE_COUNT
                or len({item.key for item in repository_terminals})
                != action_bank.CANDIDATE_COUNT
            ):
                status = STOP_NO_TERMINAL
                terminals = ()
                repository_terminals = ()

        work = RecurrentFragmentBuildWork(
            root_rollouts=1 if status == READY or selected_actuations else 0,
            root_replay_races=root_replays,
            donor_replay_races=donor_replays,
            replay_selector_cell_steps=sum(
                fragments._race_cell_steps(race) for race in replay_races
            ),
            affordance_races=len(affordance_races),
            affordance_selector_cell_steps=sum(
                fragments._race_cell_steps(race) for race in affordance_races
            ),
            policy_races=len(policy_races),
            policy_selector_cell_steps=sum(
                fragments._race_cell_steps(race) for race in policy_races
            ),
            effect_races=len(effect_races),
            effect_selector_cell_steps=sum(
                fragments._race_cell_steps(race) for race in effect_races
            ),
            selected_actuations=selected_actuations,
            committed_edits=committed_edits,
            revisions=revisions,
            unselected_splice_executions=UNSELECTED_SPLICE_EXECUTIONS,
            terminal_program_count=len(terminals),
            terminal_case_executions=terminal_case_executions,
            physical_output_encodings=physical_output_encodings,
        )
        partial = RecurrentFragmentBuildState(
            status=status,
            policy_checkpoint_sha256=policy.checkpoint_sha256,
            case_count=len(frozen_cases),
            terminal_program_count=len(terminals),
            terminal_keys=tuple(item.key for item in terminals),
            terminal_program_sha256s=tuple(
                item.program_sha256 for item in terminals
            ),
            terminal_semantic_sha256s=tuple(
                _encoding_vector_sha256(item.physical_outputs) for item in terminals
            ),
            applied_relation_receipt_sha256s=tuple(applied),
            desired_reads_during_build=DESIRED_READS_DURING_BUILD,
            source_materializations_during_build=SOURCE_MATERIALIZATIONS_DURING_BUILD,
            work=work,
            sha256="",
        )
        state = partial._replace(sha256=_build_state_sha256(partial))
        instance = object.__new__(cls)
        instance._policy = policy
        instance._signature = signature
        instance._cases = frozen_cases
        instance._terminals = terminals
        instance._repository_terminals = repository_terminals
        instance._state = state
        instance._materialized = {}
        instance._source_materialization_count = 0
        instance._action_bank = (
            action_bank.FrozenRepositoryActionBankV1(
                policy_checkpoint_sha256=policy.checkpoint_sha256,
                frontier_sha256=state.sha256,
                root_program_sha256=root.sha256,
                case_count=len(frozen_cases),
                records=tuple(
                    action_bank.make_repository_action_candidate_v1(
                        frontier_sha256=state.sha256,
                        program_sha256=item.program_sha256,
                        physical_outputs=item.physical_outputs,
                        applied_relation_receipt_sha256s=(
                            item.applied_relation_receipt_sha256s
                        ),
                    )
                    for item in repository_terminals
                ),
            )
            if status == READY
            and len(repository_terminals) == action_bank.CANDIDATE_COUNT
            else None
        )
        instance._authenticate()
        return instance

    @property
    def state(self) -> RecurrentFragmentBuildState:
        self._authenticate()
        return self._state

    @property
    def status(self) -> str:
        return self._state.status

    @property
    def source_materialization_count(self) -> int:
        return self._source_materialization_count

    @property
    def root_program_sha256(self) -> str:
        return self._policy._records[0].base_program_sha256

    @property
    def repository_action_bank(self) -> action_bank.FrozenRepositoryActionBankV1:
        if self._action_bank is None:
            raise RuntimeError("List frontier has no complete repository action bank")
        self._action_bank.authenticate()
        return self._action_bank

    @property
    def candidate_records(
        self,
    ) -> tuple[action_bank.RepositoryActionCandidateRecordV1, ...]:
        return self.repository_action_bank.records

    def materialize_selected(
        self,
        candidate: action_bank.RepositoryActionCandidateRecordV1,
    ) -> action_bank.RepositoryPatchWitnessV1:
        self._authenticate()
        candidate.authenticate()
        matches = tuple(
            terminal
            for terminal in self._repository_terminals
            if terminal.program_sha256 == candidate.program_sha256
            and terminal.applied_relation_receipt_sha256s
            == candidate.applied_relation_receipt_sha256s
            and terminal.physical_outputs == candidate.physical_outputs
            and candidate.frontier_sha256 == self._state.sha256
        )
        if len(matches) != 1 or candidate not in self.repository_action_bank.records:
            raise ValueError("selected List candidate record is not authentic")
        terminal = matches[0]
        program = typed_program.Program(terminal.tokens, self._signature)
        source = typed_program.compile(program).source
        self._source_materialization_count += 1
        return action_bank.make_repository_patch_witness_v1(
            candidate=candidate,
            source=source,
        )

    def encode_visible_expected(
        self,
        raw: object,
    ) -> tuple[output_cells.TypedOutputEncoding, ...]:
        if (
            type(raw) is not list
            or len(raw) != self._state.case_count
            or any(type(value) is not int for value in raw)
        ):
            raise ValueError("visible expected values must align exact Int outputs")
        encoded = fragments._encode_outputs(tuple(raw))
        if encoded is None:
            raise ValueError("visible expected Int outputs are not physically encodable")
        return encoded

    def race_repository_consequence(
        self,
        passed_cell: int,
    ) -> action_bank.RepositoryConsequenceDecisionV1:
        if passed_cell not in (0, 1):
            raise ValueError("repository consequence must be one physical bit")
        feature = FEATURE_EFFECT_MATCH if passed_cell else FEATURE_EFFECT_MISMATCH
        action, race = _policy_race(
            self._policy._policy_excitatory,
            self._policy._policy_inhibitory,
            feature,
        )
        expected_action = ACTION_COMMIT if passed_cell else ACTION_REVISE
        if action != expected_action:
            raise RuntimeError("List consequence policy returned the wrong action")
        return action_bank.make_repository_consequence_decision_v1(
            policy_checkpoint_sha256=self._policy.checkpoint_sha256,
            passed_cell=passed_cell,
            outcome_key=(
                action_bank.OUTCOME_COMMIT
                if passed_cell
                else action_bank.OUTCOME_REVISE
            ),
            physical_race_sha256=race.sha256,
        )

    def _authenticate(self) -> None:
        self._policy.authenticate()
        state = self._state
        if (
            self._signature != self._policy._library._signature
            or state.policy_checkpoint_sha256 != self._policy.checkpoint_sha256
            or state.case_count != len(self._cases)
            or state.terminal_program_count != len(self._terminals)
            or state.terminal_keys != tuple(item.key for item in self._terminals)
            or state.terminal_program_sha256s
            != tuple(item.program_sha256 for item in self._terminals)
            or state.terminal_semantic_sha256s
            != tuple(
                _encoding_vector_sha256(item.physical_outputs)
                for item in self._terminals
            )
            or state.desired_reads_during_build != 0
            or state.source_materializations_during_build != 0
            or state.sha256 != _build_state_sha256(state)
        ):
            raise RuntimeError("recurrent relation frontier changed")
        if state.status == READY:
            bank = self.repository_action_bank
            if (
                len(self._repository_terminals) != action_bank.CANDIDATE_COUNT
                or bank.policy_checkpoint_sha256 != state.policy_checkpoint_sha256
                or bank.frontier_sha256 != state.sha256
                or bank.root_program_sha256 != self.root_program_sha256
                or bank.case_count != state.case_count
            ):
                raise RuntimeError("recurrent repository action bank changed")
        for terminal in (*self._terminals, *self._repository_terminals):
            if (
                typed_program.Program(terminal.tokens, self._signature).sha256
                != terminal.program_sha256
            ):
                raise RuntimeError("terminal program authentication failed")

    def select(
        self,
        desired_outputs: Sequence[output_cells.TypedOutputEncoding],
    ) -> RecurrentFragmentSelection:
        self._authenticate()
        if self._state.status != READY:
            return RecurrentFragmentSelection(
                status=self._state.status,
                winner_key=None,
                exact_winner_keys=(),
                exact_proof_count=0,
                witness=None,
                materialized_witnesses=0,
                race=None,
            )
        race = semantic_race.select_exact_semantic_vector(
            self._state.terminal_keys,
            tuple(item.physical_outputs for item in self._terminals),
            desired_outputs,
        )
        if race.status in {STOP_NO_EXACT, STOP_MULTIPLE_EXACT}:
            return RecurrentFragmentSelection(
                status=race.status,
                winner_key=None,
                exact_winner_keys=race.exact_winner_keys,
                exact_proof_count=0,
                witness=None,
                materialized_witnesses=0,
                race=race,
            )
        if race.status != SELECTED or race.winner_key is None:
            raise RuntimeError("terminal semantic race returned unsupported state")
        terminal = next(
            item for item in self._terminals if item.key == race.winner_key
        )
        witness = self._materialized.get(terminal.key)
        materialized = 0
        if witness is None:
            program = typed_program.Program(terminal.tokens, self._signature)
            source = typed_program.compile(program).source
            witness = RecurrentFragmentWitness(
                program=program,
                source=source,
                applied_relation_receipt_sha256s=(
                    terminal.applied_relation_receipt_sha256s
                ),
                sha256=_digest(
                    _WITNESS_NAMESPACE,
                    {
                        "frontier": self._state.sha256,
                        "program": program.sha256,
                        "relations": terminal.applied_relation_receipt_sha256s,
                        "source": source,
                    },
                ),
            )
            self._materialized[terminal.key] = witness
            self._source_materialization_count += 1
            materialized = 1
        return RecurrentFragmentSelection(
            status=SELECTED,
            winner_key=terminal.key,
            exact_winner_keys=race.exact_winner_keys,
            exact_proof_count=1,
            witness=witness,
            materialized_witnesses=materialized,
            race=race,
        )


__all__ = [
    "ACTION_COUNT",
    "CLAIM_BOUNDARY",
    "FEATURE_COUNT",
    "MAX_AFFORDANCE_RACES_PER_ROOT",
    "MAX_COMMITTED_EDITS",
    "MAX_EFFECT_RACES_PER_ROOT",
    "MAX_POLICY_RACES_PER_ROOT",
    "MAX_REVISIONS_PER_ROOT",
    "MAX_ROOTS",
    "READY",
    "RELATION_COUNT",
    "SELECTED",
    "STOP_ACTUATION",
    "STOP_AMBIGUOUS_AFFORDANCE",
    "STOP_EFFECT",
    "STOP_MULTIPLE_EXACT",
    "STOP_NO_EXACT",
    "STOP_POLICY",
    "STOP_RELATION",
    "STOP_REPLAY",
    "FragmentRelationLesson",
    "FragmentRelationReceipt",
    "FrozenCausalRecurrentFragmentRelationPolicyV2",
    "RecurrentFragmentBuildState",
    "RecurrentFragmentRelationFrontierV2",
    "RecurrentFragmentSelection",
    "RecurrentFragmentWitness",
    "RelationPolicyFitStop",
    "RelationPolicyState",
    "fit_causal_recurrent_fragment_relation_policy_v2",
]
