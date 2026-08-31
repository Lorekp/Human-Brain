"""Bounded pure-Python adapter for rooted entry-return position provenance.

One explicit one-argument function may contain at most four eager assignment or
pure-expression occurrences followed by ``return <local>``.  An isolated
subprocess observes every occurrence on two source-root-permuted unique-symbol
probes.  Every probe occurrence becomes one opaque 16-byte event containing a
bijective output-position map or a neutral no-map observation.  Two anonymous
input carriers feed the events, events feed anonymous occurrence carriers, and
only the carrier supplying the returned local reaches an anonymous observable
boundary carrier.  No action, relevance, opcode, lexeme, raw source, probe
symbol, or hidden evaluation information survives the adapter.
"""

from __future__ import annotations

import ast
import base64
import json
import struct
import subprocess
import sys
from hashlib import sha256
from typing import NamedTuple

from experiments.r3_pilot.common_rooted_structured_event_graph_v1 import (
    RootedStructuredEventGraphV1,
    project_rooted_structured_event_graph_v1,
)

SCHEMA = "PYTHON_ENTRY_RETURN_PROVENANCE_V1"

MAX_SOURCE_BYTES = 256
MAX_FUNCTIONS = 1
MAX_OCCURRENCES = 4
MAX_AST_NODES = 64
PROBE_LENGTHS = (5, 6)
PROBE_TIMEOUT_SECONDS = 2.0
PAYLOAD_BYTES = 16

_PAYLOAD_VERSION = 1
_INPUT_CARRIER_FAMILY = 0
_OCCURRENCE_CARRIER_FAMILY = 1
_BOUNDARY_CARRIER_FAMILY = 2
_UNUSED_POSITION = 0xFF
_NO_POSITION_MAP = 0
_BIJECTIVE_POSITION_MAP = 1
_SCHEMA_NAMESPACE = b"python-entry-return-provenance-schema-v1"
_BOUNDARY_NAMESPACE = b"python-entry-return-provenance-boundary-v1"
_RESULT_NAMESPACE = b"python-entry-return-provenance-result-v1"
_SCHEMA_SHA256 = sha256(
    _SCHEMA_NAMESPACE
    + b"\0one-entry;eager-occurrences:4;source-root-probes:5,6;position-observation:6"
).hexdigest()

_SAFE_BUILTINS = frozenset({"len", "reversed"})
_ALLOWED_NODE_TYPES = (
    ast.Module,
    ast.FunctionDef,
    ast.arguments,
    ast.arg,
    ast.Assign,
    ast.Expr,
    ast.Return,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.Subscript,
    ast.Slice,
    ast.UnaryOp,
    ast.USub,
    ast.Call,
    ast.Attribute,
    ast.IfExp,
    ast.BinOp,
    ast.Mod,
)

_RUNNER = r"""
import ast
import base64
import json
import sys

request = json.loads(sys.stdin.read())
source = base64.b64decode(request["source"]).decode("utf-8")
tree = ast.parse(source, mode="exec", type_comments=False)
function_node = tree.body[0]
instrumented = []
occurrence_index = 0
for statement in function_node.body[:-1]:
    if isinstance(statement, ast.Assign):
        instrumented.append(statement)
        captured_value = ast.Name(id=statement.targets[0].id, ctx=ast.Load())
    else:
        captured_value = statement.value
    instrumented.append(
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id="__provenance_capture__", ctx=ast.Load()),
                args=[ast.Constant(value=occurrence_index), captured_value],
                keywords=[],
            )
        )
    )
    occurrence_index += 1
instrumented.append(function_node.body[-1])
function_node.body = instrumented
ast.fix_missing_locations(tree)
scope = {"__builtins__": {"len": len, "reversed": reversed}}
exec(compile(tree, "<bounded-provenance>", "exec", dont_inherit=True, optimize=2), scope, scope)
function = scope[request["entry"]]
transcript = []
for probe in request["probes"]:
    captured = []
    def capture(index, value):
        captured.append((index, value))
        return value
    scope["__provenance_capture__"] = capture
    try:
        returned = function(probe)
    except BaseException:
        transcript.append({
            "captures": [[False, ""] for _ in range(occurrence_index)],
            "returned": [False, ""],
        })
        continue
    if len(captured) != occurrence_index or any(
        index != expected for expected, (index, _value) in enumerate(captured)
    ):
        transcript.append({
            "captures": [[False, ""] for _ in range(occurrence_index)],
            "returned": [False, ""],
        })
        continue
    encoded_captures = [
        [True, value] if type(value) is str and len(value) <= 256 else [False, ""]
        for _index, value in captured
    ]
    encoded_return = (
        [True, returned]
        if type(returned) is str and len(returned) <= 256
        else [False, ""]
    )
    transcript.append({"captures": encoded_captures, "returned": encoded_return})
sys.stdout.write(json.dumps(transcript, ensure_ascii=True, separators=(",", ":")))
"""


class PythonEntryReturnProvenanceError(ValueError):
    """Base error for a source that cannot form the bounded v1 observation."""


class PythonEntryReturnProvenanceCapacityError(PythonEntryReturnProvenanceError):
    """The source exceeds a frozen byte, occurrence, or AST-node capacity."""


class PythonEntryReturnProvenanceSafetyError(PythonEntryReturnProvenanceError):
    """The source is outside the deliberately tiny pure-Python subset."""


class PythonEntryReturnProvenanceExecutionError(PythonEntryReturnProvenanceError):
    """The isolated observation process did not return a bounded transcript."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _valid_sha256(value: object) -> bool:
    if type(value) is not str or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


class PythonEntryReturnProvenanceV1(NamedTuple):
    """Authenticated roots plus one neutral graph; no source or lexemes survive."""

    source_sha256: str
    boundary_sha256: str
    schema_sha256: str
    graph: RootedStructuredEventGraphV1
    sha256: str

    def authenticate(self) -> None:
        if type(self.graph) is not RootedStructuredEventGraphV1:
            raise TypeError("exact rooted structured-event graph required")
        self.graph.authenticate()
        if (
            self.graph.root_carrier_indices != (0, 1)
            or self.graph.boundary_carrier_indices
            != (self.graph.carrier_count - 1,)
            or not 4 <= self.graph.carrier_count <= 2 + MAX_OCCURRENCES + 1
        ):
            raise PythonEntryReturnProvenanceError(
                "entry-return graph has invalid anonymous carriers"
            )
        carrier_count = self.graph.carrier_count
        boundary_index = carrier_count - 1
        boundary_edges = tuple(
            edge
            for edge in self.graph.incidences
            if edge.target_index == boundary_index
        )
        if (
            len(boundary_edges) != 1
            or not 2 <= boundary_edges[0].source_index < boundary_index
            or boundary_edges[0].ordinal != 0
        ):
            raise PythonEntryReturnProvenanceError(
                "entry-return graph must have one selected boundary incidence"
            )
        selected_carrier = boundary_edges[0].source_index
        occurrence_count = carrier_count - 3
        if self.graph.event_count != occurrence_count * len(PROBE_LENGTHS):
            raise PythonEntryReturnProvenanceError(
                "entry-return graph must retain every probe occurrence"
            )
        expected_edges = {(selected_carrier, boundary_index, 0)}
        for occurrence_index in range(occurrence_count):
            occurrence_carrier = 2 + occurrence_index
            for probe_index in range(len(PROBE_LENGTHS)):
                event_vertex = (
                    carrier_count
                    + occurrence_index * len(PROBE_LENGTHS)
                    + probe_index
                )
                expected_edges.add((probe_index, event_vertex, probe_index))
                expected_edges.add(
                    (event_vertex, occurrence_carrier, probe_index)
                )
        actual_edges = {
            (edge.source_index, edge.target_index, edge.ordinal)
            for edge in self.graph.incidences
        }
        if actual_edges != expected_edges:
            raise PythonEntryReturnProvenanceError(
                "entry-return graph contains an incidence outside its schema"
            )
        boundary_payload = {
            "source": self.source_sha256,
            "schema": self.schema_sha256,
            "graph": self.graph.sha256,
            "carrier": self.graph.carrier_roots[boundary_index].sha256,
        }
        expected_boundary = _digest(_BOUNDARY_NAMESPACE, boundary_payload)
        payload = {
            "source": self.source_sha256,
            "boundary": self.boundary_sha256,
            "schema": self.schema_sha256,
            "graph": self.graph.sha256,
        }
        if (
            not _valid_sha256(self.source_sha256)
            or self.schema_sha256 != _SCHEMA_SHA256
            or self.boundary_sha256 != expected_boundary
            or self.sha256 != _digest(_RESULT_NAMESPACE, payload)
        ):
            raise PythonEntryReturnProvenanceError(
                "Python entry-return provenance authentication failed"
            )


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


def _slice_atom_is_safe(node: ast.expr | None) -> bool:
    if node is None:
        return True
    if isinstance(node, ast.Constant):
        return type(node.value) is int and abs(node.value) <= 64
    return (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) is int
        and 0 <= node.operand.value <= 64
    )


def _validate_call(call: ast.Call, argument_name: str) -> None:
    if call.keywords or len(call.args) != 1:
        raise PythonEntryReturnProvenanceSafetyError(
            "calls require one positional argument and no keywords"
        )
    if isinstance(call.func, ast.Name):
        if call.func.id not in _SAFE_BUILTINS:
            raise PythonEntryReturnProvenanceSafetyError(
                "only len and reversed calls are allowed"
            )
        return
    if not (
        isinstance(call.func, ast.Attribute)
        and call.func.attr == "join"
        and isinstance(call.func.value, ast.Constant)
        and type(call.func.value.value) is str
        and call.func.value.value == ""
        and isinstance(call.args[0], ast.Call)
        and isinstance(call.args[0].func, ast.Name)
        and call.args[0].func.id == "reversed"
        and len(call.args[0].args) == 1
        and not call.args[0].keywords
        and isinstance(call.args[0].args[0], ast.Name)
        and call.args[0].args[0].id == argument_name
    ):
        raise PythonEntryReturnProvenanceSafetyError(
            "the only allowed attribute call is ''.join(reversed(argument))"
        )


def _validate_expression(expression: ast.expr, argument_name: str) -> None:
    for node in ast.walk(expression):
        if not isinstance(node, _ALLOWED_NODE_TYPES):
            raise PythonEntryReturnProvenanceSafetyError(
                "expression contains a construct outside the pure subset"
            )
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id != argument_name and node.id not in _SAFE_BUILTINS:
                raise PythonEntryReturnProvenanceSafetyError(
                    "occurrence reads outside its input and safe builtins"
                )
        if isinstance(node, ast.Constant):
            if type(node.value) not in (str, bool, int, type(None)):
                raise PythonEntryReturnProvenanceSafetyError(
                    "only bounded scalar constants are allowed"
                )
            if type(node.value) is str and len(node.value) > MAX_SOURCE_BYTES:
                raise PythonEntryReturnProvenanceCapacityError(
                    "string constant exceeds the source capacity"
                )
        if isinstance(node, ast.Call):
            _validate_call(node, argument_name)
        if isinstance(node, ast.Attribute) and not any(
            isinstance(call, ast.Call) and call.func is node
            for call in ast.walk(expression)
        ):
            raise PythonEntryReturnProvenanceSafetyError(
                "attributes are allowed only as the join call target"
            )
        if isinstance(node, ast.Subscript):
            if (
                not isinstance(node.value, ast.Name)
                or node.value.id != argument_name
                or not isinstance(node.slice, ast.Slice)
                or not _slice_atom_is_safe(node.slice.lower)
                or not _slice_atom_is_safe(node.slice.upper)
                or not _slice_atom_is_safe(node.slice.step)
            ):
                raise PythonEntryReturnProvenanceSafetyError(
                    "only bounded input slices are allowed"
                )
        if isinstance(node, ast.BinOp) and not (
            isinstance(node.op, ast.Mod)
            and isinstance(node.left, ast.Call)
            and isinstance(node.left.func, ast.Name)
            and node.left.func.id == "len"
            and len(node.left.args) == 1
            and isinstance(node.left.args[0], ast.Name)
            and node.left.args[0].id == argument_name
            and isinstance(node.right, ast.Constant)
            and type(node.right.value) is int
            and 1 <= node.right.value <= 16
        ):
            raise PythonEntryReturnProvenanceSafetyError(
                "only bounded len(input) modulo expressions are allowed"
            )
        if isinstance(node, ast.UnaryOp) and not (
            isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and type(node.operand.value) is int
            and 0 <= node.operand.value <= 64
        ):
            raise PythonEntryReturnProvenanceSafetyError(
                "only bounded negative integer slice atoms are allowed"
            )


def _validate_source(
    source: bytes,
    entry_point: str,
) -> tuple[tuple[ast.stmt, ...], int]:
    if type(source) is not bytes:
        raise TypeError("Python source must be exact bytes")
    if len(source) > MAX_SOURCE_BYTES:
        raise PythonEntryReturnProvenanceCapacityError(
            f"Python source exceeds {MAX_SOURCE_BYTES} bytes"
        )
    if (
        type(entry_point) is not str
        or not entry_point.isidentifier()
        or entry_point in _SAFE_BUILTINS
        or _is_dunder(entry_point)
    ):
        raise PythonEntryReturnProvenanceSafetyError(
            "entry point must be one explicit non-dunder identifier"
        )
    try:
        text = source.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        raise PythonEntryReturnProvenanceSafetyError(
            "Python source must be strict UTF-8"
        ) from error
    if "\x00" in text:
        raise PythonEntryReturnProvenanceSafetyError("NUL source bytes are forbidden")
    try:
        tree = ast.parse(text, mode="exec", type_comments=False)
    except SyntaxError as error:
        raise PythonEntryReturnProvenanceSafetyError(
            "Python source must parse before execution"
        ) from error
    nodes = tuple(ast.walk(tree))
    if len(nodes) > MAX_AST_NODES:
        raise PythonEntryReturnProvenanceCapacityError(
            f"Python source exceeds {MAX_AST_NODES} AST nodes"
        )
    if any(not isinstance(node, _ALLOWED_NODE_TYPES) for node in nodes):
        raise PythonEntryReturnProvenanceSafetyError(
            "Python source contains a construct outside the pure subset"
        )
    if len(tree.body) != MAX_FUNCTIONS or not isinstance(tree.body[0], ast.FunctionDef):
        raise PythonEntryReturnProvenanceSafetyError(
            "source must contain exactly one top-level entry function"
        )
    function = tree.body[0]
    positional = function.args.posonlyargs + function.args.args
    if (
        function.name != entry_point
        or len(positional) != 1
        or function.args.vararg is not None
        or function.args.kwonlyargs
        or function.args.kw_defaults
        or function.args.kwarg is not None
        or function.args.defaults
        or function.decorator_list
        or function.returns is not None
        or function.type_comment is not None
        or positional[0].annotation is not None
        or positional[0].arg in _SAFE_BUILTINS
        or _is_dunder(positional[0].arg)
    ):
        raise PythonEntryReturnProvenanceSafetyError(
            "entry must take one unannotated positional argument"
        )
    if len(function.body) < 2 or not isinstance(function.body[-1], ast.Return):
        raise PythonEntryReturnProvenanceSafetyError(
            "entry must end with return of an eager assignment"
        )
    occurrences = tuple(function.body[:-1])
    if not 1 <= len(occurrences) <= MAX_OCCURRENCES:
        raise PythonEntryReturnProvenanceCapacityError(
            f"entry requires 1..{MAX_OCCURRENCES} eager occurrences"
        )
    if any(not isinstance(node, (ast.Assign, ast.Expr)) for node in occurrences):
        raise PythonEntryReturnProvenanceSafetyError(
            "entry body may contain only eager assignments or pure expressions"
        )
    assignments = tuple(
        node for node in occurrences if isinstance(node, ast.Assign)
    )
    if any(
        len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name)
        for node in assignments
    ):
        raise PythonEntryReturnProvenanceSafetyError(
            "assignments require one local-name target"
        )
    target_names = tuple(node.targets[0].id for node in assignments)
    if (
        len(set(target_names)) != len(target_names)
        or any(
            name == positional[0].arg or name in _SAFE_BUILTINS or _is_dunder(name)
            for name in target_names
        )
    ):
        raise PythonEntryReturnProvenanceSafetyError(
            "assignment targets must be unique anonymous locals"
        )
    returned = function.body[-1].value
    if not isinstance(returned, ast.Name) or returned.id not in target_names:
        raise PythonEntryReturnProvenanceSafetyError(
            "entry return must name exactly one eager assignment"
        )
    argument_name = positional[0].arg
    for occurrence in occurrences:
        expression = occurrence.value
        _validate_expression(expression, argument_name)
    selected_occurrence = next(
        index
        for index, occurrence in enumerate(occurrences)
        if isinstance(occurrence, ast.Assign)
        and occurrence.targets[0].id == returned.id
    )
    return occurrences, selected_occurrence


def _source_root_permuted_probes(source_sha256: str) -> tuple[str, str]:
    source_root = bytes.fromhex(source_sha256)
    probes: list[str] = []
    for lane, length in enumerate(PROBE_LENGTHS):
        candidates = tuple(chr(0xE000 + lane * 16 + index) for index in range(16))
        ordered = sorted(
            candidates,
            key=lambda symbol: sha256(
                source_root + bytes((lane, length)) + symbol.encode("utf-8")
            ).digest(),
        )
        probes.append("".join(ordered[:length]))
    return probes[0], probes[1]


def _observe_occurrences_in_subprocess(
    source: bytes,
    entry_point: str,
    occurrence_count: int,
    probes: tuple[str, str],
) -> tuple[tuple[tuple[bool, str], ...], ...]:
    request = json.dumps(
        {
            "source": base64.b64encode(source).decode("ascii"),
            "entry": entry_point,
            "probes": probes,
        },
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-S", "-c", _RUNNER],
            input=request,
            text=True,
            capture_output=True,
            timeout=PROBE_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise PythonEntryReturnProvenanceExecutionError(
            "bounded provenance subprocess failed"
        ) from error
    if process.returncode != 0 or len(process.stdout) > 16_384:
        raise PythonEntryReturnProvenanceExecutionError(
            "bounded provenance subprocess returned an invalid transcript"
        )
    try:
        decoded = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise PythonEntryReturnProvenanceExecutionError(
            "bounded provenance transcript is not JSON"
        ) from error
    if type(decoded) is not list or len(decoded) != len(PROBE_LENGTHS):
        raise PythonEntryReturnProvenanceExecutionError(
            "bounded provenance transcript changed probe count"
        )
    by_probe: list[tuple[tuple[bool, str], ...]] = []
    for probe_observation in decoded:
        if type(probe_observation) is not dict:
            raise PythonEntryReturnProvenanceExecutionError(
                "bounded provenance transcript changed shape"
            )
        captures = probe_observation.get("captures")
        returned = probe_observation.get("returned")
        if type(captures) is not list or len(captures) != occurrence_count:
            raise PythonEntryReturnProvenanceExecutionError(
                "bounded provenance transcript changed occurrence count"
            )
        values = captures + [returned]
        row: list[tuple[bool, str]] = []
        for observation in values:
            if (
                type(observation) is not list
                or len(observation) != 2
                or type(observation[0]) is not bool
                or type(observation[1]) is not str
                or len(observation[1]) > MAX_SOURCE_BYTES
            ):
                raise PythonEntryReturnProvenanceExecutionError(
                    "bounded provenance observation changed shape"
                )
            row.append((observation[0], observation[1]))
        by_probe.append(tuple(row))
    return tuple(by_probe)


def _position_mapping(probe: str, output: str) -> tuple[int, ...] | None:
    if len(output) != len(probe):
        return None
    positions = {symbol: index for index, symbol in enumerate(probe)}
    if len(positions) != len(probe) or set(output) != set(probe):
        return None
    mapping = tuple(positions[symbol] for symbol in output)
    if len(set(mapping)) != len(probe):
        return None
    return mapping


def _carrier_payload(family: int, ordinal: int) -> bytes:
    payload = struct.pack("<BBBB12x", _PAYLOAD_VERSION, family, ordinal, 0)
    if len(payload) != PAYLOAD_BYTES:
        raise RuntimeError("carrier payload encoding changed")
    return payload


def _position_observation_payload(
    mapping: tuple[int, ...] | None,
    probe_length: int,
) -> bytes:
    if mapping is None:
        observation = _NO_POSITION_MAP
        padded = (_UNUSED_POSITION,) * 6
    else:
        if len(mapping) != probe_length:
            raise RuntimeError("position observation length changed")
        observation = _BIJECTIVE_POSITION_MAP
        padded = mapping + (_UNUSED_POSITION,) * (6 - len(mapping))
    payload = struct.pack(
        "<BBBB6B6x",
        _PAYLOAD_VERSION,
        probe_length,
        observation,
        0,
        *padded,
    )
    if len(payload) != PAYLOAD_BYTES:
        raise RuntimeError("position-observation payload encoding changed")
    return payload


def observe_python_entry_return_provenance_v1(
    source: bytes,
    *,
    entry_point: str,
) -> PythonEntryReturnProvenanceV1:
    """Observe every eager occurrence and root only the returned local."""

    occurrences, selected_occurrence = _validate_source(source, entry_point)
    source_root = sha256(source).hexdigest()
    probes = _source_root_permuted_probes(source_root)
    observations = _observe_occurrences_in_subprocess(
        source, entry_point, len(occurrences), probes
    )

    carrier_payloads = [
        _carrier_payload(_INPUT_CARRIER_FAMILY, 0),
        _carrier_payload(_INPUT_CARRIER_FAMILY, 1),
    ]
    carrier_payloads.extend(
        _carrier_payload(_OCCURRENCE_CARRIER_FAMILY, index)
        for index in range(len(occurrences))
    )
    boundary_index = len(carrier_payloads)
    carrier_payloads.append(_carrier_payload(_BOUNDARY_CARRIER_FAMILY, 0))
    carrier_count = len(carrier_payloads)

    event_payloads: list[bytes] = []
    incidences: list[tuple[int, int, int]] = []
    for occurrence_index in range(len(occurrences)):
        occurrence_carrier = 2 + occurrence_index
        for probe_index, probe in enumerate(probes):
            is_string, output = observations[probe_index][occurrence_index]
            mapping = _position_mapping(probe, output) if is_string else None
            event_index = len(event_payloads)
            event_payloads.append(
                _position_observation_payload(mapping, len(probe))
            )
            event_vertex = carrier_count + event_index
            incidences.append((probe_index, event_vertex, probe_index))
            incidences.append((event_vertex, occurrence_carrier, probe_index))
    selected_carrier = 2 + selected_occurrence
    incidences.append((selected_carrier, boundary_index, 0))

    graph = project_rooted_structured_event_graph_v1(
        tuple(event_payloads),
        carrier_payloads=tuple(carrier_payloads),
        root_carrier_indices=(0, 1),
        boundary_carrier_indices=(boundary_index,),
        incidences=tuple(sorted(incidences)),
    )
    boundary_payload = {
        "source": source_root,
        "schema": _SCHEMA_SHA256,
        "graph": graph.sha256,
        "carrier": graph.carrier_roots[boundary_index].sha256,
    }
    boundary_root = _digest(_BOUNDARY_NAMESPACE, boundary_payload)
    payload = {
        "source": source_root,
        "boundary": boundary_root,
        "schema": _SCHEMA_SHA256,
        "graph": graph.sha256,
    }
    result = PythonEntryReturnProvenanceV1(
        source_sha256=source_root,
        boundary_sha256=boundary_root,
        schema_sha256=_SCHEMA_SHA256,
        graph=graph,
        sha256=_digest(_RESULT_NAMESPACE, payload),
    )
    result.authenticate()
    return result
