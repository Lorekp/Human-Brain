"""Transparent literal-reuse reference over complete raw public observations.

The frozen reference still chooses documents/demonstrations by trigram cosine,
copies its first boundary-quoted literal, and reuses only a unique nonempty taught
edit anchor. This adapter changes input framing and transport bounds, not those
prediction rules. No corpus restoration, prediction or teacher runs on import.
"""

from __future__ import annotations

import hashlib

from experiments.r4_integrated.persistent_language_action_tasks_v1 import SimpleBaseline
from humanbrain.sequence_assembly_observation import (
    MAX_REPLACEMENT_BYTES,
    QUERY_FLAG_NAMES,
    RawObservation,
    SpliceProposal,
)


def similarity_query(observation: RawObservation) -> bytes:
    """Complete request plus mechanically framed actual feedback, without parsing.

    No feedback returns the exact historical raw request. With feedback, append
    NUL, uint32LE length + raw stdout, uint32LE length + raw stderr, one returncode
    presence byte, signed64LE code (zero if absent), then the five outcome flags.
    Length/status fields only frame actual public input; they never encode a grade.
    """
    if not isinstance(observation, RawObservation):
        raise TypeError("reference requires the complete RawObservation")
    feedback = observation.feedback
    if feedback is None:
        return observation.request
    code = feedback.returncode
    return b"".join((
        observation.request, b"\x00",
        len(feedback.stdout).to_bytes(4, "little"), feedback.stdout,
        len(feedback.stderr).to_bytes(4, "little"), feedback.stderr,
        bytes((code is not None,)),
        (0 if code is None else code).to_bytes(8, "little", signed=True),
        bytes(getattr(feedback, name) for name in QUERY_FLAG_NAMES[2:]),
    ))


def _files(observation: RawObservation) -> list[dict]:
    return ([] if observation.workspace is None else [
        {"path": item.path, "content_hex": item.content.hex()}
        for item in observation.workspace.files
    ])


def _proposal(record: dict, observation: RawObservation) -> SpliceProposal:
    if observation.workspace is None:
        raise ValueError("reference edit requires an actual bounded workspace")
    return SpliceProposal(
        operation=record["operation"], path=record["path"],
        start=record["start"], end=record["end"],
        replacement=bytes.fromhex(record["replacement_hex"]),
        stop_reason=record["stop_reason"], workspace_sha256=observation.workspace.sha256,
    )


class SimpleReference(SimpleBaseline):
    """Same lexical/literal policy, with whole diagnostic bytes in its query.

    Reading and STOP corrections remain unstored; SPLICE corrections add the
    observed old anchor and offered literal. The caller, not this class, owns
    the predetermined twelve correction occasions and execution-before-teacher
    custody. Prediction accepts no episode identifier, grade or desired output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.new_teacher_offers = {"reading": 0, "STOP": 0, "SPLICE": 0}

    @classmethod
    def from_restored(cls, previous: SimpleBaseline) -> SimpleReference:
        """Copy authenticated history, without calling predict/observe_teacher."""
        if not isinstance(previous, SimpleBaseline):
            raise TypeError("reference restoration needs the authenticated prior reference")
        if any(not isinstance(document, bytes) for document in previous.documents):
            raise ValueError("restored documents must be raw bytes")
        names = {"query", "old", "replacement", "source_sha256", "query_sha256"}
        for demo in previous.demonstrations:
            if (not isinstance(demo, dict) or set(demo) != names
                    or any(not isinstance(demo[name], bytes)
                           for name in ("query", "old", "replacement"))
                    or len(demo["replacement"]) > MAX_REPLACEMENT_BYTES
                    or hashlib.sha256(demo["query"]).hexdigest() != demo["query_sha256"]):
                raise ValueError("restored edit demonstration changed its raw query contract")
            for name in ("source_sha256", "query_sha256"):
                value = demo[name]
                if (not isinstance(value, str) or len(value) != 64
                        or any(char not in "0123456789abcdef" for char in value)):
                    raise ValueError("restored demonstration digest is malformed")
        result = cls()
        result.documents = list(previous.documents)
        result.demonstrations = [dict(demo) for demo in previous.demonstrations]
        return result

    def predict(self, observation: RawObservation, *, kind: str) -> dict:
        if kind not in {"reading", "edit"}:
            raise ValueError("reference kind must be reading or edit")
        query = similarity_query(observation)
        files = _files(observation)
        if kind == "edit" and observation.workspace is None:
            raise ValueError("reference edit requires a workspace")
        result = super().predict(query, files, kind=kind)
        if kind == "edit":
            # Bind unchanged literal choices to the larger transport's real
            # workspace. No legacy128-file/16-payload constructor is used.
            result["proposal"] = _proposal(result["proposal"], observation).to_record()
        feedback = observation.feedback
        result["attribution"].update({
            "reference_schema": "persistent-language-action-simple-reference-v2",
            "query_framing": "request unchanged; optional full raw feedback frame-v1",
            "similarity_query_sha256": hashlib.sha256(query).hexdigest(),
            "similarity_query_bytes": len(query),
            "request_bytes": len(observation.request),
            "feedback_in_similarity": feedback is not None,
            "feedback_stdout_bytes": 0 if feedback is None else len(feedback.stdout),
            "feedback_stderr_bytes": 0 if feedback is None else len(feedback.stderr),
            "observation_sha256": observation.sha256,
            "scope": (
                "generic lexical feedback contribution; no error interpretation or repair parser"),
        })
        return result

    def observe_teacher(self, observation: RawObservation, *, kind: str, teacher: dict) -> dict:
        if (kind not in {"reading", "edit"} or not isinstance(teacher, dict)
                or teacher.get("kind") != kind):
            raise ValueError("reference correction must retain its offered kind")
        query = similarity_query(observation)
        if kind == "reading":
            self.new_teacher_offers["reading"] += 1
            return {"offered": True, "stored": False, "reason": "reading_targets_not_stored"}
        proposal = _proposal(teacher, observation)
        if proposal.operation == "STOP":
            self.new_teacher_offers["STOP"] += 1
            return {"offered": True, "stored": False, "reason": "STOP_has_no_literal_anchor"}
        if proposal.stop_reason != "eos":
            raise ValueError("a supplied SPLICE demonstration must include its complete payload")
        # The inherited method checks the exact observed teacher span and stores
        # only its raw anchor/replacement. Empty insertion anchors remain unusable
        # for prediction, rather than inventing a placement rule.
        super().observe_teacher(query, _files(observation), kind=kind, teacher=teacher)
        self.new_teacher_offers["SPLICE"] += 1
        return {"offered": True, "stored": True, "reason": "supplied_literal_edit",
                "demonstration_index": len(self.demonstrations) - 1,
                "similarity_query_sha256": hashlib.sha256(query).hexdigest()}
