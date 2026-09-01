"""Content-only cell projection for authenticated consolidated knowledge cards.

The projection consumes every card in one corpus-bound acquire consolidation.
Only each card's selected lexical-factor tuple can affect cognition identities or
the rooted event graph.  Source, scope, chapter, path, declared-symbol, and
provenance values survive only as authenticated receipt roots.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from hashlib import sha256

from experiments.r3_pilot import common_rooted_structured_event_graph_v1 as structured_graph

from humanbrain.knowledge_consolidation import (
    LEXICAL_FACTOR_SCHEMA,
    MAX_SELECTED_FACTORS_PER_CARD,
    KnowledgeConceptCardV1,
    KnowledgeConsolidationError,
    KnowledgeConsolidationV1,
)
from humanbrain.knowledge_corpus import KnowledgeCorpus

SCHEMA = "humanbrain.consolidated-knowledge-cell-projection.v1"
CLAIM_BOUNDARY = (
    "content-only selected-factor cognition projection with provenance retained "
    "in receipts; not cell-learned abstraction or coding evidence"
)
ADMITTED_SPLIT = "acquire"
MAX_CONCEPTS = 16
MAX_FACTOR_ASSOCIATIONS = MAX_CONCEPTS * MAX_SELECTED_FACTORS_PER_CARD

_SHA256 = re.compile(r"[0-9a-f]{64}")
_FACTOR = re.compile(r"tok:[a-z_][a-z0-9_]*(?:\.[a-z_][a-z0-9_]*)*")
_CONCEPT_SYMBOL = re.compile(r"knowledge\.concept\.c[0-9a-f]{64}")

_COGNITION_NAMESPACE = b"consolidated-knowledge-cognition-v1"
_COGNITION_CORPUS_NAMESPACE = b"consolidated-knowledge-cognition-corpus-v1"
_CONCEPT_RECEIPT_NAMESPACE = b"consolidated-knowledge-concept-receipt-v1"
_PROVENANCE_NAMESPACE = b"consolidated-knowledge-provenance-v1"
_PROJECTION_NAMESPACE = b"consolidated-knowledge-cell-projection-v1"
_SCHEMA_NAMESPACE = b"consolidated-knowledge-cell-projection-schema-v1"
_ATTESTATION_NAMESPACE = b"consolidated-knowledge-cell-projection-attestation-v1"
_EVENT_PAYLOAD_NAMESPACE = b"consolidated-knowledge-cell-event-v1"
_ROOT_PAYLOAD_NAMESPACE = b"consolidated-knowledge-cell-root-v1"
_BOUNDARY_PAYLOAD_NAMESPACE = b"consolidated-knowledge-cell-boundary-v1"


class ConsolidatedKnowledgeCellProjectionError(ValueError):
    """A consolidated cognition projection failed integrity or capacity checks."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _digest(namespace: bytes, value: object) -> str:
    return sha256(namespace + b"\0" + _canonical_json(value)).hexdigest()


def _payload16(namespace: bytes, value: object) -> bytes:
    return sha256(namespace + b"\0" + _canonical_json(value)).digest()[:16]


def _valid_sha256(value: object) -> bool:
    return type(value) is str and _SHA256.fullmatch(value) is not None


def _cognition_sha256(content_factors: tuple[str, ...]) -> str:
    return _digest(
        _COGNITION_NAMESPACE,
        {
            "lexical_factor_schema": LEXICAL_FACTOR_SCHEMA,
            "selected_factors": content_factors,
        },
    )


def _concept_symbol(cognition_sha256: str) -> str:
    return f"knowledge.concept.c{cognition_sha256}"


@dataclass(frozen=True, slots=True)
class ProjectedKnowledgeConceptV1:
    """One cognition-facing row derived only from a selected factor tuple."""

    concept_symbol: str
    content_factors: tuple[str, ...]
    cognition_sha256: str

    def authenticate(self) -> None:
        if (
            type(self.content_factors) is not tuple
            or not self.content_factors
            or len(self.content_factors) > MAX_SELECTED_FACTORS_PER_CARD
            or self.content_factors != tuple(sorted(set(self.content_factors)))
            or any(_FACTOR.fullmatch(value) is None for value in self.content_factors)
            or not _valid_sha256(self.cognition_sha256)
            or self.cognition_sha256 != _cognition_sha256(self.content_factors)
            or _CONCEPT_SYMBOL.fullmatch(self.concept_symbol) is None
            or self.concept_symbol != _concept_symbol(self.cognition_sha256)
        ):
            raise ConsolidatedKnowledgeCellProjectionError(
                "projected knowledge cognition row authentication failed"
            )


@dataclass(frozen=True, slots=True)
class ProjectedKnowledgeConceptReceiptV1:
    """Audit-only provenance commitment for one cognition row."""

    cognition_sha256: str
    card_sha256: str
    provenance_roots: tuple[str, ...]
    sha256: str

    def authenticate(self) -> None:
        payload = {
            "cognition_sha256": self.cognition_sha256,
            "card_sha256": self.card_sha256,
            "provenance_roots": self.provenance_roots,
        }
        if (
            not _valid_sha256(self.cognition_sha256)
            or not _valid_sha256(self.card_sha256)
            or type(self.provenance_roots) is not tuple
            or not self.provenance_roots
            or self.provenance_roots != tuple(sorted(set(self.provenance_roots)))
            or any(not _valid_sha256(value) for value in self.provenance_roots)
            or self.card_sha256 not in self.provenance_roots
            or self.sha256 != _digest(_CONCEPT_RECEIPT_NAMESPACE, payload)
        ):
            raise ConsolidatedKnowledgeCellProjectionError(
                "projected knowledge concept receipt authentication failed"
            )


def _concept_from_card(card: KnowledgeConceptCardV1) -> ProjectedKnowledgeConceptV1:
    cognition_sha256 = _cognition_sha256(card.selected_factors)
    result = ProjectedKnowledgeConceptV1(
        concept_symbol=_concept_symbol(cognition_sha256),
        content_factors=card.selected_factors,
        cognition_sha256=cognition_sha256,
    )
    result.authenticate()
    return result


def _receipt_from_card(
    card: KnowledgeConceptCardV1,
    concept: ProjectedKnowledgeConceptV1,
    consolidation: KnowledgeConsolidationV1,
) -> ProjectedKnowledgeConceptReceiptV1:
    provenance_roots = tuple(
        sorted(
            {
                card.sha256,
                consolidation.input_records_sha256,
                consolidation.corpus_sha256,
                consolidation.producer_sha256,
            }
        )
    )
    payload = {
        "cognition_sha256": concept.cognition_sha256,
        "card_sha256": card.sha256,
        "provenance_roots": provenance_roots,
    }
    result = ProjectedKnowledgeConceptReceiptV1(
        cognition_sha256=concept.cognition_sha256,
        card_sha256=card.sha256,
        provenance_roots=provenance_roots,
        sha256=_digest(_CONCEPT_RECEIPT_NAMESPACE, payload),
    )
    result.authenticate()
    return result


def _cognition_corpus_sha256(
    concepts: tuple[ProjectedKnowledgeConceptV1, ...],
) -> str:
    return _digest(
        _COGNITION_CORPUS_NAMESPACE,
        tuple(concept.cognition_sha256 for concept in concepts),
    )


def _build_graph(
    concepts: tuple[ProjectedKnowledgeConceptV1, ...],
) -> structured_graph.RootedStructuredEventGraphV1:
    cognition_roots = tuple(concept.cognition_sha256 for concept in concepts)
    cognition_corpus_sha256 = _cognition_corpus_sha256(concepts)
    event_payloads = tuple(
        _payload16(_EVENT_PAYLOAD_NAMESPACE, cognition_sha256)
        for cognition_sha256 in cognition_roots
    )
    carrier_payloads = (
        _payload16(_ROOT_PAYLOAD_NAMESPACE, cognition_corpus_sha256),
        _payload16(_BOUNDARY_PAYLOAD_NAMESPACE, cognition_roots),
    )
    return structured_graph.project_rooted_structured_event_graph_v1(
        event_payloads,
        carrier_payloads=carrier_payloads,
        root_carrier_indices=(0,),
        boundary_carrier_indices=(1,),
        incidences=(
            (0, 1, 0),
            *((1, 2 + ordinal, ordinal) for ordinal in range(len(concepts))),
        ),
    )


def _producer_schema_sha256() -> str:
    return _digest(
        _SCHEMA_NAMESPACE,
        {
            "schema": SCHEMA,
            "admitted_split": ADMITTED_SPLIT,
            "max_concepts": MAX_CONCEPTS,
            "max_factor_associations": MAX_FACTOR_ASSOCIATIONS,
            "max_selected_factors_per_concept": MAX_SELECTED_FACTORS_PER_CARD,
            "cognition_identity": "selected-factor-tuple-only",
            "graph_payload": "one-factor-derived-event-per-concept",
            "provenance_location": "receipts-only",
        },
    )


def _provenance_sha256(
    *,
    input_records_sha256: str,
    policy_sha256: str,
    consolidation_corpus_sha256: str,
    consolidation_producer_sha256: str,
    receipts: tuple[ProjectedKnowledgeConceptReceiptV1, ...],
) -> str:
    return _digest(
        _PROVENANCE_NAMESPACE,
        {
            "input_records_sha256": input_records_sha256,
            "policy_sha256": policy_sha256,
            "consolidation_corpus_sha256": consolidation_corpus_sha256,
            "consolidation_producer_sha256": consolidation_producer_sha256,
            "concept_receipt_sha256s": tuple(receipt.sha256 for receipt in receipts),
        },
    )


def _producer_attestation_sha256(
    *,
    split: str,
    concepts: tuple[ProjectedKnowledgeConceptV1, ...],
    receipts: tuple[ProjectedKnowledgeConceptReceiptV1, ...],
    cognition_corpus_sha256: str,
    provenance_sha256: str,
    graph_sha256: str,
) -> str:
    return _digest(
        _ATTESTATION_NAMESPACE,
        {
            "split": split,
            "cognition_sha256s": tuple(concept.cognition_sha256 for concept in concepts),
            "concept_receipt_sha256s": tuple(receipt.sha256 for receipt in receipts),
            "cognition_corpus_sha256": cognition_corpus_sha256,
            "provenance_sha256": provenance_sha256,
            "graph_sha256": graph_sha256,
        },
    )


def _projection_sha256(
    *,
    split: str,
    input_records_sha256: str,
    policy_sha256: str,
    consolidation_corpus_sha256: str,
    consolidation_producer_sha256: str,
    concepts: tuple[ProjectedKnowledgeConceptV1, ...],
    receipts: tuple[ProjectedKnowledgeConceptReceiptV1, ...],
    cognition_corpus_sha256: str,
    provenance_sha256: str,
    graph_sha256: str,
    producer_receipt_sha256: str,
) -> str:
    return _digest(
        _PROJECTION_NAMESPACE,
        {
            "schema": SCHEMA,
            "claim_boundary": CLAIM_BOUNDARY,
            "split": split,
            "input_records_sha256": input_records_sha256,
            "policy_sha256": policy_sha256,
            "consolidation_corpus_sha256": consolidation_corpus_sha256,
            "consolidation_producer_sha256": consolidation_producer_sha256,
            "cognition_sha256s": tuple(concept.cognition_sha256 for concept in concepts),
            "concept_receipt_sha256s": tuple(receipt.sha256 for receipt in receipts),
            "cognition_corpus_sha256": cognition_corpus_sha256,
            "provenance_sha256": provenance_sha256,
            "graph_sha256": graph_sha256,
            "producer_receipt_sha256": producer_receipt_sha256,
        },
    )


@dataclass(frozen=True, slots=True)
class ConsolidatedKnowledgeCellProjectionV1:
    """All cognition rows and their separate corpus-bound provenance receipts."""

    split: str
    input_records_sha256: str
    policy_sha256: str
    consolidation_corpus_sha256: str
    consolidation_producer_sha256: str
    concepts: tuple[ProjectedKnowledgeConceptV1, ...]
    concept_receipts: tuple[ProjectedKnowledgeConceptReceiptV1, ...]
    cognition_corpus_sha256: str
    concept_count: int
    factor_association_count: int
    distinct_factor_count: int
    graph: structured_graph.RootedStructuredEventGraphV1
    producer_receipt: structured_graph.StructuredEventGraphProducerReceiptV1
    provenance_sha256: str
    sha256: str

    def authenticate_structure(self) -> None:
        if (
            self.split != ADMITTED_SPLIT
            or not _valid_sha256(self.input_records_sha256)
            or not _valid_sha256(self.policy_sha256)
            or not _valid_sha256(self.consolidation_corpus_sha256)
            or not _valid_sha256(self.consolidation_producer_sha256)
            or type(self.concepts) is not tuple
            or not 1 <= len(self.concepts) <= MAX_CONCEPTS
            or any(type(value) is not ProjectedKnowledgeConceptV1 for value in self.concepts)
            or type(self.concept_receipts) is not tuple
            or len(self.concept_receipts) != len(self.concepts)
            or any(
                type(value) is not ProjectedKnowledgeConceptReceiptV1
                for value in self.concept_receipts
            )
        ):
            raise ConsolidatedKnowledgeCellProjectionError(
                "consolidated knowledge projection shape changed"
            )
        for concept in self.concepts:
            concept.authenticate()
        for receipt in self.concept_receipts:
            receipt.authenticate()

        cognition_roots = tuple(concept.cognition_sha256 for concept in self.concepts)
        if (
            cognition_roots != tuple(sorted(cognition_roots))
            or len(set(cognition_roots)) != len(cognition_roots)
            or tuple(receipt.cognition_sha256 for receipt in self.concept_receipts)
            != cognition_roots
        ):
            raise ConsolidatedKnowledgeCellProjectionError(
                "cognition rows are not canonical and unique"
            )

        global_roots = {
            self.input_records_sha256,
            self.consolidation_corpus_sha256,
            self.consolidation_producer_sha256,
        }
        for receipt in self.concept_receipts:
            expected_roots = tuple(sorted({receipt.card_sha256, *global_roots}))
            if receipt.provenance_roots != expected_roots:
                raise ConsolidatedKnowledgeCellProjectionError("concept provenance roots changed")

        factor_associations = sum(len(concept.content_factors) for concept in self.concepts)
        distinct_factors = len(
            {factor for concept in self.concepts for factor in concept.content_factors}
        )
        if (
            factor_associations > MAX_FACTOR_ASSOCIATIONS
            or self.concept_count != len(self.concepts)
            or self.factor_association_count != factor_associations
            or self.distinct_factor_count != distinct_factors
            or self.cognition_corpus_sha256 != _cognition_corpus_sha256(self.concepts)
        ):
            raise ConsolidatedKnowledgeCellProjectionError(
                "consolidated knowledge projection counts changed"
            )

        expected_provenance = _provenance_sha256(
            input_records_sha256=self.input_records_sha256,
            policy_sha256=self.policy_sha256,
            consolidation_corpus_sha256=self.consolidation_corpus_sha256,
            consolidation_producer_sha256=self.consolidation_producer_sha256,
            receipts=self.concept_receipts,
        )
        if self.provenance_sha256 != expected_provenance:
            raise ConsolidatedKnowledgeCellProjectionError(
                "consolidated knowledge provenance root changed"
            )

        if type(self.graph) is not structured_graph.RootedStructuredEventGraphV1:
            raise TypeError("exact rooted structured event graph required")
        self.graph.authenticate()
        if self.graph != _build_graph(self.concepts):
            raise ConsolidatedKnowledgeCellProjectionError(
                "consolidated knowledge cognition graph changed"
            )
        if (
            type(self.producer_receipt)
            is not structured_graph.StructuredEventGraphProducerReceiptV1
        ):
            raise TypeError("exact structured graph producer receipt required")
        self.producer_receipt.authenticate()
        attestation = _producer_attestation_sha256(
            split=self.split,
            concepts=self.concepts,
            receipts=self.concept_receipts,
            cognition_corpus_sha256=self.cognition_corpus_sha256,
            provenance_sha256=self.provenance_sha256,
            graph_sha256=self.graph.sha256,
        )
        if (
            self.producer_receipt.source_sha256 != self.provenance_sha256
            or self.producer_receipt.graph_sha256 != self.graph.sha256
            or self.producer_receipt.producer_schema_sha256 != _producer_schema_sha256()
            or self.producer_receipt.producer_attestation_sha256 != attestation
            or self.sha256
            != _projection_sha256(
                split=self.split,
                input_records_sha256=self.input_records_sha256,
                policy_sha256=self.policy_sha256,
                consolidation_corpus_sha256=self.consolidation_corpus_sha256,
                consolidation_producer_sha256=self.consolidation_producer_sha256,
                concepts=self.concepts,
                receipts=self.concept_receipts,
                cognition_corpus_sha256=self.cognition_corpus_sha256,
                provenance_sha256=self.provenance_sha256,
                graph_sha256=self.graph.sha256,
                producer_receipt_sha256=self.producer_receipt.sha256,
            )
        ):
            raise ConsolidatedKnowledgeCellProjectionError(
                "consolidated knowledge producer receipt authentication failed"
            )

    def authenticate(
        self,
        consolidation: KnowledgeConsolidationV1,
        *,
        corpus: KnowledgeCorpus,
    ) -> None:
        """Prove exact full-card derivation from one corpus-bound consolidation."""

        self.authenticate_structure()
        _authenticate_bound_consolidation(consolidation, corpus)
        expected_pairs = tuple(
            sorted(
                ((_concept_from_card(card), card) for card in consolidation.cards),
                key=lambda pair: pair[0].cognition_sha256,
            )
        )
        expected_concepts = tuple(pair[0] for pair in expected_pairs)
        expected_receipts = tuple(
            _receipt_from_card(card, concept, consolidation) for concept, card in expected_pairs
        )
        if (
            self.split != consolidation.split
            or self.input_records_sha256 != consolidation.input_records_sha256
            or self.policy_sha256 != consolidation.policy_sha256
            or self.consolidation_corpus_sha256 != consolidation.corpus_sha256
            or self.consolidation_producer_sha256 != consolidation.producer_sha256
            or self.concepts != expected_concepts
            or self.concept_receipts != expected_receipts
        ):
            raise ConsolidatedKnowledgeCellProjectionError(
                "projection is not exactly derived from the bound consolidation"
            )


def _authenticate_bound_consolidation(
    consolidation: KnowledgeConsolidationV1,
    corpus: KnowledgeCorpus,
) -> None:
    if type(consolidation) is not KnowledgeConsolidationV1:
        raise TypeError("exact immutable KnowledgeConsolidationV1 required")
    if type(corpus) is not KnowledgeCorpus:
        raise TypeError("exact immutable KnowledgeCorpus required")
    try:
        consolidation.authenticate(corpus)
    except KnowledgeConsolidationError as error:
        raise ConsolidatedKnowledgeCellProjectionError(
            "corpus-bound knowledge consolidation authentication failed"
        ) from error


def _project_authenticated_consolidation(
    consolidation: KnowledgeConsolidationV1,
) -> ConsolidatedKnowledgeCellProjectionV1:
    if consolidation.split != ADMITTED_SPLIT:
        raise ConsolidatedKnowledgeCellProjectionError(
            "only authenticated acquire knowledge may enter cognition"
        )
    if not 1 <= len(consolidation.cards) <= MAX_CONCEPTS:
        raise ConsolidatedKnowledgeCellProjectionError(
            f"projection requires 1..{MAX_CONCEPTS} complete concept cards"
        )

    pairs = tuple(
        (
            _concept_from_card(card),
            card,
        )
        for card in consolidation.cards
    )
    canonical_pairs = tuple(sorted(pairs, key=lambda pair: pair[0].cognition_sha256))
    concepts = tuple(pair[0] for pair in canonical_pairs)
    if len({concept.cognition_sha256 for concept in concepts}) != len(concepts):
        raise ConsolidatedKnowledgeCellProjectionError(
            "distinct cards have duplicate cognition factor tuples"
        )
    factor_associations = sum(len(concept.content_factors) for concept in concepts)
    if factor_associations > MAX_FACTOR_ASSOCIATIONS:
        raise ConsolidatedKnowledgeCellProjectionError(
            "complete concept projection exceeds factor-association capacity"
        )
    receipts = tuple(
        _receipt_from_card(card, concept, consolidation) for concept, card in canonical_pairs
    )
    cognition_corpus_sha256 = _cognition_corpus_sha256(concepts)
    provenance_sha256 = _provenance_sha256(
        input_records_sha256=consolidation.input_records_sha256,
        policy_sha256=consolidation.policy_sha256,
        consolidation_corpus_sha256=consolidation.corpus_sha256,
        consolidation_producer_sha256=consolidation.producer_sha256,
        receipts=receipts,
    )
    graph = _build_graph(concepts)
    attestation = _producer_attestation_sha256(
        split=consolidation.split,
        concepts=concepts,
        receipts=receipts,
        cognition_corpus_sha256=cognition_corpus_sha256,
        provenance_sha256=provenance_sha256,
        graph_sha256=graph.sha256,
    )
    producer_receipt = structured_graph.make_structured_event_graph_producer_receipt_v1(
        source_sha256=provenance_sha256,
        graph_sha256=graph.sha256,
        producer_schema_sha256=_producer_schema_sha256(),
        producer_attestation_sha256=attestation,
    )
    distinct_factor_count = len(
        {factor for concept in concepts for factor in concept.content_factors}
    )
    projection_sha256 = _projection_sha256(
        split=consolidation.split,
        input_records_sha256=consolidation.input_records_sha256,
        policy_sha256=consolidation.policy_sha256,
        consolidation_corpus_sha256=consolidation.corpus_sha256,
        consolidation_producer_sha256=consolidation.producer_sha256,
        concepts=concepts,
        receipts=receipts,
        cognition_corpus_sha256=cognition_corpus_sha256,
        provenance_sha256=provenance_sha256,
        graph_sha256=graph.sha256,
        producer_receipt_sha256=producer_receipt.sha256,
    )
    result = ConsolidatedKnowledgeCellProjectionV1(
        split=consolidation.split,
        input_records_sha256=consolidation.input_records_sha256,
        policy_sha256=consolidation.policy_sha256,
        consolidation_corpus_sha256=consolidation.corpus_sha256,
        consolidation_producer_sha256=consolidation.producer_sha256,
        concepts=concepts,
        concept_receipts=receipts,
        cognition_corpus_sha256=cognition_corpus_sha256,
        concept_count=len(concepts),
        factor_association_count=factor_associations,
        distinct_factor_count=distinct_factor_count,
        graph=graph,
        producer_receipt=producer_receipt,
        provenance_sha256=provenance_sha256,
        sha256=projection_sha256,
    )
    result.authenticate_structure()
    return result


def project_consolidated_knowledge_to_cells_v1(
    consolidation: KnowledgeConsolidationV1,
    *,
    corpus: KnowledgeCorpus,
) -> ConsolidatedKnowledgeCellProjectionV1:
    """Project every authenticated acquire card without task or action inputs."""

    _authenticate_bound_consolidation(consolidation, corpus)
    result = _project_authenticated_consolidation(consolidation)
    result.authenticate(consolidation, corpus=corpus)
    return result


__all__ = [
    "ADMITTED_SPLIT",
    "CLAIM_BOUNDARY",
    "MAX_CONCEPTS",
    "MAX_FACTOR_ASSOCIATIONS",
    "SCHEMA",
    "ConsolidatedKnowledgeCellProjectionError",
    "ConsolidatedKnowledgeCellProjectionV1",
    "ProjectedKnowledgeConceptReceiptV1",
    "ProjectedKnowledgeConceptV1",
    "project_consolidated_knowledge_to_cells_v1",
]
