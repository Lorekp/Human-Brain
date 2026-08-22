from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from humanbrain.cells import AllenGlif3Parameters, AllenGlif5Parameters
from humanbrain.human_cell_identity import (
    HUMAN_GLIF3_IDENTITY,
    HUMAN_GLIF5_IDENTITY,
)
from humanbrain.procedural_topology import ProceduralTopology, TopologyManifest


ROOT = Path(__file__).resolve().parents[1]


def test_pinned_human_cell_identities_are_the_provenance_source() -> None:
    assert HUMAN_GLIF3_IDENTITY.to_record() == {
        "model_id": 503612028,
        "specimen_id": 488392806,
        "model_name": "3 LIF + Afterspike Currents (LIF-ASC)_H15.06.017.02.02.01",
        "species": "Homo sapiens",
        "region": "middle temporal gyrus",
        "config_sha256": (
            "57e20326d27146c3afbd412ca3ba0bcaee67a22f27729292b36666b9ff36c889"
        ),
    }
    assert HUMAN_GLIF5_IDENTITY.to_record() == {
        "model_id": 523000114,
        "specimen_id": 508378649,
        "model_name": (
            "5 LIF-R-ASC + Threshold Adaptation "
            "(LIF-R-ASC-A)_H16.06.004.01.05.02"
        ),
        "species": "Homo sapiens",
        "region": "middle temporal gyrus",
        "config_sha256": (
            "69fe726023e3289ac85857ea9125c55fee1618891e7b65403784c1b961bad149"
        ),
    }


def test_cells_and_procedural_descriptors_use_the_pinned_identities() -> None:
    assert AllenGlif3Parameters().model_id == HUMAN_GLIF3_IDENTITY.model_id
    assert AllenGlif3Parameters().specimen_id == HUMAN_GLIF3_IDENTITY.specimen_id
    assert AllenGlif5Parameters().model_id == HUMAN_GLIF5_IDENTITY.model_id
    assert AllenGlif5Parameters().specimen_id == HUMAN_GLIF5_IDENTITY.specimen_id

    descriptor = ProceduralTopology(
        TopologyManifest(17, 4, "random", 71)
    ).describe_cell(0)
    assert descriptor.intrinsic_model == HUMAN_GLIF5_IDENTITY


def test_identity_hashes_match_the_pinned_allen_configuration_bytes() -> None:
    assert sha256(
        (ROOT / "data/allen/503612028_neuron_config.json").read_bytes()
    ).hexdigest() == HUMAN_GLIF3_IDENTITY.config_sha256
    assert sha256(
        (ROOT / "data/allen/523000114_neuron_config.json").read_bytes()
    ).hexdigest() == HUMAN_GLIF5_IDENTITY.config_sha256
