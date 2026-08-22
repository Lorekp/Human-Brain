"""Immutable provenance identities for the pinned human Allen cell fits."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class HumanCellIdentity:
    model_id: int
    specimen_id: int
    model_name: str
    species: str
    region: str
    config_sha256: str

    def to_record(self) -> dict[str, int | str]:
        return {
            "model_id": self.model_id,
            "specimen_id": self.specimen_id,
            "model_name": self.model_name,
            "species": self.species,
            "region": self.region,
            "config_sha256": self.config_sha256,
        }


HUMAN_GLIF3_IDENTITY = HumanCellIdentity(
    model_id=503612028,
    specimen_id=488392806,
    model_name="3 LIF + Afterspike Currents (LIF-ASC)_H15.06.017.02.02.01",
    species="Homo sapiens",
    region="middle temporal gyrus",
    config_sha256=(
        "57e20326d27146c3afbd412ca3ba0bcaee67a22f27729292b36666b9ff36c889"
    ),
)

HUMAN_GLIF5_IDENTITY = HumanCellIdentity(
    model_id=523000114,
    specimen_id=508378649,
    model_name=(
        "5 LIF-R-ASC + Threshold Adaptation "
        "(LIF-R-ASC-A)_H16.06.004.01.05.02"
    ),
    species="Homo sapiens",
    region="middle temporal gyrus",
    config_sha256=(
        "69fe726023e3289ac85857ea9125c55fee1618891e7b65403784c1b961bad149"
    ),
)


__all__ = ["HUMAN_GLIF3_IDENTITY", "HUMAN_GLIF5_IDENTITY", "HumanCellIdentity"]
