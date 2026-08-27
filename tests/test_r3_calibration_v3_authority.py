from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from experiments.r3_pilot import calibration_v3_authority as authority_contract


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def test_authority_owns_one_immutable_canonical_protocol_snapshot(
    tmp_path: Path,
) -> None:
    protocol = {
        "nested": {"value": 1},
        "source_binding": {"blockers": [], "runnable": True},
    }
    frozen_bytes = b'{"nested":{"value":1},"source_binding":{"blockers":[],"runnable":true}}'
    frozen_sha256 = hashlib.sha256(frozen_bytes).hexdigest()

    authority = authority_contract.FrozenPlanAuthority(
        repository_root=tmp_path,
        protocol=protocol,
        protocol_sha256=frozen_sha256,
        source_bundle_identity_sha256="22" * 32,
    )

    protocol["nested"]["value"] = 2
    protocol["source_binding"]["blockers"].append("late mutation")
    first_read = authority.protocol
    first_read["nested"]["value"] = 3

    assert authority.protocol == {
        "nested": {"value": 1},
        "source_binding": {"blockers": [], "runnable": True},
    }
    assert authority.protocol_sha256 == frozen_sha256
    assert _canonical_bytes(authority.protocol) == frozen_bytes


def test_authority_rejects_a_digest_from_a_different_protocol_snapshot(
    tmp_path: Path,
) -> None:
    protocol = {"source_binding": {"blockers": [], "runnable": True}}

    with pytest.raises(
        authority_contract.FrozenPlanAuthorityError,
        match="does not match its canonical snapshot",
    ):
        authority_contract.FrozenPlanAuthority(
            repository_root=tmp_path,
            protocol=protocol,
            protocol_sha256="11" * 32,
            source_bundle_identity_sha256="22" * 32,
        )


@pytest.mark.parametrize(
    ("protocol", "protocol_sha256", "source_sha256"),
    [
        ({"value": 1.5}, "11" * 32, "22" * 32),
        ({1: "non-string key"}, "11" * 32, "22" * 32),
        ({"value": 1}, "not-a-digest", "22" * 32),
        ({"value": 1}, "11" * 32, "NOT-LOWERCASE"),
    ],
)
def test_authority_rejects_malformed_transport_values(
    tmp_path: Path,
    protocol: dict[object, object],
    protocol_sha256: str,
    source_sha256: str,
) -> None:
    with pytest.raises(authority_contract.FrozenPlanAuthorityError):
        authority_contract.FrozenPlanAuthority(
            repository_root=tmp_path,
            protocol=protocol,  # type: ignore[arg-type]
            protocol_sha256=protocol_sha256,
            source_bundle_identity_sha256=source_sha256,
        )
