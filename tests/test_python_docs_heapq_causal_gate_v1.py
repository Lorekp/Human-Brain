from __future__ import annotations

from experiments.r3_pilot import common_knowledge_action_memory_v1 as knowledge_action
from experiments.r3_pilot import run_python_docs_heapq_causal_gate_v1 as gate


def test_preregistration_authenticates_without_spending_gate(monkeypatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("construction authentication spent the causal gate")

    monkeypatch.setattr(
        knowledge_action.FrozenCommonKnowledgeActionMemoryV1,
        "recall",
        forbidden,
    )
    monkeypatch.setattr(
        knowledge_action.KnowledgePatchEnvironmentV1,
        "materialize_selected",
        forbidden,
    )
    monkeypatch.setattr(
        knowledge_action.KnowledgePatchEnvironmentV1,
        "execute_selected",
        forbidden,
    )

    manifest = gate.load_manifest()
    roots = gate.authenticate_preregistration(manifest)

    assert roots == {
        "candidate_projection_sha256": (
            "a4f6c738492d0b8024ca7ed00ca81627745ed540fd8766208ea15cd8900c800e",
            "dddcd52fdb117557f969dc8490b718b339debc6631a7f6bc10386fbb0e59204e",
        ),
        "candidate_set_sha256": (
            "c8e9f2d2220d59f7e2698d06ee002a798bc0fe8f7fb84041fa2a83b012467d04"
        ),
        "manifest_sha256": (
            "b3420802d1f773b49ecb9680a3019cdff8411a741cffecac4dce8e5073edc4cd"
        ),
        "task_sha256": (
            "4b053735f618d049ede6eb067cf617cde86fc3317abe9d1c7fff0c8132ef0ab4"
        ),
    }
