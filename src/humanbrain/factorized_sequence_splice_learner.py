"""The existing executed-action transaction with the factorized shared reader."""

from humanbrain.factorized_sequence_memory_learner import FactorizedSequenceMemoryLearner
from humanbrain.sequence_assembly_splice_learner import SequenceAssemblySpliceLearner


class FactorizedSequenceSpliceLearner(SequenceAssemblySpliceLearner):
    """No second executor or fitting path: reading and payloads use one reader."""

    def __init__(self, *, mode="learned", seed=733, reader=None):
        if mode not in self.MODES:
            raise ValueError("invalid factorized action mode")
        if reader is None:
            reader = FactorizedSequenceMemoryLearner(mode=self.MODES[mode], seed=seed)
        if not isinstance(reader, FactorizedSequenceMemoryLearner):
            raise ValueError("factorized actions require the same factorized reader")
        super().__init__(mode=mode, seed=seed, reader=reader)

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "factorized-sequence-splice-learner-v1",
            "reader": "factorized-sequence-memory-learner-v1",
            "publication": (
                "all action proposals then joint router/assembly/views then action copies"),
        })
        return metadata

    def load_state_dict(self, state_dict, strict=True, assign=False, *, work=None):
        self._require_idle()
        self._check(work)
        # Parent recursive loading bypasses the reader's custom loader. Prepare
        # its derived state before any parent/action/reader parameter publication.
        prepared = self.reader.prepare_state_views(state_dict, "reader.", work)
        result = super().load_state_dict(state_dict, strict=strict, assign=assign)
        self.reader.install_validated_views(prepared)
        return result
