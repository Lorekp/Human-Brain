"""The existing actual-action transaction with the same recognition reader."""

from humanbrain.factorized_sequence_splice_learner import FactorizedSequenceSpliceLearner
from humanbrain.recognition_sequence_memory_learner import RecognitionSequenceMemoryLearner


class RecognitionSequenceSpliceLearner(FactorizedSequenceSpliceLearner):
    """One reader for language and payload, joint staged correction after execution."""

    def __init__(self, *, mode="learned", seed=733, reader=None):
        if mode not in self.MODES:
            raise ValueError("invalid recognition action mode")
        if reader is None:
            reader = RecognitionSequenceMemoryLearner(mode=self.MODES[mode], seed=seed)
        if not isinstance(reader, RecognitionSequenceMemoryLearner):
            raise ValueError("recognition actions require the same recognition reader")
        super().__init__(mode=mode, seed=seed, reader=reader)

    def get_extra_state(self):
        metadata = super().get_extra_state()
        metadata.update({
            "schema": "recognition-sequence-splice-learner-v1",
            "reader": "recognition-sequence-memory-learner-v1",
            "publication": (
                "all action proposals then joint recognition/assembly/wiring then action copies"),
        })
        return metadata
