"""The existing action/feedback loop with a request-conditioned neuronal reader."""

from collections.abc import Mapping

import torch

from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner
from humanbrain.request_conditioned_memory_learner import RequestConditionedMemoryLearner


class RequestConditionedSpliceLearner(NeuronalSpliceLearner):
    """Preserve action custody and validate fixed context wiring before loading."""

    def __init__(self, *, mode="learned", seed=733, reader=None):
        if mode not in self.MODES:
            raise ValueError("invalid request-conditioned splice mode")
        if reader is None:
            reader = RequestConditionedMemoryLearner(mode=self.MODES[mode], seed=seed)
        if not isinstance(reader, RequestConditionedMemoryLearner):
            raise TypeError("request-conditioned actions require the same conditioned reader")
        super().__init__(mode=mode, seed=seed, reader=reader)

    def _check(self):
        super()._check()
        self.reader._check_conditioned_parameters()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        if not isinstance(state_dict, Mapping):
            raise ValueError("checkpoint must be a state mapping")
        expected = self.state_dict()
        for name in self.reader.FIXED_STATE_NAMES:
            key = "reader." + name
            if key not in state_dict or not isinstance(state_dict[key], torch.Tensor):
                raise ValueError("checkpoint omits fixed reader wiring")
            if not torch.equal(state_dict[key], expected[key]):
                raise ValueError("checkpoint changes fixed reader wiring")
        # The base validates every key, tensor shape, dtype and metadata before
        # publishing any value. Its existing pending-action check remains active.
        return super().load_state_dict(state_dict, strict=strict, assign=assign)
