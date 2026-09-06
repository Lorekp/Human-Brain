"""Teacher-free arm identity and model construction for the fixed comparison.

Importing this module creates no tensors. The same factory supplies persistent
online parents and strictly restored isolated endpoint instances.
"""

ARMS = ("learned", "action_frozen", "slow_frozen", "memory_disabled", "marginal_reference")


def reader_policy(arm):
    if arm not in ARMS:
        raise ValueError("unknown competitive splice comparison arm")
    return "marginal_free" if arm == "marginal_reference" else "competitive_free"


def new_model(arm, *, seed=733):
    """Construct one reader/action system with its declared free policy and gates."""
    policy = reader_policy(arm)
    from humanbrain.competitive_local_memory_learner import CompetitiveLocalMemoryLearner
    from humanbrain.local_memory_learner import LocalMemoryLearner
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner

    mode = "learned" if arm == "marginal_reference" else arm
    reader_type = LocalMemoryLearner if policy == "marginal_free" else CompetitiveLocalMemoryLearner
    reader = reader_type(mode=NeuronalSpliceLearner.MODES[mode], seed=seed)
    return NeuronalSpliceLearner(mode=mode, seed=seed, reader=reader)
