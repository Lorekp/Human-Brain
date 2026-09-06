"""Teacher-free five-arm factory with separate recall and feature identities.

Importing the factory creates no tensors. Every arm uses the same neuronal
competition; four use categorical coincidence and one the original graded keys.
"""

ARMS = ("learned", "action_frozen", "slow_frozen", "memory_disabled", "graded_reference")


def reader_policy(arm):
    if arm not in ARMS:
        raise ValueError("unknown coincidence splice comparison arm")
    return "competitive_free"


def address_frontend(arm):
    reader_policy(arm)
    return "graded_random_code" if arm == "graded_reference" else "coincidence"


def new_model(arm, *, seed=733):
    """Construct one persistent reader/action system, without loading a prior."""
    frontend = address_frontend(arm)
    from humanbrain.coincidence_local_memory_learner import CoincidenceLocalMemoryLearner
    from humanbrain.competitive_local_memory_learner import CompetitiveLocalMemoryLearner
    from humanbrain.neuronal_splice_learner import NeuronalSpliceLearner

    mode = "learned" if arm == "graded_reference" else arm
    reader_type = (CompetitiveLocalMemoryLearner if frontend == "graded_random_code"
                   else CoincidenceLocalMemoryLearner)
    reader = reader_type(mode=NeuronalSpliceLearner.MODES[mode], seed=seed)
    return NeuronalSpliceLearner(mode=mode, seed=seed, reader=reader)
