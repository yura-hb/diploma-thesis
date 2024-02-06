
from dataclasses import dataclass
from typing import Any


class TDSimulator:
    """
    A simulator, which launches several long-term simulations in parallel. Rewards are added to agent memory as soon
    as they can be calculated.
    """

    @dataclass
    class Configuration:
        episode_count: int
        parallel_environments: int
        terminating_condition: 'Any'
        return_estimation: 'Any'

