
from .action_selector import ActionSelector
from .epsilon_greedy import EpsilonGreedy
from .upper_confidence_bound import UpperConfidenceBound
from .sample import Sample
from .uniform import Uniform
from .phase_selector import PhaseSelector


key_to_cls = {
    'epsilon_greedy': EpsilonGreedy,
    'upper_confidence_bound': UpperConfidenceBound,
    'sample': Sample,
    'uniform': Uniform,
    'phase_selector': PhaseSelector
}


def from_cli(parameters) -> ActionSelector:
    cls = key_to_cls[parameters['kind']]

    return cls.from_cli(parameters.get('parameters', {}))