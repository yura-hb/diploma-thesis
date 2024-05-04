
from .action_selector import ActionSelector
from .greedy import Greedy
from .epsilon_greedy import EpsilonGreedy
from .epsilon_sample import EpsilonSample
from .sample import Sample
from .uniform import Uniform
from .phase_selector import PhaseSelector

from functools import partial
from utils import from_cli

key_to_cls = {
    'epsilon_greedy': EpsilonGreedy,
    'epsilon_sample': EpsilonSample,
    'greedy': Greedy,
    'sample': Sample,
    'uniform': Uniform,
    'phase_selector': PhaseSelector
}


from_cli = partial(from_cli, key_to_class=key_to_cls)