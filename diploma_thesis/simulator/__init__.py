from functools import partial

from utils import from_cli
from .configuration import RunConfiguration, EvaluateConfiguration
from .configuration import run_configuration_from_cli, evaluate_configuration_from_cli
from .episodic import EpisodicSimulator
from .simulation import Simulation
from .simulator import Simulator, RewardCache
from .td import TDSimulator

key_to_class = {
    "episodic": EpisodicSimulator,
    "td": TDSimulator
}


from_cli = partial(from_cli, key_to_class=key_to_class)
