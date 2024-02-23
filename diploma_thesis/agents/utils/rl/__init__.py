
from functools import partial

from agents.utils.memory import from_cli as memory_from_cli
from agents.utils.nn import LossCLI, OptimizerCLI

from utils import from_cli as _from_cli
from agents.utils.rl.ddqn import DoubleDeepQTrainer
from agents.utils.rl.dqn import DeepQTrainer
from .rl import RLTrainer

key_to_class = {
    'dqn': DeepQTrainer,
    'ddqn': DoubleDeepQTrainer,
}


def from_cli(parameters):
    _parameters = parameters['parameters']

    memory = memory_from_cli(_parameters['memory'])
    loss = LossCLI.from_cli(_parameters['loss'])
    optimizer = OptimizerCLI.from_cli(_parameters['optimizer'])

    return partial(_from_cli, key_to_class=key_to_class, memory=memory, loss=loss, optimizer=optimizer)(parameters)
