
from functools import partial

from agents.utils.memory import from_cli as memory_from_cli
from agents.utils.nn import Loss, Optimizer

from utils import from_cli as _from_cli
from agents.utils.rl.ddqn import DoubleDeepQTrainer
from agents.utils.rl.dqn import DeepQTrainer
from agents.utils.return_estimator import from_cli as return_estimator_from_cli
from .rl import RLTrainer

key_to_class = {
    'dqn': DeepQTrainer,
    'ddqn': DoubleDeepQTrainer,
}


def from_cli(parameters):
    _parameters = parameters['parameters']

    memory = memory_from_cli(_parameters['memory'])
    loss = Loss.from_cli(_parameters['loss'])
    optimizer = Optimizer.from_cli(_parameters['optimizer'])
    return_estimator = return_estimator_from_cli(_parameters['return'])

    return partial(
        _from_cli,
        key_to_class=key_to_class,
        memory=memory, loss=loss,
        optimizer=optimizer,
        return_estimator=return_estimator
    )(parameters)
