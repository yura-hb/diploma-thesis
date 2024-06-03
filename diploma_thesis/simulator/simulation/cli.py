
from logging import Logger
from typing import Dict, List, Iterable

from utils import modified
from .jsp_dataset import JSPDataset
from .simulation import Simulation


def from_cli(prefix: str, parameters: List[Dict], logger: Logger) -> [Simulation]:
    if isinstance(parameters, dict):
        parameters = [parameters]

    simulations = list(flatten((_from_cli(param, logger) for param in parameters)))

    ids = dict()

    for index, simulation in enumerate(simulations):
        ids[simulation.simulation_id] = ids.get(simulation.simulation_id, 0) + 1

        simulation.update_name(prefix + simulation.simulation_id + '_' + str(ids[simulation.simulation_id]))
        simulation.update_index(index)

    return simulations


def _from_cli(parameters: Dict, logger: Logger) -> [Simulation]:
    from .multi_value import MultiValueCLITemplate

    key_to_class = {
        'simulation': Simulation,
        'jsp_dataset': JSPDataset,
        'multi_value': MultiValueCLITemplate,
    }

    if parameters['kind'] == 'mod':
        return _from_cli(modified(parameters['parameters']), logger)

    cls = key_to_class[parameters['kind']]

    result = cls.from_cli(name=parameters.get('name', ''), parameters=parameters['parameters'], logger=logger)

    if isinstance(result, list):
        return result

    return [result]


def flatten(items):
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x