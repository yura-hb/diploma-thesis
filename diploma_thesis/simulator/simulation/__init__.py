from typing import Dict, List
from logging import Logger
from functools import reduce

from .simulation import Simulation
from .multi_value import MultiValueCLITemplate


key_to_class = {
    'simulation': Simulation,
    'multi_value': MultiValueCLITemplate,
}


def from_cli(prefix: str, parameters: List[Dict], logger: Logger) -> [Simulation]:
    if isinstance(parameters, dict):
        parameters = [parameters]

    simulations = reduce(lambda x, y: x + _from_cli(y, logger), parameters, [])

    ids = dict()

    for index, simulation in enumerate(simulations):
        ids[simulation.simulation_id] = ids.get(simulation.simulation_id, 0) + 1

        simulation.update_name(prefix + simulation.simulation_id + '_' + str(ids[simulation.simulation_id]))
        simulation.update_index(index)

    return simulations


def _from_cli(parameters: Dict, logger: Logger) -> [Simulation]:
    cls = key_to_class[parameters['kind']]

    result = cls.from_cli(name=parameters.get('name', ''), parameters=parameters['parameters'], logger=logger)

    if isinstance(result, list):
        return result

    return [result]
