from .simulation import Simulation
from .dynamic_simulation import DynamicSimulation
from typing import Dict, List
from logging import Logger


key_to_class = {
    'dynamic': DynamicSimulation
}


def from_cli(logger: Logger, parameters: Dict):
    cls = key_to_class[parameters['kind']]

    return cls.from_cli(name=parameters.get('name', ''), logger=logger, parameters=parameters['parameters'])


def from_cli_list(logger: Logger, parameters: List[Dict]):
    simulation_names = [parameter.get('name', '') for parameter in parameters]
    names = dict()

    for idx, key in enumerate(simulation_names):
        if key in names:
            names[key] += 1
            key += '[' + str(names[key]) + ']'
        else:
            names[key] = 1
            key += '[0]'

        parameters[idx]['name'] = key

    return [from_cli(logger, parameter) for parameter in parameters]
