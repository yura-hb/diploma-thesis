
import os

from typing import Dict

from joblib import Parallel, delayed

from utils import multi_value_cli, task
from .simulation import Simulation


@task(lambda s: s['name'])
def __run__(s: Dict):
    s = Simulation(s)

    s.run()


class MultiSimulation:

    def __init__(self, parameters: Dict):
        self.parameters = parameters

    @property
    def workflow_id(self) -> str:
        return ''

    def run(self):
        parameters = self.__fetch_tasks__()
        parameters = self.__add_debug_info__(parameters)
        parameters = self.__append_output_dir__(parameters)
        parameters = self.__fix_names__(parameters)

        print(f'Running {len(parameters)} simulations')

        n_workers = self.parameters.get('n_workers', -1)

        Parallel(
            n_jobs=n_workers,
        )(delayed(__run__)(s) for s in parameters)

    def __fetch_tasks__(self):
        result: [Dict] = []

        for simulation in self.parameters['tasks']:
            match self.parameters['kind']:
                case 'task':
                    result += [simulation['parameters']]
                case 'multi_task':
                    result += multi_value_cli(simulation['parameters'], lambda p: p)
                case _:
                    raise ValueError(f"Unknown kind: {self.parameters['kind']}")

        return result

    def __add_debug_info__(self, simulations: [Dict]):
        result = simulations

        if self.parameters.get('debug', False):
            for index, _ in enumerate(result):
                result[index]['debug'] = True

        return result

    def __append_output_dir__(self, simulations: [Dict]):
        result = simulations

        output_dir = self.parameters.get('output_dir')

        if output_dir:
            for index, _ in enumerate(result):
                result[index]['output_dir'] = os.path.join(output_dir, result[index]['output_dir'])

        return result

    def __fix_names__(self, simulations: [Dict]):
        result = simulations

        for i, simulation in enumerate(result):
            result[i]['name'] = f"{simulation['name']}_{i}"

        return result
