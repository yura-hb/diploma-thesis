
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
        parameters = self.__passthrough_parameters__(dict(
            debug=False,
            output_dir='',
            store_run_statistics=False
        ), parameters)
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

    def __passthrough_parameters__(self, values, simulations: [Dict]):
        result = simulations

        for key, default in values.items():
            value = self.parameters.get(key, default)

            if value is not None:
                for index, _ in enumerate(result):
                    result[index][key] = value

        return result

    def __fix_names__(self, simulations: [Dict]):
        result = simulations

        for i, simulation in enumerate(result):
            result[i]['name'] = f"{simulation['name']}_{i}"

        return result
