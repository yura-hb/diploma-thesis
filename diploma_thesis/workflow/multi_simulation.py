
import os

from typing import Dict

from joblib import Parallel, delayed

from utils import multi_value_cli, task
from .simulation import Simulation
from .workflow import *


@task(lambda s: s['name'])
def __run__(s: Dict):
    s = Simulation(s)

    s.run()


class MultiSimulation(Workflow):

    def __init__(self, parameters: Dict):
        self.parameters = parameters

    @property
    def workflow_id(self) -> str:
        return ''

    def run(self):
        def __merge__(key, lhs, rhs):
            if key == 'output_dir':
                return os.path.join(rhs, lhs)

            return rhs

        parameters = self.__fetch_tasks__()
        parameters = self.__passthrough_parameters__(dict(
            debug=False,
            output_dir='',
            store_run_statistics=False
        ), parameters, merge=__merge__)
        parameters = self.__fix_names__(parameters)

        print(f'Running {len(parameters)} simulations')

        n_workers = self.parameters.get('n_workers', -1)

        torch.set_num_threads(n_workers)
        torch.set_num_interop_threads(n_workers)

        Parallel(n_jobs=n_workers, backend='loky')(delayed(__run__)(s) for s in parameters)

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

    def __passthrough_parameters__(self, values, simulations: [Dict], merge=lambda key, lhs, rhs: rhs):
        result = simulations

        for key, default in values.items():
            value = self.parameters.get(key, default)

            if value is not None:
                for index, _ in enumerate(result):
                    result[index][key] = merge(key, result[index].get(key, default), value)

        return result

    def __fix_names__(self, simulations: [Dict]):
        result = simulations

        for i, simulation in enumerate(result):
            result[i]['name'] = f"{simulation['name']}_{i}"

        return result
