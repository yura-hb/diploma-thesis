import gc
import traceback
import time
from typing import Dict

from joblib import Parallel, delayed

from utils.multi_value_cli import multi_value_cli
from .simulation import Simulation


def __run__(s: Dict):
    s = Simulation(s)

    print(f'Simulation started {s.parameters["name"]}')

    start = time.time()

    try:
        s.run()
    except Exception as e:
        print(f'Error in simulation {s.parameters["name"]}: {e}')
        print(traceback.format_exc())

    print(f'Simulation finished {s.parameters["name"]}. Elapsed time: {time.time() - start} seconds.')

    del s

    gc.collect()


class MultiSimulation:

    def __init__(self, parameters: Dict):
        self.parameters = parameters

    @property
    def workflow_id(self) -> str:
        return ''

    def run(self):
        parameters = self.__fetch_tasks__()
        parameters = self.__add_debug_info__(parameters)
        parameters = self.__fix_names__(parameters)

        print(f'Running {len(parameters)} simulations')

        n_workers = self.parameters.get('n_workers', -1)

        Parallel(
            n_jobs=n_workers
        )(delayed(__run__)(s) for s in parameters)

    def __fetch_tasks__(self):
        result: [Dict] = []

        for task in self.parameters['tasks']:
            match self.parameters['kind']:
                case 'task':
                    result += [task['parameters']]
                case 'multi_task':
                    result += multi_value_cli(task['parameters'], lambda p: p)
                case _:
                    raise ValueError(f"Unknown kind: {self.parameters['kind']}")

        return result

    def __add_debug_info__(self, simulations: [Dict]):
        result = simulations

        if self.parameters.get('debug', False):
            for index, _ in enumerate(result):
                result[index]['debug'] = True

        return result

    def __fix_names__(self, simulations: [Dict]):
        result = simulations

        for i, simulation in enumerate(result):
            result[i]['name'] = f"{simulation['name']}_{i}"

        return result
