
import tqdm

from .simulation import Simulation
from utils.multi_value_cli import multi_value_cli
from typing import Dict
from joblib import Parallel, delayed


class MultiSimulation:

    def __init__(self, parameters: Dict):
        self.parameters = parameters

    def run(self):
        simulations = self.__fetch_tasks__()

        self.__add_debug_info__(simulations)
        self.__fix_names__(simulations)

        n_workers = self.parameters.get('n_workers', -1)

        def __run__(s: Simulation):
            try:
                s.run()
            except Exception as e:
                print(f'Error in simulation {s.parameters["name"]}: {e}')

            return s

        iter = Parallel(
            n_jobs=n_workers,
            backend='loky',
            return_as='generator',
            prefer='processes',
        )(delayed(lambda s: __run__(s))(s) for s in simulations)

        for s in tqdm.tqdm(iter, total=len(simulations)):
            print(f'Simulation finished {s.parameters["name"]}')

    def __fetch_tasks__(self):
        result: [Simulation] = []

        for task in self.parameters['tasks']:
            match self.parameters['kind']:
                case 'task':
                    result += [Simulation(task['parameters'])]
                case 'multi_task':
                    result += multi_value_cli(task['parameters'], lambda p: Simulation(p))
                case _:
                    raise ValueError(f"Unknown kind: {self.parameters['kind']}")

        return result

    def __add_debug_info__(self, simulations: [Simulation]):
        if self.parameters.get('debug', False):
            for simulation in simulations:
                simulation.parameters['debug'] = True

    def __fix_names__(self, simulations: [Simulation]):
        for i, simulation in enumerate(simulations):
            simulation.parameters['name'] = f"{simulation.parameters['name']}_{i}"
