import logging
import math
import multiprocessing
from functools import reduce
from typing import Dict
from typing import List

import pandas as pd
import simpy
import numpy as np
from tabulate import tabulate
from joblib import Parallel, delayed

from simulator import EvaluateConfiguration, EpisodicSimulator, Simulation
from workflow.candidates import from_cli as candidates_from_cli, Candidate
from workflow.criterion import from_cli as criterion_from_cli, Criterion, Direction, Scale
from .workflow import Workflow

reward_suffix = '_reward'


class Tournament(Workflow):

    def __init__(self, parameters: Dict):
        self.parameters = parameters

    def run(self):
        candidates = self.__make_candidates__()
        criteria = self.__make_criteria__()

        cpus = multiprocessing.cpu_count()

        result = Parallel(n_jobs=cpus)(delayed(self.__evaluate__)(candidate, criteria) for candidate in candidates)
        result = reduce(lambda x, y: x + y, result, [])
        result = pd.DataFrame(result)

        result = self.__reward__(result, criteria)

        self.__print_report__(result)

    def __make_candidates__(self) -> List[Candidate]:
        candidates = self.parameters['candidates']

        if not isinstance(candidates, list):
            candidates = [candidates]

        return reduce(lambda x, y: x + candidates_from_cli(y), candidates, [])

    def __make_criteria__(self):
        criteria = self.parameters['criteria']

        if not isinstance(criteria, list):
            criteria = [criteria]

        return reduce(lambda x, y: x + [criterion_from_cli(y)], criteria, [])

    def __evaluate__(self, candidate: Candidate, criteria: List[Criterion]):
        environment = simpy.Environment()
        logger = self.__make_logger__(name=candidate.name, environment=environment, log_stdout=True)
        logger.setLevel(logging.INFO)
        configuration = EvaluateConfiguration.from_cli(logger, self.parameters['simulator'])

        simulator = EpisodicSimulator(
            machine=candidate.machine,
            work_center=candidate.work_center,
            reward_model=None,
            environment=environment,
            logger=logger
        )

        simulator.evaluate(configuration)

        return self.__evaluate_criteria__(candidate, configuration.simulations, criteria)

    def __evaluate_criteria__(
        self, candidate: Candidate, simulations: List[Simulation], criteria: List[Criterion]
    ) -> List[Dict]:
        result = list()

        for simulation in simulations:
            record = {
                'candidate': candidate.name,
                'simulation': simulation.shop_floor_id,
            }

            statistics = simulation.shop_floor.statistics
            record |= {criterion.key: criterion.compute(statistics) for criterion in criteria}

            result.append(record)

        return result

    def __reward__(self, metrics: pd.DataFrame, criteria: List['Criterion']) -> pd.DataFrame:
        reward_parameters = self.parameters.get('reward', {})

        top_k = reward_parameters.get('top_k', -1)
        points = reward_parameters.get('points', 1)

        for simulation in metrics['simulation'].unique():
            mask = metrics['simulation'] == simulation
            simulation_metrics = metrics[mask]

            for criterion in criteria:
                reward_key = f'{criterion.key}{reward_suffix}'

                order = criterion.direction == Direction.minimize
                simulation_metrics = simulation_metrics.sort_values(criterion.key, ascending=order)
                top_k_index = simulation_metrics[:top_k].index

                simulation_metrics[reward_key] = 0.0

                final_value = simulation_metrics.loc[top_k_index[-1], criterion.key]

                performance = abs(simulation_metrics.loc[top_k_index, criterion.key] - final_value) / final_value

                match criterion.scale:
                    case Scale.linear:
                        pass
                    case Scale.log:
                        performance = np.log(1 + performance)
                    case Scale.exp:
                        performance = np.exp(performance)

                performance -= performance.min()
                performance /= performance.sum()

                performance *= points

                simulation_metrics.loc[top_k_index, reward_key] = performance * criterion.weight
                metrics.loc[mask, reward_key] = simulation_metrics[reward_key]

        return metrics

    def __print_report__(self, report: pd.DataFrame):
        reward_columns = [column for column in report.columns if column.endswith(reward_suffix)]

        rewards = report.groupby('candidate')[reward_columns].sum().sort_values(reward_columns, ascending=False)

        rewards['total_sum'] = rewards[reward_columns].sum(axis=1)
        rewards = rewards.sort_values('total_sum', ascending=False)

        print(tabulate(rewards, headers='keys', tablefmt='pretty'))
