import os.path
from functools import reduce
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import simpy
import torch
import tqdm
from joblib import Parallel, delayed
from tabulate import tabulate

from environment import Statistics
from simulator import EvaluateConfiguration, EpisodicSimulator, Simulation
from tape import TapeModel, NoMachineReward, NoWorkCenterReward
from utils import task
from workflow.candidates import from_cli as candidates_from_cli, Candidate
from workflow.criterion import from_cli as criterion_from_cli, Criterion, Direction, Scale
from .workflow import Workflow

reward_suffix = '_reward'


@task(lambda _, candidate, *args: candidate.name)
def __evaluate__(tournament: 'Tournament', candidate: Candidate, criteria: List[Criterion], output_dir: str):
    environment = simpy.Environment()
    candidate_output_dir = os.path.join(output_dir, 'candidates', candidate.name)

    log_file = os.path.join(candidate_output_dir, 'log.txt')
    logger = tournament.__make_logger__(name=candidate.name, filename=log_file, log_stdout=False)

    configuration = EvaluateConfiguration.from_cli(parameters=tournament.parameters['simulator'], logger=logger)

    candidate.machine.with_logger(logger)
    candidate.work_center.with_logger(logger)

    simulator = EpisodicSimulator(
        machine=candidate.machine,
        work_center=candidate.work_center,
        tape_model=TapeModel(NoMachineReward(), NoWorkCenterReward()),
    )

    simulator.with_logger(logger)

    if not tournament.debug:
        simulator.evaluate(environment=environment, config=configuration)

        return tournament.__evaluate_criteria__(candidate, configuration.simulations, criteria, candidate_output_dir)

    return {}


class Tournament(Workflow):

    def __init__(self, parameters: Dict):
        self.parameters = parameters

    @property
    def workflow_id(self) -> str:
        return ''

    @property
    def debug(self) -> bool:
        return self.parameters.get('debug', False)

    def run(self):
        candidates = self.__make_candidates__()
        criteria = self.__make_criteria__()
        output_dir = self.__make_output_dir__(self.parameters['name'], self.parameters['output_dir'])

        print(f'Evaluating {len(candidates)} candidates')

        cpus = self.parameters['n_workers']

        iter = Parallel(n_jobs=cpus, return_as='generator')(
            delayed(__evaluate__)(self, candidate, criteria, output_dir) for candidate in candidates
        )

        result = []

        for metrics in tqdm.tqdm(iter, total=len(candidates)):
            result += metrics

        result = pd.DataFrame(result)
        result = self.__reward__(result, criteria)

        self.__save_result__(result, output_dir=output_dir)
        self.__save_report__(result, output_dir=output_dir)

    def __evaluate_criteria__(self,
                              candidate: Candidate,
                              simulations: List[Simulation],
                              criteria: List[Criterion],
                              output_dir: str) -> List[Dict]:
        result = list()

        for simulation in simulations:
            record = {
                'candidate': candidate.name,
                'run': simulation.simulation_id,
            }

            statistics = simulation.shop_floor.statistics
            record |= {criterion.key: criterion.compute(statistics) for criterion in criteria}

            self.__save_statistics__(simulation, statistics, output_dir)

            result.append(record)

        return result

    def __reward__(self, metrics: pd.DataFrame, criteria: List['Criterion']) -> pd.DataFrame:
        reward_parameters = self.parameters.get('tape', {})

        top_k = reward_parameters.get('top_k', -1)
        points = reward_parameters.get('points', 1)

        for simulation in metrics['run'].unique():
            mask = metrics['run'] == simulation
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

    def __save_result__(self, result: pd.DataFrame, output_dir: str):
        result_output_file = os.path.join(output_dir, 'result.csv')
        result.to_csv(result_output_file)

    def __save_report__(self, result: pd.DataFrame, output_dir: str):
        report_output_file = os.path.join(output_dir, 'report.txt')
        report = self.__make_report__(result)

        with open(report_output_file, 'w') as f:
            f.write(str(report))

    def __make_report__(self, report: pd.DataFrame):
        reward_columns = [column for column in report.columns if column.endswith(reward_suffix)]

        rewards = report.groupby('candidate')[reward_columns].sum().sort_values(reward_columns, ascending=False)

        rewards['total_sum'] = rewards[reward_columns].sum(axis=1)
        rewards = rewards.sort_values('total_sum', ascending=False)

        return tabulate(rewards, headers='keys', tablefmt='pretty')

    def __save_statistics__(self, simulation: Simulation, statistics: Statistics, output_dir: str):
        simulation_output_dir = os.path.join(output_dir, simulation.name)

        if not os.path.exists(simulation_output_dir):
            os.makedirs(simulation_output_dir)

        statistics_output_dir = os.path.join(simulation_output_dir, 'statistics')
        statistics.save(statistics_output_dir)

        report_output_file = os.path.join(simulation_output_dir, 'report.txt')
        report = statistics.report()

        with open(report_output_file, 'w') as f:
            f.write(str(report))
