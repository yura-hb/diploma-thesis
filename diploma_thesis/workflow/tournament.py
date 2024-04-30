import logging
import os.path
import traceback
from functools import reduce
from typing import Dict, List

import numpy as np
import pandas as pd
import simpy
import tqdm
from joblib import Parallel, delayed
from tabulate import tabulate
from tqdm import tqdm

from environment import Statistics
from simulator import EvaluateConfiguration, EpisodicSimulator, Simulation
from simulator.graph import GraphModel
from simulator.graph.graph_model import Configuration as GraphRunConfiguration
from simulator.graph.transition import No as NoTransitionModel
from simulator.tape import TapeModel, NoMachineReward, NoWorkCenterReward
from utils import task
from workflow.candidates import from_cli as candidates_from_cli, Candidate
from workflow.criterion import from_cli as criterion_from_cli, Criterion, Direction, Scale
from .workflow import Workflow

reward_suffix = '_reward'
result_filename = 'result.csv'
candidates_dir = 'candidates'
statistics_filename = 'statistics'


@task(lambda _, candidate, *args: candidate.name)
def __evaluate__(tournament: 'Tournament',
                 candidate: Candidate,
                 criteria: List[Criterion],
                 configuration: EvaluateConfiguration,
                 logger: logging.Logger,
                 output_dir: str,
                 threads: int):
    import torch
    import random

    # torch.set_num_threads(threads)
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    environment = simpy.Environment()
    candidate_output_dir = os.path.join(output_dir, candidates_dir, candidate.name)

    graph_model = GraphModel(transition_model=NoTransitionModel(forward_transition=None, schedule_transition=None),
                             configuration=GraphRunConfiguration(memory=1))

    if 'graph' in candidate.parameters:
        graph_model = GraphModel.from_cli(candidate.parameters['graph'])

    mods = []

    if 'machine_agent' in candidate.parameters:
        for m in candidate.parameters['machine_agent']['parameters']['mods']:
            if isinstance(m, str):
                if 'cuda' not in 'm':
                    mods.append(m)
                continue

            if isinstance(m, list):
                mods += [m_ for m_ in m if 'cuda' not in m_]
                continue

        candidate.parameters['machine_agent']['parameters']['mods'] = mods

    try:
        machine, work_center = candidate.load()
    except:
        traceback.print_exc()
        print(f'Error loading candidate {candidate.name}')
        return []

    machine.with_logger(logger)
    work_center.with_logger(logger)

    simulator = EpisodicSimulator(
        machine=machine,
        work_center=work_center,
        tape_model=TapeModel(NoMachineReward(), NoWorkCenterReward()),
        graph_model=graph_model,
    )

    simulator.with_logger(logger)

    if not tournament.debug:
        result = []

        def on_simulation_end(simulation: Simulation, *args, **kwargs):
            nonlocal result

            try:
                result += tournament.__evaluate_criteria__(candidate,
                                                           simulations=[simulation],
                                                           criteria=criteria,
                                                           output_dir=candidate_output_dir)
            except:
                print(f'Skip simulation {simulation.simulation_id} due to error')
                traceback.print_exc()

        simulator.evaluate(environment=environment, config=configuration, on_simulation_end=on_simulation_end)

        return result

    return []


class Tournament(Workflow):

    def __init__(self, parameters: Dict):
        self.parameters = parameters

    @property
    def workflow_id(self) -> str:
        return ''

    def run_log_file(self, output_dir):
        if not self.parameters.get('log_run', False):
            return None

        return os.path.join(output_dir, 'log.txt')

    @property
    def store_run_statistics(self):
        return self.parameters.get('store_run_statistics', False)

    @property
    def should_update(self):
        return self.parameters.get('update', False)

    @property
    def debug(self) -> bool:
        return self.parameters.get('debug', False)

    def run(self):
        candidates = self.__make_candidates__()
        criteria = self.__make_criteria__()
        output_dir = self.__make_output_dir__(
            self.parameters['name'], self.parameters['output_dir'], remove=not self.should_update
        )

        logger = self.__make_logger__(name='Tournament',
                                      filename=self.run_log_file(output_dir),
                                      log_stdout=self.parameters.get('log_stdout', False))

        configuration = EvaluateConfiguration.from_cli(parameters=self.parameters['simulator'], logger=logger)

        n_workers = self.parameters['n_workers']
        threads = self.__get_n_threads__(n_workers, self.parameters.get('n_threads'))

        result = []

        if self.should_update:
            if filtered := self.__load_and_filter_current_candidates__(candidates, criteria, output_dir):
                candidates = filtered[0]
                result += filtered[1]

        print(f'Evaluating {len(candidates)} candidates')

        if len(candidates) > 0:
            iter = Parallel(n_jobs=n_workers, return_as='generator')(
                delayed(__evaluate__)(self, candidate, criteria, configuration, logger, output_dir, threads)
                for candidate in candidates
            )

            for metrics in tqdm(iter, total=len(candidates)):
                if metrics is not None:
                    result += metrics

        result = pd.DataFrame(result)

        self.__save_result__(result, output_dir=output_dir)

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
            statistics = simulation.shop_floor.statistics
            record = self.__evaluate_criteria_for_statistics__(candidate, simulation.simulation_id, statistics, criteria)

            self.__save_statistics__(simulation, statistics, output_dir)

            result.append(record)

        return result

    @staticmethod
    def __evaluate_criteria_for_statistics__(
        candidate: Candidate, run_id: str, statistics: Statistics, criteria: List[Criterion]
    ) -> Dict:
        record = {
            'candidate': candidate.name,
            'run': run_id,
        }

        record |= {criterion.key: criterion.compute(statistics) for criterion in criteria}

        return record

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

    def __load_and_filter_current_candidates__(self, candidates, criteria, output_dir):
        result_output_file = os.path.join(output_dir, result_filename)

        if not os.path.exists(result_output_file):
            return None

        records = []
        result = []

        for candidate in tqdm(candidates):
            candidate_output_path: str = os.path.join(output_dir, candidates_dir, candidate.name)

            if os.path.exists(candidate_output_path):
                for run in os.listdir(candidate_output_path):
                    run_path = os.path.join(candidate_output_path, run)

                    if os.path.isdir(run_path):
                        statistics = Statistics.load(os.path.join(run_path, statistics_filename))

                        records += [
                            self.__evaluate_criteria_for_statistics__(candidate, run, statistics, criteria)
                        ]
            else:
                result += [candidate]

        return result, records

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
        result_output_file = os.path.join(output_dir, result_filename)
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

        if self.store_run_statistics:
            statistics_output_dir = os.path.join(simulation_output_dir, statistics_filename)
            statistics.save(statistics_output_dir)

        report_output_file = os.path.join(simulation_output_dir, 'report.txt')
        report = statistics.report()

        with open(report_output_file, 'w') as f:
            f.write(str(report))
