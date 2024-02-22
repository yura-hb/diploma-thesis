import os
import shutil
from typing import Dict, List

import simpy
import pandas as pd
import torch

import simulator
from agents import work_center_from_cli, machine_from_cli
from simulator import from_cli as simulator_from_cli, Simulator, RewardCache
from simulator import run_configuration_from_cli, evaluate_configuration_from_cli
from tape import TapeModel
from utils import save
from .workflow import Workflow


class Simulation(Workflow):

    def __init__(self, parameters: Dict):
        super().__init__()

        self.parameters = parameters

    @property
    def log_stdout(self):
        return self.parameters.get('log_stdout', False)

    def run(self):
        output_dir = self.__make_output_dir__()
        log_file = os.path.join(output_dir, 'log.txt')

        simulator = self.__make_simulator__()
        logger = self.__make_logger__(name='Simulation', filename=log_file, log_stdout=self.log_stdout)
        simulator.with_logger(logger)

        self.__run__(simulator, output_dir)
        self.__evaluate__(simulator, output_dir)

    def __run__(self, simulator: Simulator, output_dir: str):
        config = self.parameters.get('run')

        if not config:
            return

        environment = simpy.Environment()
        simulation_output_dir = os.path.join(output_dir, 'run')
        simulation_log_file = os.path.join(simulation_output_dir, 'log.txt')

        logger = self.__make_time_logger__(name='Run',
                                           environment=environment,
                                           filename=simulation_log_file,
                                           log_stdout=self.log_stdout)

        config = run_configuration_from_cli(config, logger=logger)
        reward_cache = simulator.train(environment, config)

        self.__store_simulations__(config.simulations, reward_cache, simulation_output_dir)

        agent_output_dir = os.path.join(output_dir, 'agent')
        self.__store_agents__(simulator, agent_output_dir)

    def __evaluate__(self, simulator: Simulator, output_dir: str):
        config = self.parameters.get('evaluate')

        if not config:
            return

        environment = simpy.Environment()
        simulation_output_dir = os.path.join(output_dir, 'evaluate')
        simulation_log_file = os.path.join(simulation_output_dir, 'log.txt')
        logger = self.__make_time_logger__(
            name='Evaluate',
            environment=environment,
            filename=simulation_log_file,
            log_stdout=self.log_stdout
        )

        config = evaluate_configuration_from_cli(config, logger=logger)

        simulator.evaluate(environment, config)

        self.__store_simulations__(config.simulations, reward_cache=None, output_dir=simulation_output_dir)

    def __make_simulator__(self):
        machine = machine_from_cli(parameters=self.parameters['machine_agent'])
        work_center = work_center_from_cli(parameters=self.parameters['work_center_agent'])
        tape = TapeModel.from_cli(parameters=self.parameters['tape'])
        simulator = simulator_from_cli(
            machine=machine, work_center=work_center, tape_model=tape, parameters=self.parameters['simulator']
        )

        return simulator

    def __make_output_dir__(self):
        name = self.parameters['name']
        output_dir = self.parameters.get('output_dir')

        output_path = os.path.join(output_dir, name)

        if os.path.exists(output_path):
            shutil.rmtree(output_path)

        os.makedirs(output_path)

        return output_path

    @staticmethod
    def __store_simulations__(simulations: List[simulator.Simulation], reward_cache: RewardCache, output_dir: str):
        reward_cache = RewardCache(batch_size=[]) if reward_cache is None else reward_cache
        machine_reward, work_center_reward = Simulation.__process_reward_cache__(reward_cache)

        for simulation in simulations:
            path = os.path.join(output_dir, simulation.simulation_id)

            if shop_floor := simulation.shop_floor:
                statistics = shop_floor.statistics
                statistics.save(path)

                sh_id = shop_floor.id.item()

                Simulation.__store_reward_record__(path, machine_reward, sh_id, 'machine_reward.csv')
                Simulation.__store_reward_record__(path, work_center_reward, sh_id, 'work_center_reward.csv')

                with open(os.path.join(path, 'report.txt'), 'w') as file:
                    file.write(str(statistics.report()))

    @staticmethod
    def __process_reward_cache__(reward_cache: RewardCache):
        def __to_dataframe__(data):
            if data.batch_size == torch.Size([]):
                return pd.DataFrame(columns=['shop_floor_id'])

            return pd.DataFrame(data)

        machine_reward = __to_dataframe__(reward_cache.machines)
        work_center_reward = __to_dataframe__(reward_cache.work_centers)

        for df in [machine_reward, work_center_reward]:
            for column in df.columns:
                if column in ['moment', 'reward']:
                    df[column] = df[column].astype(float)
                else:
                    df[column] = df[column].astype(int)

        return machine_reward, work_center_reward

    @staticmethod
    def __store_agents__(simulator: Simulator, output_dir: str):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        machine_path = os.path.join(output_dir, 'machine.pt')
        save(obj=simulator.machine, path=machine_path)

        work_center_path = os.path.join(output_dir, 'work_center.pt')
        save(obj=simulator.work_center, path=work_center_path)

    @staticmethod
    def __store_reward_record__(output_path, df, shop_floor_id, filename):
        path = os.path.join(output_path, filename)
        df[df['shop_floor_id'] == shop_floor_id].to_csv(path, index=False)
