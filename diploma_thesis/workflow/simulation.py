import os
from typing import Dict, List

import pandas as pd
import simpy
import torch
import yaml

import simulator
from agents import work_center_from_cli, machine_from_cli
from agents.base.rl_agent import RLAgent
from simulator import from_cli as simulator_from_cli, Simulator, RewardCache
from simulator import run_configuration_from_cli, evaluate_configuration_from_cli
from simulator.tape import TapeModel
from simulator.graph import GraphModel
from utils import save
from .workflow import Workflow


torch.set_num_threads(1)
torch._dynamo.config.suppress_errors = True


class Simulation(Workflow):

    def __init__(self, parameters: Dict):
        super().__init__()

        self.parameters = parameters

    @property
    def workflow_id(self) -> str:
        return self.parameters.get('name', '')

    @property
    def log_stdout(self):
        return self.parameters.get('log_stdout', False)

    @property
    def store_run_statistics(self):
        return self.parameters.get('store_run_statistics', False)

    def run_log_file(self, output_dir):
        if not self.parameters.get('log_run', False):
            return None

        return os.path.join(output_dir, 'log.txt')

    @property
    def is_debug(self):
        return self.parameters.get('debug', False)

    def run(self):
        output_dir = self.__make_output_dir__(self.parameters['name'], self.parameters['output_dir'])
        log_file = os.path.join(output_dir, 'log.txt')

        simulator = self.__make_simulator__()
        logger = self.__make_logger__(name='Simulation', filename=log_file, log_stdout=self.log_stdout)
        simulator.with_logger(logger)

        parameters_file = os.path.join(output_dir, 'parameters.yml')

        with open(parameters_file, 'w') as file:
            yaml.dump(self.parameters, file)

        self.__run__(simulator, output_dir)

        self.__evaluate__(simulator, output_dir)

    def __run__(self, simulator: Simulator, output_dir: str):
        config = self.parameters.get('run')

        if not config:
            return

        environment = simpy.Environment()
        simulation_output_dir = os.path.join(output_dir, 'run')
        logger = self.__make_time_logger__(name='Run',
                                           environment=environment,
                                           filename=self.run_log_file(simulation_output_dir),
                                           log_stdout=self.log_stdout)

        config = run_configuration_from_cli(config, logger=logger)

        if self.is_debug:
            config.timeline.duration = 4096
            config.n_workers = 1
            config.timeline.warm_up_phases = [0]

            simulator.train(environment, config)
        else:
            reward_cache = simulator.train(environment, config)

            self.__store_simulations__(config.simulations,
                                       reward_cache,
                                       self.store_run_statistics,
                                       simulation_output_dir)

        agent_output_dir = os.path.join(output_dir, 'agent')

        try:
            os.makedirs(agent_output_dir)
        except:
            pass

        self.__process_rl_agents__(simulator, agent_output_dir)
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

        if not self.is_debug:
            simulator.evaluate(environment, config)

            self.__store_simulations__(config.simulations,
                                       reward_cache=None,
                                       store_run_statistics=self.store_run_statistics,
                                       output_dir=simulation_output_dir)

    def __make_simulator__(self):
        machine = machine_from_cli(parameters=self.parameters['machine_agent'])
        work_center = work_center_from_cli(parameters=self.parameters['work_center_agent'])
        tape = TapeModel.from_cli(parameters=self.parameters['tape'])
        graph = GraphModel.from_cli(parameters=self.parameters['graph'])
        simulator = simulator_from_cli(
            machine=machine,
            work_center=work_center,
            tape_model=tape,
            graph_model=graph,
            parameters=self.parameters['simulator']
        )

        return simulator

    @staticmethod
    def __store_simulations__(simulations: List[simulator.Simulation],
                              reward_cache: RewardCache,
                              store_run_statistics: bool,
                              output_dir: str):
        reward_cache = RewardCache(batch_size=[]) if reward_cache is None else reward_cache
        machine_reward, work_center_reward = Simulation.__process_reward_cache__(reward_cache)

        for simulation in simulations:
            path = os.path.join(output_dir, simulation.simulation_id)

            if not os.path.exists(path):
                os.makedirs(path)

            if shop_floor := simulation.shop_floor:
                statistics = shop_floor.statistics

                if store_run_statistics:
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

            data = data.to_dict()

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
        machine_path = os.path.join(output_dir, 'machine.pt')
        save(obj=simulator.machine, path=machine_path)

        work_center_path = os.path.join(output_dir, 'work_center.pt')
        save(obj=simulator.work_center, path=work_center_path)

    @staticmethod
    def __process_rl_agents__(simulator: Simulator, output_dir: str):
        models = [simulator.machine, simulator.work_center]
        loss_paths = ['machine_loss.csv', 'work_center_loss.csv']

        for model, loss_path in zip(models, loss_paths):
            if isinstance(model, RLAgent):
                loss = model.loss_record()
                path = os.path.join(output_dir, loss_path)
                loss.to_csv(path, index=True)
                model.clear_memory()

    @staticmethod
    def __store_reward_record__(output_path, df, shop_floor_id, filename):
        path = os.path.join(output_path, filename)
        df[df['shop_floor_id'] == shop_floor_id].to_csv(path, index=False)
