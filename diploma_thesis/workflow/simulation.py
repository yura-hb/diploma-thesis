import os
import shutil
from typing import Dict, List

import simpy

import environment
import simulator

from agents import work_center_from_cli, machine_from_cli
from simulator import from_cli as simulator_from_cli, RunConfiguration, EvaluateConfiguration, Simulator
from tape import TapeModel
from .workflow import Workflow


class Simulation(Workflow):

    def __init__(self, parameters: Dict):
        super().__init__()

        self.parameters = parameters

    def run(self):
        simulator = self.__make_simulator__()
        logger = self.__make_logger__(name='Simulation', log_stdout=True)
        simulator.with_logger(logger)

        output_dir = self.__make_output_dir__()

        self.__run__(simulator, output_dir)
        self.__evaluate__(simulator, output_dir)

    def __run__(self, simulator: Simulator, output_dir: str):
        config = self.parameters.get('run')

        if not config:
            return

        environment = simpy.Environment()
        logger = self.__make_time_logger__(name='Run', environment=environment, log_stdout=True)

        config = RunConfiguration.from_cli(logger, config)

        simulator.train(environment, config)

        simulation_output_dir = os.path.join(output_dir, 'run')
        self.__store_simulations__(config.simulations, simulation_output_dir)

        agent_output_dir = os.path.join(simulation_output_dir, 'agents')
        self.__store_agents__(agent_output_dir)

    def __evaluate__(self, simulator: Simulator, output_dir: str):
        config = self.parameters.get('evaluate')

        if not config:
            return

        environment = simpy.Environment()
        logger = self.__make_time_logger__(name='Evaluate', environment=environment, log_stdout=True)

        config = EvaluateConfiguration.from_cli(logger, config)

        simulator.evaluate(environment, config)

        simulation_output_dir = os.path.join(output_dir, 'evaluate')
        self.__store_simulations__(config.simulations, simulation_output_dir)

    def __make_simulator__(self):
        machine = machine_from_cli(parameters=self.parameters['machine_agent'])
        work_center = work_center_from_cli(parameters=self.parameters['work_center_agent'])
        tape = TapeModel.from_cli(parameters=self.parameters['tape'])

        simulator = simulator_from_cli(
            machine=machine, work_center=work_center, tape=tape, parameters=self.parameters['simulator']
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

    def __store_simulations__(self, simulations: List[simulator.Simulation], output_dir: str):
        for simulation in simulations:
            path = os.path.join(output_dir, simulation.simulation_id)

            if shop_floor := simulation.shop_floor:
                statistics = shop_floor.statistics
                statistics.save(path)

                with open(os.path.join(path, 'report.txt'), 'w') as file:
                    file.write(str(statistics.report()))

    def __store_agents__(self, output_dir: str):
        pass
