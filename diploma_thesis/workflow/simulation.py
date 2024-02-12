import logging

import simpy

from .workflow import Workflow
from typing import Dict

from agents import work_center_from_cli, machine_from_cli
from simulator import from_cli as simulator_from_cli, RunConfiguration, EvaluateConfiguration


class Simulation(Workflow):

    def __init__(self, parameters: Dict):
        super().__init__()

        self.parameters = parameters

    def run(self):
        environment = simpy.Environment()
        logger = self.__make_logger__(name='', environment=environment, log_stdout=True)

        machine = machine_from_cli(parameters=self.parameters['machine_agent']).with_logger(logger)
        work_center = work_center_from_cli(parameters=self.parameters['work_center_agent']).with_logger(logger)

        simulator_logger = logger.getChild('Simulator')

        run_logger = logger.getChild('Run')
        run_logger.setLevel(logging.INFO)

        evaluate_logger = logger.getChild('Evaluate')
        evaluate_logger.setLevel(logging.INFO)

        # TODO: Implement Reward Model
        simulator = simulator_from_cli(
            machine=machine,
            work_center=work_center,
            reward_model=None,
            environment=environment,
            logger=simulator_logger,
            parameters=self.parameters['simulator']
        )

        if run_config := self.parameters.get('run'):
            run_config = RunConfiguration.from_cli(run_logger, run_config)

            simulator.train(run_config)

            for simulation in run_config.simulations:
                if shop_floor := simulation.shop_floor:
                    print(shop_floor.statistics.report())

        if evaluate_config := self.parameters.get('evaluate'):
            evaluate_config = EvaluateConfiguration.from_cli(evaluate_logger, evaluate_config)

            simulator.evaluate(evaluate_config)

            for simulation in evaluate_config.simulations:
                if shop_floor := simulation.shop_floor:
                    print(shop_floor.statistics.report())
