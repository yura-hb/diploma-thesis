
from .workflow import Workflow
from typing import Dict

from agents import work_center_from_cli, machine_from_cli
from simulator import from_cli as simulator_from_cli, RunConfiguration, EvaluateConfiguration


class Simulation(Workflow):

    def __init__(self, parameters: Dict):
        super().__init__()

        self.parameters = parameters

    def run(self):
        machine = machine_from_cli(parameters=self.parameters['machine_agent'])
        work_center = work_center_from_cli(parameters=self.parameters['work_center_agent'])
        # TODO: Implement Reward Model
        simulator = simulator_from_cli(
            machine=machine,
            work_center=work_center,
            reward_model=None,
            logger=self.__make_logger__('Simulator', log_stdout=True),
            parameters=self.parameters['simulator']
        )

        if run_config := self.parameters.get('run'):
            run_config = RunConfiguration.from_cli(self.__make_logger__('Run', log_stdout=False), run_config)

            simulator.train(run_config)

        if evaluate_config := self.parameters.get('evaluate'):
            evaluate_config = EvaluateConfiguration.from_cli(self.__make_logger__('Evaluate'), evaluate_config)

            simulator.evaluate(evaluate_config)
