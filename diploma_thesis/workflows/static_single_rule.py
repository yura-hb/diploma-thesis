import simpy

from environment.problem import Problem
from environment.shopfloor import ShopFloor
from .workflow import Workflow


class StaticSingleRule(Workflow):

    def __init__(self, problem: Problem):
        super().__init__()

        self.problem = problem

    def run(self):
        environment = simpy.Environment()

        configuration = ShopFloor.Configuration(
            environment=environment,
            problem=self.problem,
        )

        shopfloor = ShopFloor(
            configuration,
            logger=self.__make_logger__('ShopFloor', log_stdout=True),
        )

        shopfloor.simulate()

        environment.run(until=self.problem.timespan)

