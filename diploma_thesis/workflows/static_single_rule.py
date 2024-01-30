import simpy

from environment.problem import Problem
from environment.shop_floor import ShopFloor
from .workflow import Workflow
from tabulate import tabulate


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

        statistics = shopfloor.statistics

        predicate = statistics.Predicate
        time_predicate = predicate.TimePredicate(at=1000, kind=predicate.TimePredicate.Kind.less_than)

        report = statistics.report(time_predicate)

        print(report)
