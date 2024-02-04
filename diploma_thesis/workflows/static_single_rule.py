import simpy

from environment.problem import Problem
from environment.shop_floor import ShopFloor
from environment.job_samplers import UniformJobSampler
from .workflow import Workflow


class StaticSingleRule(Workflow):

    def __init__(self, problem: Problem, sampler: UniformJobSampler):
        super().__init__()

        self.problem = problem
        self.sampler = sampler

    def run(self):
        environment = simpy.Environment()

        configuration = ShopFloor.Configuration(
            environment=environment,
            problem=self.problem,
            sampler=self.sampler
        )

        shopfloor = ShopFloor(
            configuration,
            logger=self.__make_logger__('ShopFloor', log_stdout=True)
        )

        shopfloor.simulate()

        environment.run(until=self.problem.timespan)

        statistics = shopfloor.statistics

        predicate = statistics.Predicate
        time_predicate = predicate.TimePredicate(at=self.problem.timespan, kind=predicate.TimePredicate.Kind.less_than)

        report = statistics.report(time_predicate)

        print(report)
