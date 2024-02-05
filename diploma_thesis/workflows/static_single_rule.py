from typing import Dict

import simpy

from environment.configuration import Configuration
from environment.shop_floor import ShopFloor
from job_samplers import from_cli_arguments as job_sampler_from_cli_arguments
from agent.dispatch.routing import from_cli_arguments as routing_model_from_cli_arguments
from agent.dispatch.scheduling import from_cli_arguments as scheduling_model_from_cli_arguments
from .workflow import Workflow


class SingleModel(Workflow):

    def __init__(self, parameters: Dict):
        super().__init__()

        self.parameters = parameters

    def run(self):
        environment = simpy.Environment()

        problem = Configuration.from_cli_arguments(self.parameters['problem'])
        sampler = job_sampler_from_cli_arguments(problem, environment, self.parameters['sampler'])
        scheduling_model = scheduling_model_from_cli_arguments(self.parameters['scheduling_model'])
        routing_model = routing_model_from_cli_arguments(self.parameters['routing_model'])

        configuration = ShopFloor.Configuration(
            environment=environment,
            problem=problem,
            sampler=sampler,
            scheduling_rule=scheduling_model,
            routing_rule=routing_model
        )

        shopfloor = ShopFloor(
            configuration,
            logger=self.__make_logger__('ShopFloor', log_stdout=True),
            delegate=self.make_delegate([scheduling_model, routing_model])
        )

        shopfloor.simulate()

        environment.run(until=problem.timespan)

        statistics = shopfloor.statistics

        predicate = statistics.Predicate
        time_predicate = predicate.TimePredicate(at=problem.timespan, kind=predicate.TimePredicate.Kind.less_than)

        report = statistics.report(time_predicate)

        print(report)
