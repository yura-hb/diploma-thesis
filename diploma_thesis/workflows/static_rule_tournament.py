import simpy
import torch

from environment.problem import Problem
from environment.shop_floor import ShopFloor
from routing_rules import ALL_ROUTING_RULES, RandomRoutingRule
from scheduling_rules import ALL_SCHEDULING_RULES, RandomSchedulingRule
from .workflow import Workflow


class StaticRuleTournament(Workflow):

    def __init__(self, problem: Problem):
        super().__init__()

        self.problem = problem

    def run(self):
        routing_rules = ALL_ROUTING_RULES
        scheduling_rules = ALL_SCHEDULING_RULES

        routing_rule_parameters_db = dict({
            RandomRoutingRule: {'generator': self.__make_generator__(self.problem)}
        })
        scheduling_rule_parameters_db = dict({
            RandomSchedulingRule: {'generator': self.__make_generator__(self.problem)}
        })

        # For single machine problem routing rule doesn't have any effect on scheduling
        if self.problem.machines_per_workcenter <= 1:
            routing_rules = routing_rules[:1]

        logger = self.__make_logger__(log_stdout=True)

        for routing_rule in routing_rules:
            for scheduling_rule in scheduling_rules:
                logger.info(f'Start evaluating routing rule { routing_rule } and scheduling rule { scheduling_rule }')

                environment = simpy.Environment()

                routing_rule_parameters = routing_rule_parameters_db.get(routing_rule, {})
                scheduling_rule_parameters = scheduling_rule_parameters_db.get(scheduling_rule, {})

                configuration = ShopFloor.Configuration(
                    environment=environment,
                    problem=self.problem,
                    routing_rule=routing_rule(**routing_rule_parameters),
                    scheduling_rule=scheduling_rule(**scheduling_rule_parameters),
                )

                shopfloor = ShopFloor(
                    configuration=configuration,
                    logger=self.__make_logger__('ShopFloor', log_stdout=False),
                )

                shopfloor.simulate()

                environment.run(until=self.problem.timespan)

                logger.info(f'Finish evaluating routing rule {routing_rule} and scheduling rule {scheduling_rule}')
