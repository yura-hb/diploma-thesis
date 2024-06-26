from .template import Template, Candidate

from typing import List

from agents.machine.model import SchedulingRule
from agents.machine.model.rule import ALL_SCHEDULING_RULES
from agents.workcenter.model import RoutingRule
from agents.workcenter.model.rule import ALL_ROUTING_RULES
from .template import Template, Candidate


class StaticCandidates(Template):

    @classmethod
    def from_cli(cls, parameters: dict):
        scheduling_rules = parameters['scheduling']
        routing_rules = parameters['routing']

        scheduling_rules = StaticCandidates.__make_scheduling_rules__(scheduling_rules)
        routing_rules = StaticCandidates.__make_routing_rules__(routing_rules)

        return [
            Candidate(
                name=cls.__id__(scheduling_rule, routing_rule),
                kind='static',
                parameters=dict(scheduling_rule=scheduling_rule, routing_rule=routing_rule)
            )
            for scheduling_rule in scheduling_rules
            for routing_rule in routing_rules
        ]

    @classmethod
    def __make_scheduling_rules__(cls, rules: str | List[str]) -> List[SchedulingRule]:
        if rules == "all":
            return [rule() for rule in ALL_SCHEDULING_RULES.values()]

        return [ALL_SCHEDULING_RULES[rule]() for rule in rules]

    @classmethod
    def __make_routing_rules__(cls, routing: str | List[str]) -> dict:
        if routing == "all":
            return [rule() for rule in ALL_ROUTING_RULES.values()]

        return [ALL_ROUTING_RULES[rule]() for rule in routing]

    @classmethod
    def __id__(cls, scheduling_rule: SchedulingRule, routing_rule: RoutingRule):
        return f"{scheduling_rule.__class__.__name__}_{routing_rule.__class__.__name__}"
