from .routing_rule import *


class RandomRoutingRule(RoutingRule):
    """
    Selects a machine at random
    """

    def __call__(self, job: Job, work_center_idx: int, machines: List['Machine']) -> 'Machine | None':
        idx = torch.randint(0, len(machines), (1,))

        return machines[idx]
