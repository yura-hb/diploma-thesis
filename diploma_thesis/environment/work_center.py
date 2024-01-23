
from dataclasses import dataclass, field
from typing import List
import simpy

from environment.job import Job

from routing_rules import RoutingRule, WorkCenterState


@dataclass
class State:
    idx: int = 0

    queue: List[Job] = field(default_factory=list)

    def with_new_job(self, job: Job):
        self.queue += [job]

        return self

    def with_flushed_queue(self):
        self.queue = []

        return self


class History:
    ...


class Context:
    machines: List['Machine'] = field(default_factory=list)
    work_centers: List['WorkCenter'] = field(default_factory=list)
    shopfloor: 'ShopFloor' = field(default_factory=lambda: None)

    def with_info(self, machines: List['Machine'], work_centers: List['WorkCenter'], shopfloor: 'ShopFloor'):
        self.machines = machines
        self.work_centers = work_centers
        self.shopfloor = shopfloor

        return self


class WorkCenter:

    def __init__(self, environment: simpy.Environment, work_center_idx: int, rule: 'RoutingRule'):
        self.environment = environment
        self.rule = rule

        self.state = State(idx=work_center_idx)
        self.history = History()
        self.context = Context()

        self.on_route = environment.event()

    def connect(self, machines: List['Machine'], work_centers: List['WorkCenter'], shopfloor: 'ShopFloor'):
        """
        Args:
            machines: The list of machines in the work center!!!
            work_centers: The list of work centers in the shop floor
            shopfloor: reference to the shop floor
        """
        self.context.with_info(machines, work_centers, shopfloor)

        self.environment.process(self.dispatch())

    def dispatch(self):
        assert self.context.shopfloor is not None, "Work center is not connected to the shop floor"

        while True:
            yield self.on_route

            for job in self.state.queue:
                if len(self.context.machines) == 1:
                    machine = self.context.machines[0]
                    machine.receive(job)
                    continue

                state = WorkCenterState(work_center_idx=self.state.idx, machines=self.context.machines)

                machine = self.rule.select_machine(job, state)

                machine.receive(job)

            self.state.with_flushed_queue()

            self.on_route = self.environment.event()

    def receive(self, job: Job):
        self.state.with_new_job(job)

        self.on_route.succeed()
