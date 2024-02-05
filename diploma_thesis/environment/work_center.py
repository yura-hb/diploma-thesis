
from dataclasses import dataclass, field
from typing import List

import simpy
import torch

import environment
from model.routing.static.routing_rules import RoutingRule, WorkCenterState


@dataclass
class State:
    idx: int = 0

    queue: List[environment.Job] = field(default_factory=list)

    def with_new_job(self, job: environment.Job):
        self.queue += [job]

        return self

    def with_flushed_queue(self):
        self.queue = []

        return self


@dataclass
class History:
    decision_times: torch.FloatTensor = field(default_factory=lambda: torch.FloatTensor([]))

    def with_decision_time(self, decision_time: float):
        if not isinstance(decision_time, torch.Tensor):
            decision_time = torch.FloatTensor([decision_time])

        self.decision_times = torch.cat([self.decision_times, torch.atleast_1d(decision_time)])

        return self


class Context:
    machines: List['environment.Machine'] = field(default_factory=list)
    work_centers: List['environment.WorkCenter'] = field(default_factory=list)
    shopfloor: 'environment.ShopFloor' = field(default_factory=lambda: None)

    def with_info(self,
                  machines: List['environment.Machine'],
                  work_centers: List['environment.WorkCenter'],
                  shopfloor: 'environment.ShopFloor'):
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

    def connect(self, machines: List['environment.Machine'],
                work_centers: List['environment.WorkCenter'],
                shopfloor: 'environment.ShopFloor'):
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

            self.history.with_decision_time(self.environment.now)

            for job in self.state.queue:
                if len(self.context.machines) == 1:
                    machine = self.context.machines[0]
                    machine.receive(job)
                    continue

                state = WorkCenterState(work_center_idx=self.state.idx, machines=self.context.machines)

                self.context.shopfloor.will_dispatch(job, self)

                machine = self.rule.select_machine(job, state)

                machine.receive(job)

                self.context.shopfloor.did_dispatch(job, self, machine)

            self.state.with_flushed_queue()

            self.context.shopfloor.did_finish_dispatch(self)

            self.on_route = self.environment.event()

    def receive(self, job: environment.Job):
        self.state.with_new_job(job)

        job.with_event(
            environment.JobEvent(
                moment=self.environment.now,
                kind=environment.JobEvent.Kind.arrival_on_work_center,
                work_center_idx=self.state.idx
            )
        )

        self.did_receive_job()

    @property
    def machines(self) -> List['environment.Machine']:
        return self.context.machines

    def did_receive_job(self):
        # Simpy doesn't allow repeated triggering of the same event. Yet, in context of the simulation
        # the model shouldn't care
        try:
            self.on_route.succeed()
        except:
            pass
