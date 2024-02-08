
from dataclasses import dataclass, field
from typing import List

import simpy
import torch

import environment


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


class WorkCenter:

    def __init__(self, environment: simpy.Environment, work_center_idx: int):
        self.environment = environment

        self.state = State(idx=work_center_idx)
        self.history = History()
        self._machines = []
        self._shop_floor = None

        self.on_route = environment.event()

    def connect(self, shop_floor: 'environment.ShopFloor', machines: List['environment.Machine']):
        self._shop_floor = shop_floor
        self._machines = machines

    def simulate(self):
        for machine in self.machines:
            machine.simulate()

        self.environment.process(self.dispatch())

    def dispatch(self):
        assert self.shop_floor is not None, "Work center is not connected to the shop floor"

        while True:
            yield self.on_route

            self.history.with_decision_time(self.environment.now)

            for job in self.state.queue:
                if len(self.machines) == 1:
                    machine = self.machines[0]
                    machine.receive(job)
                    continue

                self.shop_floor.will_dispatch(job, self)

                # TODO: React on None
                machine = self.shop_floor.route(job, work_center_idx=self.state.idx, machines=self.machines)
                machine.receive(job)

                self.shop_floor.did_dispatch(job, self, machine)

            self.state.with_flushed_queue()

            self.shop_floor.did_finish_dispatch(self)

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
    def shop_floor(self):
        return self._shop_floor()

    @property
    def machines(self) -> List['environment.Machine']:
        return self._machines

    def did_receive_job(self):
        # Simpy doesn't allow repeated triggering of the same event. Yet, in context of the simulation
        # the machine shouldn't care
        try:
            self.on_route.succeed()
        except:
            pass
