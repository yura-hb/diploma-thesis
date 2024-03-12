
from dataclasses import dataclass, field
from typing import List

import simpy
import torch

import environment


@dataclass
class State:
    idx: torch.LongTensor = torch.LongTensor([0])

    queue: List[environment.Job] = field(default_factory=list)

    def __post_init__(self):
        self.idx = torch.tensor(self.idx)

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


@dataclass(frozen=True)
class Key:
    work_center_id: int


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

        self.environment.process(self.__dispatch__())

    def reset(self):
        self.state = State(idx=self.state.idx)
        self.history = History()
        self.on_route = self.environment.event()

        for machine in self.machines:
            machine.reset()

    def receive(self, job: environment.Job):
        self.state.with_new_job(job)

        job.with_event(
            environment.JobEvent(
                moment=self.environment.now,
                kind=environment.JobEvent.Kind.arrival_on_work_center,
                work_center_idx=self.state.idx
            )
        )

        self.__did_receive_job__()

    @property
    def shop_floor(self) -> 'environment.ShopFloor':
        return self._shop_floor()

    @property
    def work_center_idx(self) -> torch.LongTensor:
        return self.state.idx

    @property
    def machines(self) -> List['environment.Machine']:
        return self._machines

    @property
    def key(self) -> Key:
        return Key(work_center_id=self.work_center_idx.item())

    @property
    def work_load(self) -> torch.FloatTensor:
        processing_times = [machine.cumulative_processing_time for machine in self.machines]
        processing_times = torch.FloatTensor(processing_times)

        return processing_times.mean()

    @property
    def average_waiting_time(self) -> torch.FloatTensor:
        waiting_times = torch.FloatTensor([machine.time_till_available for machine in self.machines])

        return waiting_times.mean()

    # Timeline

    def __dispatch__(self):
        assert self.shop_floor is not None, "Work center is not connected to the shop floor"

        while True:
            self.history.with_decision_time(self.environment.now)

            non_dispatched_jobs = []

            for job in self.state.queue:
                self.shop_floor.will_dispatch(job, self)

                machine = self.shop_floor.route(work_center=self, job=job)

                if machine is None:
                    non_dispatched_jobs += [job]
                    continue

                machine.receive(job)

                self.shop_floor.did_dispatch(job, self, machine)

            self.state.with_flushed_queue()
            self.state.queue = non_dispatched_jobs
            self.shop_floor.did_finish_dispatch(self)

            yield self.environment.process(self.__starve__())

    def __starve__(self):
        self.on_route = self.environment.event()

        yield self.on_route

    # Utility

    def __did_receive_job__(self):
        # Simpy doesn't allow repeated triggering of the same event. Yet, in context of the run
        # the machine shouldn't care
        try:
            self.on_route.succeed()
        except:
            pass
