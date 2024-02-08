from dataclasses import dataclass, field
from functools import reduce
from typing import List

import simpy
import torch

import environment


@dataclass
class State:
    """
    Support class representing the state of the machine
    """
    # The index of the machine
    machine_idx: int = 0
    # The index of work_center, where the machine is located
    work_center_idx: int = 0
    # The list of jobs, which are queued on the machine
    queue: List[environment.Job] = field(default_factory=list)
    # The index of current processing job
    current_job_idx: int = 0
    # The moment of decision for the machine
    decision_time: torch.FloatTensor = 0
    # Total processing time for each job in the queue
    total_processing_time: int = 0
    # Expected moment of machine availability, i.e. without breakdowns and maintenances
    available_at: torch.FloatTensor = 0
    # The time of machine recover from breakdown
    restart_at: torch.FloatTensor = 0
    # Total runtime
    run_time: int = 0

    def with_new_job(self, job: environment.Job, now: int):
        """
        Adds new job to the queue

        Args:
            job: Job to schedule into queue on the machine
            now: Timestamp when the job has arrived at the machine
        """
        self.queue += [job]

        return self.with_updated_information(now)

    def with_decision_time(self, decision_time: torch.FloatTensor):
        """
        Sets the decision time for the machine

        Args:
            decision_time: Time of the decision
        """
        self.decision_time = decision_time

        return self

    def with_updated_information(self, now: torch.FloatTensor):
        """
        Updates total processing time of scheduled operations and the theoretical moment of machine availability

        Args:
            now: Current time
        """
        self.total_processing_time = torch.Tensor([
            job.current_operation_processing_time_on_machine for job in self.queue
        ]).sum()
        self.available_at = now + self.total_processing_time

        return self

    def with_breakdown(self, restart_at: torch.FloatTensor):
        """
        Sets machine into breakdown state

        Args:
            restart_at: Moment of machine restart
        """
        self.restart_at = restart_at
        self.available_at = self.restart_at + self.total_processing_time

    def with_runtime(self, operation_processing_time: int):
        self.run_time += operation_processing_time

        return self

    @property
    def is_empty(self):
        """
        Returns: True if the queue is empty, False otherwise
        """
        return len(self.queue) == 0

    def without_job(self, job_id: int, now: torch.FloatTensor):
        """
        Removes job from the queue

        Args:
            job_id: Job to remove from the queue
            now: Current time
        """
        idx = reduce(
            lambda value, pair: pair[0] if pair[1].id == job_id else value,
            enumerate(self.queue),
            None
        )

        if idx is None:
            return

        self.queue.pop(idx)

        return self.with_updated_information(now)


@dataclass
class History:
    decision_times: torch.FloatTensor = field(default_factory=lambda: torch.FloatTensor([]))

    def with_decision_time(self, decision_time: float):
        if not isinstance(decision_time, torch.Tensor):
            decision_time = torch.FloatTensor([decision_time])

        self.decision_times = torch.cat([self.decision_times, torch.atleast_1d(decision_time)])

        return self


class Machine:

    def __init__(self, environment: simpy.Environment, machine_idx: int, work_center_idx: int):
        self.environment = environment

        self.state = State(machine_idx=machine_idx, work_center_idx=work_center_idx)
        self.history = History()
        self._shop_floor = None

        # Events
        self.did_dispatch_event = self.environment.event()
        self.is_on_event = self.environment.event()

        # Initially, machine is working
        self.is_on_event.succeed()

    def connect(self, shop_floor: 'environment.ShopFloor'):
        self._shop_floor = shop_floor
        self.environment.process(self.produce())

    def simulate(self):
        self.environment.process(self.produce())

    def receive(self, job: environment.Job):
        job.with_event(self.__new_event__(environment.JobEvent.Kind.arrival_on_machine))

        self.state.with_new_job(job, self.environment.now)
        self.did_receive_job()

    def produce(self):
        if not self.did_dispatch_event.triggered and len(self.state.queue) == 0:
            yield self.environment.process(self.starve())

        while True:
            moment = self.environment.now

            self.state.with_decision_time(moment)
            self.history.with_decision_time(moment)

            job = self.select_job()

            if isinstance(job, environment.WaitInfo):
                yield self.environment.timeout(job.wait_time)
                self.starve_if_needed()
                continue

            self.__notify_job_about_production__(job, production_start=True)

            self.shop_floor.will_produce(job, self)

            processing_time = job.current_operation_processing_time_on_machine

            # Perform the operation of the job
            yield self.environment.timeout(processing_time)

            self.__notify_job_about_production__(job, production_start=False)

            self.state.with_runtime(processing_time)

            self.forward(job)
            self.breakdown_if_needed()
            self.starve_if_needed()

    def starve(self):
        self.did_dispatch_event = self.environment.event()

        yield self.did_dispatch_event

        self.breakdown_if_needed()

    def select_job(self):
        if self.state.is_empty:
            return environment.WaitInfo(wait_time=1)

        if len(self.state.queue) == 1:
            return self.state.queue[0]

        job = self.shop_floor.schedule(self, self.environment.now)

        return job

    def breakdown(self, restart_at: int):
        start = self.environment.now

        self.state.with_breakdown(restart_at)

        yield self.is_on_event

    def breakdown_if_needed(self):
        """
        Breaks down the machine, if it is not broken down yet
        """
        if not self.is_on_event.triggered:
            # TODO: Verify
            yield self.environment.process(self.breakdown(0))

    def starve_if_needed(self):
        """
        Starves the machine, if there is no job in the queue
        """
        if self.state.is_empty:
            yield self.environment.process(self.starve())

    def forward(self, job: environment.Job):
        """
        Forwards the job to the next machine by sending it to the shop floor

        Args:
            job: Job to forward

        Returns: None
        """
        self.state.without_job(job.id, now=self.environment.now)
        self.shop_floor.forward(job, from_=self)

    @property
    def shop_floor(self):
        return self._shop_floor()

    @property
    def queue(self) -> List[environment.Job]:
        return self.state.queue

    @property
    def work_center_idx(self) -> int:
        return self.state.work_center_idx

    @property
    def machine_idx(self) -> int:
        return self.state.machine_idx

    @property
    def cumulative_processing_time(self) -> int:
        return self.state.total_processing_time

    @property
    def time_till_available(self) -> int:
        return self.state.available_at - self.environment.now

    @property
    def queue_size(self) -> int:
        return len(self.state.queue)

    @property
    def cumulative_run_time(self) -> int:
        return self.state.run_time

    def did_receive_job(self):
        # Simpy doesn't allow repeated triggering of the same event. Yet, in context of the simulation
        # the machine shouldn't care
        try:
            self.did_dispatch_event.succeed()
        except:
            pass

    def __notify_job_about_production__(self, job: environment.Job, production_start: bool):
        kind = environment.JobEvent.Kind
        event = kind.production_start if production_start else kind.production_end
        event = self.__new_event__(event)

        job.with_event(event)

    def __new_event__(self, kind: environment.JobEvent.Kind):
        return environment.JobEvent(
            moment=self.environment.now,
            kind=kind,
            work_center_idx=self.state.work_center_idx,
            machine_idx=self.state.machine_idx
        )
