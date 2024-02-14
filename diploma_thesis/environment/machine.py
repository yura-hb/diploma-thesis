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
    # Total processing time for each job in the queue
    total_processing_time: int = 0
    # Expected moment of machine availability, i.e. without breakdowns and maintenances
    available_at: torch.FloatTensor = 0
    # Moment, when the machine will be free working or breakdown
    free_at: torch.FloatTensor = 0
    # The time of machine recover from breakdown
    restart_at: torch.FloatTensor = 0
    # Total runtime
    run_time: int = 0
    # Breakdown duration
    repair_duration: int = 0

    def with_new_job(self, job: environment.Job, now: int):
        """
        Adds new job to the queue

        Args:
            job: Job to schedule into queue on the machine
            now: Timestamp when the job has arrived at the machine
        """
        self.queue += [job]

        return self.with_updated_information(now)

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

    def with_initiated_work(self, processing_time: torch.FloatTensor, now: torch.FloatTensor):
        """
        Initiates the work on the machine

        Args:
            processing_time: The duration of the operation
            now: Current time
        """
        self.free_at = now + processing_time

        return self

    def with_breakdown(self, now: torch.FloatTensor):
        """
        Sets machine into breakdown state

        Args:
            now: Current time
        """
        self.restart_at = now + self.repair_duration
        self.available_at = self.restart_at + self.total_processing_time
        self.free_at = self.restart_at
        self.repair_duration = 0

    def with_runtime(self, operation_processing_time: int):
        self.run_time += operation_processing_time

        return self

    def with_repair_duration(self, duration: float):
        assert duration > 0, "Repair duration must be positive"

        self.repair_duration = duration

        return self

    @property
    def will_breakdown(self):
        return self.repair_duration > 0

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
    breakdown_start_at: torch.FloatTensor = field(default_factory=lambda: torch.FloatTensor([]))
    breakdown_end_at: torch.FloatTensor = field(default_factory=lambda: torch.FloatTensor([]))

    def with_decision_time(self, decision_time: float):
        if not isinstance(decision_time, torch.Tensor):
            decision_time = torch.FloatTensor([decision_time])

        self.decision_times = torch.cat([self.decision_times, torch.atleast_1d(decision_time)])

        return self

    def with_breakdown_start(self, start_at: torch.FloatTensor):
        if not isinstance(start_at, torch.Tensor):
            start_at = torch.FloatTensor([start_at])

        self.breakdown_start_at = torch.cat([self.breakdown_start_at, torch.atleast_1d(start_at)])

        return self

    def with_breakdown_end(self, end_at: torch.FloatTensor):
        if not isinstance(end_at, torch.Tensor):
            end_at = torch.FloatTensor([end_at])

        self.breakdown_end_at = torch.cat([self.breakdown_end_at, torch.atleast_1d(end_at)])

        return self


@dataclass(frozen=True)
class Key:
    work_center_id: int
    machine_id: int


class Machine:

    def __init__(self, environment: simpy.Environment, machine_idx: int, work_center_idx: int):
        self.environment = environment

        self.state = State(machine_idx=machine_idx, work_center_idx=work_center_idx)
        self.history = History()
        self._shop_floor = None
        self._breakdown: 'environment.Breakdown' = None

        # Events
        self.did_dispatch_event = self.environment.event()
        self.is_on_event = self.environment.event()

        # Initially, machine is working
        self.is_on_event.succeed()

    def connect(self, shop_floor: 'environment.ShopFloor'):
        self._shop_floor = shop_floor

    def simulate(self, breakdown: 'environment.Breakdown'):
        self._breakdown = breakdown

        self.environment.process(self.__produce__())
        self.environment.process(self.__breakdown__())

    def receive(self, job: environment.Job):
        job.with_event(self.__new_event__(environment.JobEvent.Kind.arrival_on_machine))

        self.state.with_new_job(job, self.environment.now)
        self.__did_receive_job__()

    def reset(self):
        self.state = State(machine_idx=self.state.machine_idx, work_center_idx=self.state.work_center_idx)
        self.history = History()
        self.did_dispatch_event = self.environment.event()
        self.is_on_event = self.environment.event()
        self.is_on_event.succeed()

    @property
    def shop_floor(self) -> 'environment.ShopFloor':
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
    def key(self) -> Key:
        return Key(work_center_id=self.work_center_idx, machine_id=self.machine_idx)

    @property
    def work_center(self) -> 'environment.WorkCenter':
        return self.shop_floor.work_center(self.work_center_idx)

    @property
    def queue_size(self) -> int:
        return len(self.state.queue)

    @property
    def cumulative_processing_time(self) -> torch.FloatTensor:
        return self.state.total_processing_time

    @property
    def time_till_available(self) -> torch.FloatTensor:
        return max(self.state.available_at - self.environment.now, 0)

    @property
    def cumulative_run_time(self) -> torch.FloatTensor:
        return self.state.run_time

    @property
    def arriving_jobs(self) -> List[environment.Job]:
        """
        Returns: A list of jobs which are about to arrive at the machine, i.e. they were selected for processing on
        the previous machine
        """
        return [
            job for job in self.shop_floor.in_system_jobs
            if job.next_work_center_idx == self.work_center_idx and job.release_moment_on_machine is not None
        ]

    # Timeline

    def __produce__(self):
        if not self.did_dispatch_event.triggered and len(self.state.queue) == 0:
            yield self.environment.process(self.__starve__())

        while True:
            did_breakdown = yield self.environment.process(self.__breakdown_if_needed__())
            did_starve = yield self.environment.process(self.__starve_if_needed__())

            if did_breakdown or did_starve:
                continue

            moment = self.environment.now

            self.history.with_decision_time(moment)

            job = self.__select_job__()

            if isinstance(job, environment.WaitInfo):
                yield self.environment.timeout(job.wait_time)
                continue

            self.__notify_job_about_production__(job, production_start=True)

            self.shop_floor.will_produce(job, self)

            processing_time = job.current_operation_processing_time_on_machine.item()

            self.state.with_initiated_work(processing_time, moment)

            # Perform the operation of the job
            yield self.environment.timeout(processing_time)

            self.__notify_job_about_production__(job, production_start=False)
            self.state.with_runtime(processing_time)
            self.shop_floor.did_produce(job, self)

            self.__forward__(job)

    def __breakdown__(self):
        if self._breakdown is None:
            return

        while True:
            if self.state.will_breakdown:
                duration = max(self.state.free_at - self.environment.now, 10)
                yield self.environment.timeout(duration)
                continue

            breakdown_arrival = self._breakdown.sample_next_breakdown_time(self)

            yield self.environment.timeout(breakdown_arrival)

            repair_duration = self._breakdown.sample_repair_duration(self)

            self.state.with_repair_duration(repair_duration)

    def __starve__(self):
        self.did_dispatch_event = self.environment.event()

        yield self.did_dispatch_event

    def __select_job__(self):
        if self.state.is_empty:
            return environment.WaitInfo(wait_time=1)

        if len(self.state.queue) == 1:
            return self.state.queue[0]

        job = self.shop_floor.schedule(self, self.environment.now)

        return job

    def __breakdown_if_needed__(self):
        """
        Breaks down the machine, if it is not broken down yet
        """
        if self.state.will_breakdown:
            repair_duration = self.state.repair_duration

            self.state.with_breakdown(self.environment.now)
            self.history.with_breakdown_start(self.environment.now)
            self.shop_floor.did_breakdown(self, repair_duration)

            yield self.environment.timeout(repair_duration)

            self.history.with_breakdown_end(self.environment.now)
            self.shop_floor.did_repair(self)

            return True

        return False

    def __starve_if_needed__(self):
        """
        Starves the machine, if there is no job in the queue
        """
        if self.state.is_empty:
            yield self.environment.process(self.__starve__())
            return True

        return False

    def __forward__(self, job: environment.Job):
        """
        Forwards the job to the next machine by sending it to the shop floor

        Args:
            job: Job to forward

        Returns: None
        """
        self.state.without_job(job.id, now=self.environment.now)
        self.shop_floor.forward(job, from_=self)

    # Utility

    def __did_receive_job__(self):
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
