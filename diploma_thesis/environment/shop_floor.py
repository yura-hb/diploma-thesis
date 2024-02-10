import logging
from dataclasses import dataclass, field
from typing import List, Dict

import simpy
import torch

import environment
from .utils import ShopFloorFactory


@dataclass
class State:
    idx: str = 0

    job_id: int = 0

    number_of_jobs_in_system: int = 0

    @property
    def dispatched_job_count(self):
        return self.job_id

    def with_machines_count(self):
        return self

    def with_new_job_id(self):
        self.job_id += 1

        return self

    def with_new_job_in_system(self):
        self.number_of_jobs_in_system += 1

        return self

    def with_job_completed(self):
        self.number_of_jobs_in_system -= 1

        return self


@dataclass
class History:
    @dataclass
    class Record:
        job_id: int
        created_at: torch.FloatTensor
        duration: int
        work_center_idx: int
        machine_idx: int

    # A list of jobs, where each job is represented by the id of machine
    jobs: Dict[int, environment.Job] = field(default_factory=dict)

    started_at: int = 0

    def with_machines_count(self):
        return self

    def with_new_job(self, job: environment.Job):
        self.jobs[job.id] = job

        return self

    def with_started_at(self, now: int):
        self.started_at = now

        return self

    def __make_record__(self, job: environment.Job, machine: environment.Machine, now: int):
        return self.Record(
            job_id=job.id,
            created_at=now,
            duration=job.current_operation_processing_time_on_machine,
            work_center_idx=machine.state.work_center_idx,
            machine_idx=machine.state.machine_idx
        )

    def job(self, job_id: int):
        return self.jobs[job_id]


class ShopFloor:
    @dataclass
    class Configuration:
        problem: environment.Configuration
        sampler: 'environment.JobSampler'
        breakdown: 'environment.Breakdown'
        agent: 'environment.Agent'
        delegate: 'environment.Delegate'
        environment: simpy.Environment = field(default_factory=simpy.Environment)

    def __init__(self, idx: str, configuration: Configuration, logger: logging.Logger):
        self.id = id
        self.configuration = configuration
        self.logger = logger
        self.agent = configuration.agent
        self.delegate = configuration.delegate

        self._work_centers: List[environment.WorkCenter] = []
        self._machines: List[environment.Machine] = []

        self.state = State(idx=idx)
        self.history = History()
        self._work_centers, self._machines = ShopFloorFactory(self.configuration, self).make()

        self.did_finish_simulation_event = configuration.environment.event()

    def simulate(self):
        torch.manual_seed(self.configuration.problem.seed)

        self.history.with_started_at(self.configuration.environment.now)
        self.delegate.did_start_simulation(self.state.idx)

        if self.configuration.problem.pre_assign_initial_jobs:
            self.__assign_initial_jobs__()

        for work_center in self.work_centers:
            work_center.simulate(self.configuration.breakdown)

        self.configuration.environment.process(self.__dispatch_jobs__())

        return self.did_finish_simulation_event

    def reset(self):
        self.state = State(idx=self.state.idx)
        self.history = History()
        self.did_finish_simulation_event = self.configuration.environment.event()

        for work_center in self.work_centers:
            work_center.reset()

    # Utility properties

    @property
    def statistics(self) -> 'environment.Statistics':
        from .statistics import Statistics

        return Statistics(self)

    @property
    def work_centers(self) -> List['environment.WorkCenter']:
        return self._work_centers

    @property
    def machines(self) -> List['environment.Machine']:
        return self._machines

    # Navigation

    def forward(self, job: environment.Job, from_: environment.Machine):
        next_work_center_idx = job.next_work_center_idx

        if next_work_center_idx is not None:
            job.with_event(
                environment.Job.Event(
                    moment=self.configuration.environment.now,
                    kind=environment.Job.Event.Kind.forward
                )
            )

            self.work_centers[next_work_center_idx].receive(job)
        else:
            job.with_event(
                environment.Job.Event(
                    moment=self.configuration.environment.now,
                    kind=environment.Job.Event.Kind.completion
                )
            )

            self.state.with_job_completed()
            self.did_complete(job)

        self.logger.info(
            f"Job {job.id} {job.current_step_idx} has been {'completed' if job.is_completed else 'produced'} "
            f"on machine {from_.state.machine_idx} in work-center {from_.state.work_center_idx}. "
            f"Jobs in system {self.state.number_of_jobs_in_system}"
        )

    # Decision methods

    def schedule(self, machine: environment.Machine, now: int) -> 'environment.Job | environment.WaitInfo':
        return self.agent.schedule(self.state.idx, machine, now)

    def route(
        self, job: environment.Job, work_center_idx: int, machines: List['environment.Machine']
    ) -> 'environment.Machine | None':
        return self.agent.route(self.state.idx, job, work_center_idx, machines)

    # Events from subcomponents (WorkCenter, Machine)

    # TODO: Rewrite with meta programming (Low Priority)

    def will_produce(self, job: environment.Job, machine: environment.Machine):
        self.delegate.will_produce(self.state.idx, job, machine)

    def did_produce(self, job: environment.Job, machine: environment.Machine):
        self.delegate.did_produce(self.state.idx, job, machine)

    def will_dispatch(self, job: environment.Job, work_center: environment.WorkCenter):
        self.delegate.will_dispatch(self.state.idx, job, work_center)

    def did_dispatch(self, job: environment.Job, work_center: environment.WorkCenter, machine: environment.Machine):
        self.delegate.did_dispatch(self.state.idx, job, work_center, machine)

    def did_finish_dispatch(self, work_center: environment.WorkCenter):
        self.delegate.did_finish_dispatch(self.state.idx, work_center)

    def did_complete(self, job: environment.Job):
        self.delegate.did_complete(self.state.idx, job)
        self.__test_if_finished__()

    def did_breakdown(self, machine: environment.Machine, repair_time: float):
        self.logger.info(
            f'Machine {machine.state.machine_idx} in work-center {machine.state.work_center_idx} '
            f'has broken down. Repair time {repair_time}'
        )

        self.delegate.did_breakdown(self.state.idx, machine, repair_time)

    def did_repair(self, machine: environment.Machine):
        self.logger.info(
            f'Machine {machine.state.machine_idx} in work-center {machine.state.work_center_idx} '
            'has been repaired'
        )

        self.delegate.did_repair(self.state.idx, machine)

    # Utility methods

    def __assign_initial_jobs__(self):
        for work_center in self.work_centers:
            for _ in work_center.machines:
                job = self.__sample_job__(
                    initial_work_center_idx=work_center.state.idx,
                    created_at=self.history.started_at
                )

                self.history.with_new_job(job)

                self.__dispatch__(job, work_center)

            work_center.did_receive_job()

    def __dispatch_jobs__(self):
        while self.__should_dispatch__():
            arrival_time = self.configuration.sampler.sample_next_arrival_time()

            yield self.configuration.environment.timeout(arrival_time)

            job = self.__sample_job__(created_at=self.configuration.environment.now)

            self.history.with_new_job(job)

            work_center = self.work_centers[job.step_idx[0]]

            self.__dispatch__(job, work_center)

    def __sample_job__(self, initial_work_center_idx: int = None, created_at: torch.FloatTensor = 0):
        job = (
            self.configuration.sampler.sample(
                job_id=self.state.with_new_job_id().job_id,
                initial_work_center_idx=initial_work_center_idx,
                moment=created_at)
            .with_event(
                environment.Job.Event(
                    kind=environment.Job.Event.Kind.dispatch,
                    moment=created_at,
                )
            )
        )

        return job

    def __test_if_finished__(self):
        has_dispatched_all_jobs = not self.__should_dispatch__()
        is_no_jobs_in_system = self.state.number_of_jobs_in_system == 0

        if not self.did_finish_simulation_event.triggered and has_dispatched_all_jobs and is_no_jobs_in_system:
            self.delegate.did_finish_simulation(self.state.idx)

            self.did_finish_simulation_event.succeed()

    def __should_dispatch__(self):
        if number_of_jobs := self.configuration.sampler.number_of_jobs():
            return self.state.dispatched_job_count < number_of_jobs

        timespan = self.configuration.environment.now - self.history.started_at

        return timespan < self.configuration.problem.timespan

    def __dispatch__(self, job: environment.Job, work_center: environment.WorkCenter):
        self.state.with_new_job_in_system()

        work_center.receive(job)
