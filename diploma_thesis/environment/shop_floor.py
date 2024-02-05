import logging
from dataclasses import dataclass, field
from typing import List, Dict

import simpy
import torch

import environment
from .utils import ShopFloorFactory


@dataclass
class State:
    idx: int = 0

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

    def with_machines_count(self):
        return self

    def with_new_job(self, job: environment.Job):
        self.jobs[job.id] = job

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
        agent: 'environment.Agent'
        delegate: 'environment.Delegate'
        environment: simpy.Environment = field(default_factory=simpy.Environment)

    def __init__(self, idx: int, configuration: Configuration, logger: logging.Logger):
        self.id = id
        self.configuration = configuration
        self.logger = logger
        self.agent = configuration.agent
        self.delegate = configuration.delegate

        self._work_centers: List[environment.WorkCenter] = []
        self._machines: List[environment.Machine] = []

        self.state = State(idx=idx)
        self.history = History()
        self._work_centers, self._machines = ShopFloorFactory(self.configuration).make()

    def simulate(self):
        if self.configuration.problem.pre_assign_initial_jobs:
            self.__assign_initial_jobs__()

        for work_center in self.work_centers:
            work_center.simulate()

        self.configuration.environment.process(self.__dispatch_jobs__())

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

    def breakdown(self, work_center_idx: int, machine_idx: int, duration: int):
        ...

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
            f"Job {job.id} { job.current_step_idx } has been { 'completed' if job.is_completed else 'produced' } "
            f"on machine {from_.state.machine_idx} in work-center { from_.state.work_center_idx } "
            f"at {self.configuration.environment.now}. Jobs in the system { self.state.number_of_jobs_in_system }"
        )

    def schedule(self, machine: environment.Machine, now: int) -> environment.Job | environment.WaitInfo:
        return self.agent.schedule(self.state.idx, machine, now)

    def route(
        self, job: environment.Job, work_center_idx: int, machines: List['environment.Machine']
    ) -> 'environment.Machine | None':
        return self.agent.route(self.state.idx, job, work_center_idx, machines)

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

    def __assign_initial_jobs__(self):
        for work_center in self.work_centers:
            for _ in work_center.machines:
                job = self.__sample_job__(
                    initial_work_center_idx=work_center.state.idx,
                    created_at=self.configuration.environment.now
                )

                self.history.with_new_job(job)

                self.__dispatch__(job, work_center)

            work_center.did_receive_job()

    def __dispatch_jobs__(self):
        while self.state.dispatched_job_count < self.configuration.sampler.number_of_jobs():
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

    def __dispatch__(self, job: environment.Job, work_center: environment.WorkCenter):
        self.state.with_new_job_in_system()

        work_center.receive(job)

