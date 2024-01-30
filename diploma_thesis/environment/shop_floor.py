import logging
from dataclasses import dataclass, field
from functools import reduce
from typing import List, Tuple, Dict

import simpy
import torch

import environment
from routing_rules import RoutingRule, EARoutingRule
from scheduling_rules import SchedulingRule, FIFOSchedulingRule


@dataclass
class State:
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
        problem: environment.Problem
        sampler: 'environment.job_samplers.JobSampler'
        environment: simpy.Environment = field(default_factory=simpy.Environment)
        scheduling_rule: SchedulingRule = field(default_factory=lambda: FIFOSchedulingRule())
        routing_rule: RoutingRule = field(default_factory=lambda: EARoutingRule())

    def __init__(
        self,
        configuration: Configuration,
        logger: logging.Logger
    ):
        self.configuration = configuration
        self.logger = logger

        self._work_centers: List[environment.WorkCenter] = []
        self._machines: List[environment.Machine] = []

        self.state = State()
        self.history = History()
        
        self.__build__()

    def simulate(self):
        self.assign_initial_jobs()

        self.configuration.environment.process(self.dispatch_jobs())

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

    # Redefine generation of Work-Center paths

    def assign_initial_jobs(self):
        for work_center in self.work_centers:
            for _ in work_center.context.machines:
                job = self.__sample_job__(created_at=0)
                # Hold reference to the job
                self.history.with_new_job(job)
                # Send job to work-center
                self.__dispatch__(job, work_center)

            work_center.did_receive_job()

    def dispatch_jobs(self):
        while self.state.dispatched_job_count < self.configuration.problem.job_count:
            # Sample arrival time for the next job
            arrival_time = self.configuration.sampler.sample_next_arrival_time()
            yield self.configuration.environment.timeout(arrival_time)
            # Sample the job
            job = self.__sample_job__(created_at=self.configuration.environment.now)
            # Hold reference to the job
            self.history.with_new_job(job)
            # Send job to the first machine
            work_center = self.work_centers[job.step_idx[0]]

            self.__dispatch__(job, work_center)

    def will_produce(self, job: environment.Job, from_: environment.Machine):
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

        self.logger.info(
            f"Job {job.id} { job.current_step_idx } has been { 'completed' if job.is_completed else 'produced' } "
            f"on machine {from_.state.machine_idx} in work-center { from_.state.work_center_idx } "
            f"at {self.configuration.environment.now}. Jobs in the system { self.state.number_of_jobs_in_system }"
        )

    def __sample_job__(self, created_at: torch.FloatTensor = 0):
        job = (
            self.configuration.sampler.sample(job_id=self.state.with_new_job_id().job_id, moment=created_at)
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

    def __make_working_units__(self) -> Tuple[List[environment.WorkCenter], List[environment.Machine]]:
        work_centers = []
        machines = []

        for work_center_idx in range(self.configuration.problem.workcenter_count):
            work_centers += [environment.WorkCenter(self.configuration.environment,
                                                    work_center_idx,
                                                    rule=self.configuration.routing_rule)]

            batch = []

            for machine_idx in range(self.configuration.problem.machines_per_workcenter):
                batch += [environment.Machine(self.configuration.environment,
                                              machine_idx,
                                              work_center_idx,
                                              self.configuration.scheduling_rule)]

            machines += [batch]

        return work_centers, machines

    def __build__(self):
        work_centers, machines_per_wc = self.__make_working_units__()

        machines = reduce(lambda x, y: x + y, machines_per_wc, [])

        for work_center, machines_in_work_center in zip(work_centers, machines_per_wc):
            work_center.connect(machines_in_work_center, work_centers, self)

        for machine in machines:
            machine.connect(machines, work_centers, self)

        self._work_centers = work_centers
        self._machines = machines

        return self
