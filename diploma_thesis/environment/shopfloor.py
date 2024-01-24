import simpy
import torch
import logging

from typing import List, Tuple
from dataclasses import dataclass, field
from functools import reduce

from environment.job import Job
from environment.problem import Problem
from environment.machine import Machine
from environment.work_center import WorkCenter

from scheduling_rules import SchedulingRule, FIFOSchedulingRule
from routing_rules import RoutingRule, EARoutingRule


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
    class ProductionRecord:
        job_id: int
        started_at: torch.FloatTensor
        duration: int
        work_center_idx: int
        machine_idx: int

    # A list of jobs, where each job is represented by the id of machine
    jobs: List[Job] = field(default_factory=list)

    production: List[List[int]] = field(default_factory=list)

    def with_machines_count(self):
        return self

    def with_production_info(self, job: Job, machine: Machine, now: int):
        self.production += [self.ProductionRecord(
            job_id=job.id,
            started_at=now,
            duration=job.current_operation_processing_time_on_machine,
            work_center_idx=machine.state.work_center_idx,
            machine_idx=machine.state.machine_idx
        )]

        return self

    def with_new_job(self, job):
        self.jobs += [job]

        return self


class ShopFloor:
    
    @dataclass
    class Configuration:
        environment: simpy.Environment = field(default_factory=simpy.Environment)
        problem: Problem = field(default_factory=Problem)
        scheduling_rule: SchedulingRule = field(default_factory=lambda: FIFOSchedulingRule())
        routing_rule: RoutingRule = field(default_factory=lambda: EARoutingRule())

    def __init__(
        self,
        configuration: Configuration,
        logger: logging.Logger,
        generator: torch.Generator = torch.Generator()
    ):
        self.configuration = configuration
        self.logger = logger
        self.generator = generator

        self.work_centers: List['WorkCenter'] = []
        self.machines: List['Machine'] = []

        self.state = State()
        self.history = History()
        
        self.__build__()

    def simulate(self):
        self.assign_initial_jobs()

        self.configuration.environment.process(self.dispatch_jobs())

    # Redefine generation of Work-Center paths

    def assign_initial_jobs(self):
        step_idx = torch.arange(self.configuration.problem.workcenter_count, dtype=torch.long)

        for work_center in self.work_centers:
            for _ in work_center.context.machines:
                step_idx = step_idx[torch.randperm(step_idx.shape[0], generator=self.generator)]

                job = self.__sample_job__(step_idx,
                                          initial_work_center_idx=work_center.state.idx,
                                          created_at=0)
                # Hold reference to the job
                self.history.with_new_job(job)
                # Send job to work-center
                self.__dispatch__(job, work_center)

            work_center.did_receive_job()

    def dispatch_jobs(self):
        seed = torch.arange(self.configuration.problem.workcenter_count, dtype=torch.long)

        while self.state.dispatched_job_count < self.configuration.problem.job_count:
            # Sample arrival time for the next job
            arrival_time = self.configuration.problem.sample_next_arrival_time(1)[0]
            yield self.configuration.environment.timeout(arrival_time)
            # Generate new job
            seed = seed[torch.randperm(seed.shape[0], generator=self.generator)]
            job = self.__sample_job__(seed, created_at=self.configuration.environment.now)
            # Hold reference to the job
            self.history.with_new_job(job)
            # Send job to the first machine
            work_center = self.work_centers[job.step_idx[0]]

            self.__dispatch__(job, work_center)

    def will_produce(self, job: Job, from_: Machine):
        # TODO: Record information from update_global_info_progression method in agent_machine.py
        # TODO: Record information from update_global_info_anticipation method in agent_machine.py
        self.history.with_production_info(job, from_, self.configuration.environment.now)

    def forward(self, job: Job, from_: Machine):
        if next_work_center_idx := job.next_work_center_idx:
            self.work_centers[next_work_center_idx].receive(job)
        else:
            job.with_completion_time(self.configuration.environment.now)
            self.state.with_job_completed()

        self.logger.info(
            f"Job {job.id} { job.current_step_idx } has been { 'completed' if job.is_completed else 'produced' } "
            f"on machine {from_.state.machine_idx} in work-center { from_.state.work_center_idx } "
            f"at {self.configuration.environment.now}. Jobs in the system { self.state.number_of_jobs_in_system }"
        )

    def update_problem(self, problem: Problem):
        assert self.configuration.problem.machines_per_workcenter == problem.machines_per_workcenter, \
               "Expected the same number of machines per work center"
        assert self.configuration.problem.workcenter_count == problem.workcenter_count, \
               "Expected the same number of work centers"

        self.configuration.problem = problem

    def __sample_job__(self,
                       step_idx: torch.LongTensor,
                       initial_work_center_idx: int = None,
                       created_at: torch.FloatTensor = 0):
        work_center_idx = None

        if initial_work_center_idx is None:
            work_center_idx = step_idx
        else:
            work_center_idx = torch.hstack([
                torch.tensor(initial_work_center_idx), *step_idx[step_idx != initial_work_center_idx]
            ])

        operations_count = self.configuration.problem.machines_per_workcenter * self.configuration.problem.workcenter_count
        processing_times = (
            self.configuration.problem
                .sample_processing_times(operations_count, self.generator)
                .reshape(self.configuration.problem.workcenter_count, self.configuration.problem.machines_per_workcenter)
        )

        job = (
            Job(
                id=self.state.with_new_job_id().job_id,
                step_idx=work_center_idx,
                processing_times=processing_times,
                created_at=torch.FloatTensor([created_at])
            )
            .with_sampled_due_at(self.configuration.problem.tightness_factor, len(self.machines))
            .with_arrival(self.configuration.environment.now)
        )

        return job

    def __dispatch__(self, job: Job, work_center: WorkCenter):
        work_center.receive(job)

        self.state.with_new_job_in_system()

    def __make_working_units__(self) -> Tuple[List[WorkCenter], List[Machine]]:
        work_centers = []
        machines = []

        for work_center_idx in range(self.configuration.problem.workcenter_count):
            work_centers += [WorkCenter(self.configuration.environment,
                                        work_center_idx,
                                        rule=self.configuration.routing_rule)]

            batch = []

            for machine_idx in range(self.configuration.problem.machines_per_workcenter):
                batch += [Machine(self.configuration.environment,
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

        self.work_centers = work_centers
        self.machines = machines

        return self
