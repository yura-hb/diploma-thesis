import simpy
import torch

from typing import List
from dataclasses import dataclass, field

from environment.job import Job
from environment.problem import Problem
from environment.machine import Machine
from environment.work_center import WorkCenter


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


@dataclass
class History:
    # A list of jobs, where each job is represented by the id of machine
    jobs: List[Job] = field(default_factory=list)

    def with_machines_count(self):
        return self

    def with_new_job(self, job):
        self.jobs += [job]

        return self


class ShopFloor:

    def __init__(
        self,
        environment: simpy.Environment,
        machines: List[Machine],
        work_centers: List[WorkCenter],
        description: Problem,
        generator: torch.Generator = torch.Generator()
    ):
        self.environment = environment
        self.machines = machines
        self.work_centers = work_centers
        self.description = description
        self.generator = generator

        self.state = State()
        self.history = History()

    def simulate(self):
        self.assign_initial_jobs()

        self.environment.process(self.dispatch_jobs())

    # Redefine generation of Work-Center paths

    def assign_initial_jobs(self):
        step_idx = torch.arange(self.description.workcenter_count, dtype=torch.long)

        for work_center in self.work_centers:
            for _ in work_center.context.machines:
                step_idx = step_idx[torch.randperm(step_idx.shape[0], generator=self.generator)]

                job = self.__sample_job__(step_idx,
                                          initial_work_center_idx=work_center.state.idx,
                                          created_at=0)
                # Hold reference to the job
                self.history.with_new_job(job)
                # Send job to work-center
                work_center.receive(job)

            work_center.on_route.succeed()

    def dispatch_jobs(self):
        seed = torch.arange(self.description.workcenter_count, dtype=torch.long)

        while self.state.dispatched_job_count < self.description.job_count:
            # Sample arrival time for the next job
            arrival_time = self.description.sample_next_arrival_time(1)[0]
            yield self.environment.timeout(arrival_time)
            # Generate new job
            seed = torch.randperm(seed, generator=self.generator)
            job = self.__sample_job__(seed, created_at=self.environment.now)
            # Hold reference to the job
            self.history.with_new_job(job)
            # Send job to the first machine
            workcenter = self.work_centers[job.step_idx[0]]
            workcenter.receive(job)

    def will_forward(self, job: Job, from_: Machine):
        # TODO: Record information from update_global_info_progression method in agent_machine.py
        # TODO: Record information from update_global_info_anticipation method in agent_machine.py
        ...

    def forward(self, job: Job, from_: Machine):
        ...

    def update_description(self, description: Problem):
        self.description = description

    def __sample_job__(self,
                       step_idx: torch.LongTensor,
                       initial_work_center_idx: int = None,
                       created_at: int = 0):
        work_center_idx = None

        if initial_work_center_idx is None:
            work_center_idx = step_idx
        else:
            work_center_idx = torch.hstack([
                torch.tensor(initial_work_center_idx), *step_idx[step_idx != initial_work_center_idx]
            ])

        operations_count = self.description.machines_per_workcenter * self.description.workcenter_count
        processing_times = (
            self.description
                .sample_processing_times(operations_count, self.generator)
                .reshape(self.description.workcenter_count, self.description.machines_per_workcenter)
        )

        job = (
            Job(
                id=self.state.with_new_job_id().job_id,
                step_idx=work_center_idx,
                processing_times=processing_times,
                created_at=torch.LongTensor([created_at])
            )
            .with_sampled_due_at(self.description.tightness_factor, len(self.machines))
            .with_arrival(self.environment.now)
        )

        return job

