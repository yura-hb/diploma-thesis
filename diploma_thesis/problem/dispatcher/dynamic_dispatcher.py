import simpy
import torch

from typing import List
from dataclasses import dataclass, field

from problem.job import Job
from problem.problem import Problem
from problem.machine import Machine


@dataclass
class State:
    job_id: int = 0

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


class DynamicDispatcher:

    def __init__(
            self,
            environment: simpy.Environment,
            machines: List[Machine],
            description: Problem,
            generator: torch.Generator = torch.Generator()
    ):
        self.environment = environment
        self.machines = machines
        self.description = description
        self.generator = generator

        self.state = State()
        self.history = History()

        self.assign_initial_jobs()

        self.environment.process(self.dispatch_jobs())

    def assign_initial_jobs(self):
        seed = torch.arange(len(self.machines), dtype=torch.long)

        for idx, machine in enumerate(self.machines):
            seed = torch.randperm(seed, generator=self.generator)
            job = self.__sample_job__(seed, idx, created_at=0)
            # Hold reference to the job
            self.history.with_new_job(job)
            # Send job to machine
            machine.receive(job)

    def dispatch_jobs(self):
        seed = torch.arange(len(self.machines), dtype=torch.long)

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
            machine = self.machines[job.machine_idx[0]]
            machine.receive(job)

    def update_description(self, description: Problem):
        self.description = description

    def __sample_job__(self, seed: torch.LongTensor, initial_machine_idx: int = None, created_at: int = 0):
        machine_idx = None

        if initial_machine_idx is None:
            machine_idx = seed
        else:
            machine_idx = torch.hstack([torch.tensor(initial_machine_idx), *seed[seed != initial_machine_idx]])

        processing_times = self.description.sample_processing_times(machine_idx.size(), self.generator)

        job = (
            Job(
                id=self.state.with_new_job_id().job_id,
                machine_idx=machine_idx,
                processing_times=processing_times,
                created_at=torch.LongTensor([created_at])
            )
            .with_sampled_due_at(self.description.tightness_factor, len(self.machines))
            .with_arrival(self.environment.now)
        )

        return job

