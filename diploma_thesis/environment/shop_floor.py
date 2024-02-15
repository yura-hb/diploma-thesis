import logging
from dataclasses import dataclass, field
from typing import List, Dict
from tensordict.prototype import tensorclass

import simpy
import torch

import environment
from .utils import ShopFloorFactory
from typing import Set


@tensorclass
class State:
    idx: str = 0

    job_id: int = 0

    in_system_job_ids: Set[int] = field(default_factory=set)

    @property
    def dispatched_job_count(self):
        return self.job_id

    def with_machines_count(self):
        return self

    def with_new_job_id(self):
        self.job_id += 1

        return self

    def with_new_job_in_system(self, job_id: int = 0):
        self.in_system_job_ids.add(job_id)

        return self

    def with_job_completed(self, job_id: int):
        self.in_system_job_ids.remove(job_id)

        return self


@tensorclass
class History:
    # A list of jobs, where each job is represented by the id of machine
    jobs: Dict[int, environment.Job] = field(default_factory=dict)

    started_at: int = 0

    def with_machines_count(self):
        return self

    def with_new_job(self, job: environment.Job):
        self.jobs[job.id.item()] = job

        return self

    def with_started_at(self, now: int):
        self.started_at = now

        return self

    def job(self, job_id: int):
        if isinstance(job_id, torch.Tensor):
            job_id = job_id.item()

        return self.jobs[job_id]


@tensorclass
class Map:
    @tensorclass
    class WorkCenter:
        idx: torch.LongTensor
        machines: torch.LongTensor

    work_centers: List[WorkCenter]


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
        self.id = idx
        self.configuration = configuration
        self.logger = logger
        self.agent = configuration.agent
        self.delegate = configuration.delegate

        self._work_centers: List[environment.WorkCenter] = []
        self._machines: List[environment.Machine] = []

        self.state = State(idx=idx, batch_size=[])
        self.history = History(batch_size=[])
        self._work_centers, self._machines = ShopFloorFactory(self.configuration, self).make()

        self.did_finish_simulation_event = configuration.environment.event()

    def simulate(self):
        torch.manual_seed(self.configuration.problem.seed)

        self.history.with_started_at(self.configuration.environment.now)
        self.delegate.did_start_simulation(context=self.__make_context__())

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

        return Statistics.from_shop_floor(self)

    @property
    def work_centers(self) -> List['environment.WorkCenter']:
        return self._work_centers

    @property
    def machines(self) -> List['environment.Machine']:
        return self._machines

    @property
    def map(self) -> Map:
        return Map(
            work_centers=[
                Map.WorkCenter(
                    idx=work_center.work_center_idx,
                    machines=torch.cat([machine.state.machine_idx for machine in work_center.machines]),
                    batch_size=[]
                )
                for work_center in self.work_centers
            ],
            batch_size=[]
        )

    @property
    def in_system_jobs(self) -> List['environment.Job']:
        return [self.history.job(job_id) for job_id in self.state.in_system_job_ids]

    def work_center(self, idx: int) -> 'environment.WorkCenter':
        return self.work_centers[idx]

    def machine(self, work_center_idx: int, machine_idx: int) -> 'environment.Machine':
        return self.work_centers[work_center_idx].machines[machine_idx]

    def work_in_next_queue(self, job: environment.Job) -> torch.FloatTensor:
        work_center_idx = job.next_work_center_idx

        if work_center_idx is None:
            return 0

        return self.work_centers[work_center_idx].work_load

    def average_waiting_in_next_queue(self, job: environment.Job) -> torch.FloatTensor:
        work_center_idx = job.next_work_center_idx

        if work_center_idx is None:
            return 0

        return self.work_centers[work_center_idx].average_waiting_time

    @property
    def completion_rate(self) -> torch.FloatTensor:
        in_system_jobs = self.in_system_jobs

        completed_operations_count = torch.LongTensor([
            job.processed_operations_count for job in in_system_jobs
        ]).sum()

        remaining_operations_count = torch.LongTensor([
            job.remaining_operations_count for job in in_system_jobs
        ]).sum()

        return completed_operations_count / remaining_operations_count

    def tardy_rate(self, now: int) -> torch.FloatTensor:
        in_system_jobs = self.in_system_jobs

        tardy_jobs = torch.LongTensor([
            job.is_tardy_at(now) for job in in_system_jobs
        ]).sum()

        return tardy_jobs / len(in_system_jobs)

    def expected_tardy_rate(
        self, now: float, reduction_strategy: environment.JobReductionStrategy
    ) -> torch.FloatTensor:
        in_system_jobs = self.in_system_jobs

        expected_tardy_jobs = torch.LongTensor([
            job.is_expected_to_be_tardy_at(now=now, strategy=reduction_strategy) for job in in_system_jobs
        ]).sum()

        return expected_tardy_jobs / len(in_system_jobs)

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

            self.state.with_job_completed(job.id)
            self.did_complete(job)

        self.logger.info(
            f"Job {job.id.item()} has been "
            f"{'completed' if job.is_completed else 'produced'} "
            f"on machine {from_.state.machine_idx.item()} in work-center {from_.state.work_center_idx.item()}. "
            f"Jobs in system {len(self.state.in_system_job_ids)}"
        )

    # Decision methods

    def schedule(self, machine: environment.Machine) -> 'environment.Job | environment.WaitInfo':
        return self.agent.schedule(self.__make_context__(), machine)

    def route(self, work_center: 'environment.WorkCenter', job: environment.Job) -> 'environment.Machine | None':
        return self.agent.route(self.__make_context__(), job=job, work_center=work_center)

    # Events from subcomponents (WorkCenter, Machine)
    def will_produce(self, job: environment.Job, machine: environment.Machine):
        self.delegate.will_produce(context=self.__make_context__(), job=job,  machine=machine)

    def did_produce(self, job: environment.Job, machine: environment.Machine):
        self.delegate.did_produce(context=self.__make_context__(), job=job, machine=machine)

    def will_dispatch(self, job: environment.Job, work_center: environment.WorkCenter):
        self.delegate.will_dispatch(context=self.__make_context__(), job=job, work_center=work_center)

    def did_dispatch(self, job: environment.Job, work_center: environment.WorkCenter, machine: environment.Machine):
        self.delegate.did_dispatch(context=self.__make_context__(), job=job,  work_center=work_center, machine=machine)

    def did_finish_dispatch(self, work_center: environment.WorkCenter):
        self.delegate.did_finish_dispatch(context=self.__make_context__(), work_center=work_center)

    def did_complete(self, job: environment.Job):
        self.delegate.did_complete(context=self.__make_context__(), job=job)
        self.__test_if_finished__()

    def did_breakdown(self, machine: environment.Machine, repair_time: torch.FloatTensor):
        self.logger.info(
            f'Machine {machine.state.machine_idx.item()} in work-center {machine.state.work_center_idx.item()} '
            f'has broken down. Repair time {repair_time.item()}'
        )

        self.delegate.did_breakdown(context=self.__make_context__(), machine=machine, repair_time=repair_time)

    def did_repair(self, machine: environment.Machine):
        self.logger.info(
            f'Machine {machine.state.machine_idx.item()} in work-center {machine.state.work_center_idx.item()} '
            'has been repaired'
        )

        self.delegate.did_repair(context=self.__make_context__(), machine=machine)

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

            work_center.__did_receive_job__()

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
        is_no_jobs_in_system = len(self.state.in_system_job_ids) == 0

        if not self.did_finish_simulation_event.triggered and has_dispatched_all_jobs and is_no_jobs_in_system:
            self.delegate.did_finish_simulation(context=self.__make_context__())

            self.did_finish_simulation_event.succeed()

    def __should_dispatch__(self):
        number_of_jobs = self.configuration.sampler.number_of_jobs()

        if number_of_jobs is not None:
            return self.state.dispatched_job_count < number_of_jobs

        timespan = self.configuration.environment.now - self.history.started_at

        return timespan < self.configuration.problem.timespan

    def __dispatch__(self, job: environment.Job, work_center: environment.WorkCenter):
        self.state.with_new_job_in_system(job.id)

        work_center.receive(job)

    def __make_context__(self) -> 'environment.Context':
        return environment.Context(
            shop_floor=self,
            moment=self.configuration.environment.now
        )
