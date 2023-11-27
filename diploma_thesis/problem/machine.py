

from dataclasses import dataclass, field
from typing import List, Any, Tuple

import simpy
import torch

from collections import namedtuple
from job import Job
from scheduling_rules import SchedulingRule


BreakdownRecord = namedtuple('BreakdownRecord', ['start', 'duration'])


@dataclass
class State:
    id: int = 0

    # The list of jobs, which are queued on the machine
    queue: List[Job] = field(default_factory=list)
    # The index of current processing job
    current_job_idx: int = 0
    # The moment of decision for the machine
    decision_time: int = 0
    # Total processing time for each job in the queue
    total_processing_time: int = 0
    # Expected moment of machine availability, i.e. without breakdowns and maintenances
    available_at: int = 0

    def with_new_job(self, job: Job, now: int):
        self.queue += [job]

        return self.with_updated_information(now)

    def with_decision_time(self, decision_time: int):
        self.decision_time = decision_time

        return self

    def with_updated_information(self, now: int):
        self.total_processing_time = torch.sum((job.current_operation_processing_time for job in self.queue))
        self.available_at = now + self.total_processing_time

        return self


@dataclass
class History:
    number_of_jobs: List[int] = field(default_factory=list)

    # Breakdowns in format (start, duration)
    breakdowns: List[BreakdownRecord] = field(default_factory=list)

    def with_number_of_jobs(self, state: State):
        self.number_of_jobs += [len(state.queue)]

        return self

    def with_breakdown_record(self, start: int, end: int):
        self.breakdowns += [BreakdownRecord(start, end - start)]

        return self


@dataclass
class Context:
    machine_list: List[Any] = None
    dispatcher: Any = None

    def with_info(self, machine_list, dispatcher):
        self.machine_list = machine_list
        self.dispatcher = dispatcher

        return self

    # TODO: Verify
    @property
    def operation_count(self):
        return len(self.machine_list)


class Machine:

    def __init__(self, environment: simpy.Environment, idx: int, rule: SchedulingRule):
        self.environment = environment
        self.rule = rule

        self.state = State(idx)
        self.history = History()
        self.context = Context()

        # Events
        self.did_dispatch_event = self.environment.event()
        self.is_on_event = self.environment.event()

        # Initially, machine is working
        self.is_on_event.succeed()

    def connect(self, machine_list: List, shopfloor: Any):
        """
        Connects machine to other machines and dispatcher,
         so that machine can communicate with them

        Args:
            machine_list: A list of machines
            shopfloor: Shopfloor

        Returns: Nothing
        """
        self.context = self.context.with_info(machine_list, shopfloor)

        self.environment.process(self.produce())

    def receive(self, job: Job):
        self.state.with_new_job(job, self.environment.now)

        # Unblock the machine, if it is waiting for the job
        try:
            self.did_dispatch_event.succeed()
        except Exception as exc:
            # TODO: Implement logging
            print(exc)

    def produce(self):
        if not self.did_dispatch_event.triggered:
            yield self.environment.process(self.starve())

        while True:
            self.state.with_decision_time(self.environment.now)
            self.history.with_number_of_jobs(self.state)

            job = self.select_job()

            processing_time = job.current_operation_processing_time
            waiting_time = job.current_operation_waiting_time(self.environment.now)

            # Perform the operation of the job
            yield self.environment.timeout(processing_time)

            self.forward(job)
            self.breakdown_if_needed()

            # Starve, if there is no job in the queue
            if not len(self.state.queue):
                yield self.environment.process(self.starve())

    def starve(self):
        self.did_dispatch_event = self.environment.event()

        yield self.did_dispatch_event

        self.breakdown_if_needed()

    def select_job(self):

        return ...

    def breakdown(self):
        start = self.environment.now

        yield self.is_on_event

        self.history.with_breakdown_record(start, self.environment.now)

    def breakdown_if_needed(self):
        if not self.is_on_event.triggered:
            yield self.environment.process(self.breakdown())

    def forward(self, job: Job):
        next_machine = job.next_machine_idx

        if next_machine is None:
            # TODO: Delete job
            return

        machine = self.context.machine_list[next_machine]

        # TODO: advance

        machine.receive(job)
