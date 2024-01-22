
from functools import reduce

from dataclasses import dataclass, field
from typing import List, Any, Tuple

import simpy
import torch

from collections import namedtuple
from job import Job
from scheduling_rules import SchedulingRule
from shopfloor import ShopFloor
from work_center import WorkCenter

# TODO: WINQ (agent_machine.py: 365)
# TODO: AVLM (agent_machine.py: 366)
# TODO: Machine Tardiness and

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
    queue: List[Job] = field(default_factory=list)
    # The index of current processing job
    current_job_idx: int = 0
    # The moment of decision for the machine
    decision_time: int = 0
    # Total processing time for each job in the queue
    total_processing_time: int = 0
    # Expected moment of machine availability, i.e. without breakdowns and maintenances
    available_at: int = 0
    # The time of machine recover from breakdown
    restart_at: int = 0

    def with_new_job(self, job: Job, now: int):
        """
        Adds new job to the queue

        Args:
            job: Job to schedule into queue on the machine
            now: Timestamp when the job has arrived at the machine

        Returns: Reference to self

        """
        self.queue += [job]

        return self.with_updated_information(now)

    def with_decision_time(self, decision_time: int):
        """
        Sets the decision time for the machine

        Args:
            decision_time: Time of the decision

        Returns: Reference to self

        """
        self.decision_time = decision_time

        return self

    def with_updated_information(self, now: int):
        """
        Updates total processing time of scheduled operations and the theoretical moment of machine availability

        Args:
            now: Current time

        Returns: Reference to self

        """
        self.total_processing_time = torch.sum((job.current_operation_processing_time for job in self.queue))
        self.available_at = now + self.total_processing_time

        return self

    def with_breakdown(self, restart_at: int):
        """
        Sets machine into breakdown state

        Args:
            restart_at: Moment of machine restart

        Returns: Reference to self
        """

        self.restart_at = restart_at
        self.available_at = self.restart_at + self.total_processing_time

    @property
    def is_empty(self):
        """
        Returns: True if the queue is empty, False otherwise
        """
        return len(self.queue) == 0

    def without_job(self, job_id: int, now: int):
        """
        Removes job from the queue

        Args:
            job_id: Job to remove from the queue
            now: Current time

        Returns: Reference to self
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
    """
    Support class representing the history of the machine
    """

    # Number of jobs, when machine selects the job
    recorded_at: List[int] = field(default_factory=list)
    number_of_jobs: List[int] = field(default_factory=list)

    # Breakdowns of the machine
    breakdown_start_at: List[int] = field(default_factory=list)
    breakdown_end_at: List[int] = field(default_factory=list)

    # TODO: - Implement
    rewards: List = field(default_factory=list)

    # Total runtime of operations on the machine
    run_time: int = 0

    def with_number_of_jobs(self, state: State, now: int):
        self.recorded_at += [now]
        self.number_of_jobs += [len(state.queue)]

        return self

    def with_breakdown_record(self, start: int, end: int):
        self.breakdown_start_at += [start]
        self.breakdown_end_at += [end]

        return self

    def with_runtime(self, operation_processing_time: int):
        self.run_time += operation_processing_time

        return operation_processing_time


@dataclass
class Context:
    machines: List['Machine'] = None
    work_centers: List[WorkCenter] = None
    shopfloor: Any = None

    def with_info(self, machines: List['Machine'], work_centers: List[WorkCenter], shopfloor: ShopFloor):
        self.machines = machines
        self.work_centers = work_centers
        self.shopfloor = shopfloor

        return self


class Machine:

    def __init__(self,
                 environment: simpy.Environment,
                 machine_idx: int,
                 work_center_idx: int,
                 rule: SchedulingRule):
        self.environment = environment
        self.rule = rule

        self.state = State(machine_idx=machine_idx, work_center_idx=work_center_idx)

        self.history = History()
        self.context = Context()

        # Events
        self.did_dispatch_event = self.environment.event()
        self.is_on_event = self.environment.event()

        # Initially, machine is working
        self.is_on_event.succeed()

        # TODO: Implement
        self.sequence_learning_event = self.environment.event()
        self.routing_learning_event = self.environment.event()

    def connect(self, machines: List['Machine'], work_centers: List[WorkCenter], shopfloor: ShopFloor):
        """
        Connects machine to other machines and dispatcher,
        so that machine can communicate with them

        Args:
            machines: List of all machines in the shop-floor
            work_centers: List of all work-centers in the shop-floor
            shopfloor: Shop-floor which implements the dispatching logic

        Returns: Nothing
        """
        self.context = self.context.with_info(machines, work_centers, shopfloor)

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
            self.history.with_number_of_jobs(self.state, now=self.environment.now)

            job = self.select_job()

            processing_time = job.current_operation_processing_time

            self.context.shopfloor.will_forward(job, self)

            # Perform the operation of the job
            yield self.environment.timeout(processing_time)

            self.history.with_runtime(processing_time)

            self.forward(job)
            self.breakdown_if_needed()
            self.starve_if_needed()

    def starve(self):
        self.did_dispatch_event = self.environment.event()

        yield self.did_dispatch_event

        self.breakdown_if_needed()

    def select_job(self):
        assert not self.state.is_empty

        if len(self.state.queue) == 1:
            return self.state.queue[0]

        # TODO: Check if the state of machine is the same as state of the rule
        job = self.rule(self.state)

        self.job_selected_with_rule_will_produce(job)

        return job

    def breakdown(self, restart_at: int):
        start = self.environment.now

        self.state.with_breakdown(restart_at)

        yield self.is_on_event

        self.history.with_breakdown_record(start, self.environment.now)

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

    def forward(self, job: Job):
        """
        Forwards the job to the next machine by sending it to the shopfloor

        Args:
            job: Job to forward

        Returns: None

        """
        self.context.shopfloor.forward(job)

        if job.next_machine_idx is None:
            self.state.without_job(job.id, now=self.environment.now)
            return

    def job_selected_with_rule_will_produce(self, job: Job):
        # TODO: before_operation from agent_machine.py: 194
        ...

    def job_did_produce(self, job: Job):
        # TODO: after_operation from agent_machine.py: 238
        ...
