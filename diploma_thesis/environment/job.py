
import torch

from dataclasses import dataclass, field
from enum import Enum


@dataclass
class Job:

    class ReductionStrategy(Enum):
        """
        Job doesn't know in advance on which machine it will be processed inside work-center. ReductionStrategy
        defines the way, how the expected processing time on the work-center is calculated
        """
        mean = 0
        min = 1
        max = 2

    # Id of the job
    id: int
    # The sequence of work-centers, which job must visit in order to be complete
    step_idx: torch.LongTensor
    # The processing time of the job in workcenter depending on the machine it was scheduled
    processing_times: torch.LongTensor
    # The index of the workcenter that the job is currently visiting
    current_step_idx: int = 0
    # The index of the machine in the work-center where the job is being processed
    current_machine_idx: int = 0
    # The creation time of the job
    created_at: torch.LongTensor = 0
    # The time of the job completion
    completed_at: torch.LongTensor = 0
    # The due time of the job, i.e. deadline
    due_at: torch.LongTensor = 0
    # The time, when each operation arrives to the specified machine
    arrived_at: torch.LongTensor = field(default_factory=torch.LongTensor)
    # The list of the times, when operation was selected for processing on the workcenter
    started_at: torch.LongTensor = field(default_factory=torch.LongTensor)
    # Slack, i.e. the amount of time, that the job can be postponed
    # It is calculated as the due_at - current_time - remaining_processing time, and it is recorded at the arrival
    # of the job on the machine
    slack: torch.LongTensor = field(default_factory=torch.LongTensor)

    def __post_init__(self):
        self.arrived_at = torch.zeros_like(self.step_idx)
        self.started_at = torch.zeros_like(self.step_idx)
        self.slack = torch.zeros_like(self.step_idx)

    @property
    def processing_time_moments(self, reduction_strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The expectation and variance of processing times
        """
        processing_times = self.expected_processing_times(self.processing_times.float(), reduction_strategy)

        return torch.mean(processing_times), torch.std(processing_times)

    @property
    def current_operation_processing_time_on_machine(self):
        """
        Returns: Returns the processing time of the current operation in machine
        """
        return self.processing_times[self.current_step_idx][self.current_machine_idx]

    @property
    def current_operation_processing_time_in_workcenter(self):
        """
        Returns: Returns the processing time of the current operation in workcenter
        """
        return self.processing_times[self.current_step_idx]

    @property
    def remaining_processing_time(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The total processing time of the remaining operations
        """

        # Since we don't know, to which machine inside work-center the job will be dispatched next, we
        # approximate it with the average
        expected_processing_time = self.processing_times[self.current_step_idx:]
        expected_processing_time = self.expected_processing_times(expected_processing_time.float(), strategy)

        return expected_processing_time.sum()

    @property
    def remaining_operations_count(self):
        """
        Returns: The number of remaining operations
        """
        return self.processing_times.shape[0] - self.current_step_idx

    @property
    def next_work_center_idx(self):
        """
        Returns: The index of the work-center to visit or None if the job is completed
        """
        next_idx = self.current_step_idx + 1

        if next_idx >= len(self.step_idx):
            return None

        return self.step_idx[next_idx]

    @property
    def next_operation_processing_time(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The processing time of the next operation
        """
        next_idx = self.current_step_idx + 1

        if next_idx >= len(self.step_idx):
            return 0

        pt = self.processing_times[next_idx]

        return self.expected_processing_times(pt.float(), strategy)

    @property
    def slack_upon_arrival(self):
        """
        Returns: The slack upon arrival of the job on the machine
        """
        return self.slack[self.current_step_idx]

    def slack_upon_now(self, now: int):
        """
        Args:
            now: Current time

        Returns: The slack upon now of the job on the machine
        """
        return self.due_at - now - self.remaining_processing_time

    def time_until_due(self, now: int):
        """
        Computes the remaining time until the job is due, i.e. the time until the deadline

        Args:
            now: Moment to start computing time until due
        """
        return self.due_at - now

    def current_operation_waiting_time(self, now: int):
        """
        Args:
            now: Current time

        Returns: The time that the current operation has been waiting for processing on current machine
        """
        return now - self.arrived_at[self.current_step_idx]

    def operation_completion_rate(self):
        """
        The completion rate of the job based on the number of completed operations
        """
        return self.remaining_operations_count / len(self.step_idx)

    def time_completion_rate(self, now: int):
        """
        The completion rate of the job based on the remaining processing time
        """
        return self.remaining_processing_time / self.processing_times.sum()
    def with_next_step(self):
        """
        Advances the job to the next work-center
        """
        self.current_step_idx += 1
        self.current_machine_idx = -1

        return self

    def with_assigned_machine(self, machine_idx: int):
        """
        Advances the job to the next machine
        """
        self.current_machine_idx = machine_idx

        return self

    def with_arrival(self, now: int):
        """
        Records arrival of the job on the next machine

        Args:
            now: Current time

        Returns: Reference to self
        """
        return self.with_current_operation_arrival_time(now).with_current_operation_slack_upon_arrival()

    def with_current_operation_arrival_time(self, now: int):
        """
        Remembers the arrival time of the job

        Args:
            now: Current time

        Returns: Reference to self
        """
        self.arrived_at[self.current_step_idx] = now

        return self

    def with_current_operation_start_time(self, now: int):
        """
        Remembers the start time of the job

        Args:
            now: Current time

        Returns: Reference to self
        """
        self.started_at[self.current_step_idx] = now

        return self

    def with_current_operation_slack_upon_arrival(self):
        """
        Remembers slack upon arrival of the job

        Args:
            now: Slack of the job

        Returns: Reference to self
        """
        self.slack[self.current_step_idx] = self.slack_upon_now(self.arrived_at[self.current_step_idx])

        return self

    def with_sampled_due_at(self, tightness, num_machines):
        """
        Generates due moment for the job

        Args:
            tightness: The tightness factor of the simulation. We suppose that tightness is uniformly distributed and
                the input is the upper bound of the distribution
            num_machines: The number of machines in system

        Returns: Reference to self
        """
        mean, _ = self.processing_time_moments
        tightness = torch.distributions.Uniform(1, tightness).rsample((1,))

        self.due_at = torch.round(mean * num_machines * tightness + self.created_at)

        return self

    def with_completion_time(self, time: int):
        """
        Remembers the completion time of the job

        Args:
            time: Completion time of the job

        Returns: Reference to self
        """
        self.completed_at = time

        return self

    @staticmethod
    def expected_processing_times(processing_times, strategy: ReductionStrategy):
        processing_times = torch.atleast_2d(processing_times)

        match strategy:
            case Job.ReductionStrategy.mean:
                return processing_times.mean(axis=1)
            case Job.ReductionStrategy.min:
                return processing_times.min(axis=1)
            case Job.ReductionStrategy.max:
                return processing_times.max(axis=1)
            case _:
                raise ValueError(f"Unknown reduction strategy {strategy}")
