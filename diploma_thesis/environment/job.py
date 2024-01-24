
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
        none = 3

    # Id of the job
    id: int
    # The sequence of work-centers ids, which job must visit in order to be complete
    step_idx: torch.LongTensor
    # The processing time of the job in workcenter & machine,
    # i.e. the tensor of shape (num_workcenters, num_machines_per_workcenter)
    processing_times: torch.LongTensor
    # The step in job sequence, which is currently being processed
    current_step_idx: int = -1
    # The index of the machine in the work-center where the job is being processed
    current_machine_idx: int = -1
    # The creation time of the job
    created_at: torch.FloatTensor = 0
    # The time of the job completion
    completed_at: torch.FloatTensor = 0
    # The due time of the job, i.e. deadline
    due_at: torch.FloatTensor = 0
    # The time, when each operation arrives to the specified machine
    arrived_at: torch.FloatTensor = field(default_factory=torch.LongTensor)
    # The list of the times, when operation was selected for processing on the workcenter
    started_at: torch.FloatTensor = field(default_factory=torch.LongTensor)
    # Slack, i.e. the amount of time, that the job can be postponed
    # It is calculated as the due_at - current_time - remaining_processing time, and it is recorded at the arrival
    # of the job on the machine
    slack: torch.FloatTensor = field(default_factory=torch.LongTensor)

    def __post_init__(self):
        self.arrived_at = torch.zeros_like(self.step_idx)
        self.started_at = torch.zeros_like(self.step_idx)
        self.slack = torch.zeros_like(self.step_idx)

    def processing_time_moments(self, reduction_strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The expectation and variance of processing times
        """
        processing_times = self.expected_processing_times(self.processing_times.float(), reduction_strategy)

        return torch.mean(processing_times), torch.std(processing_times)

    @property
    def current_operation_processing_time_on_machine(self):
        """
        Returns: The processing time of the current operation in machine
        """
        return self.processing_times[self.current_step_idx][self.current_machine_idx]

    def current_operation_processing_time_in_work_center(self, strategy: ReductionStrategy = ReductionStrategy.none):
        """
        Returns: Returns the processing time of the current operation in work center
        """
        processing_times = self.processing_times[self.current_step_idx].float()

        return self.expected_processing_times(processing_times, strategy)

    def operation_processing_time_in_work_center(
        self,
        work_center_idx: int,
        strategy: ReductionStrategy = ReductionStrategy.mean
    ):
        """
        Returns: Returns the processing time of the current operation in workcenter
        """
        work_center_idx = torch.argwhere(self.step_idx == work_center_idx).item()
        processing_times = self.processing_times[work_center_idx].float()

        return self.expected_processing_times(processing_times, strategy)

    def remaining_processing_time(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The total processing time of the remaining operations
        """
        # Since we don't know, to which machine inside work-center the job will be dispatched next, we
        # approximate it with the average
        if self.is_completed:
            return 0

        expected_processing_time = self.processing_times[max(self.current_step_idx, 0):]
        expected_processing_time = self.expected_processing_times(expected_processing_time.float(), strategy)

        return expected_processing_time.sum()

    def total_processing_time(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The total processing time of the job
        """
        return self.expected_processing_times(self.processing_times.float(), strategy).sum()

    @property
    def is_completed(self):
        return self.completed_at > 0

    @property
    def remaining_operations_count(self):
        """
        Returns: The number of remaining operations
        """
        return max(self.step_idx.shape[0] - self.current_step_idx, 0)

    @property
    def next_work_center_idx(self):
        """
        Returns: The index of the work-center to visit or None if the job is completed
        """
        next_idx = self.current_step_idx + 1

        if next_idx >= len(self.step_idx):
            return None

        return self.step_idx[next_idx]

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

    def slack_upon_now(self, now: torch.FloatTensor, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Args:
            now: Current time
            strategy: The strategy to use for calculating the slack

        Returns: The slack upon now of the job on the machine
        """
        return self.due_at - now - self.remaining_processing_time(strategy)

    def time_until_due(self, now: torch.FloatTensor):
        """
        Computes the remaining time until the job is due, i.e. the time until the deadline

        Args:
            now: Moment to start computing time until due
        """
        return self.due_at - now

    def current_operation_waiting_time(self, now: torch.FloatTensor):
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

    def time_completion_rate(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        The completion rate of the job based on the remaining processing time
        """
        return self.remaining_processing_time(strategy) / self.total_processing_time(strategy)

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

    def with_arrival(self, now: torch.FloatTensor, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Records arrival of the job on the next machine

        Args:
            now: Current time
            strategy: The strategy to use for calculating the slack

        Returns: Reference to self
        """
        return self.with_current_operation_arrival_time(now).with_current_operation_slack_upon_arrival(strategy)

    def with_current_operation_arrival_time(self, now: torch.FloatTensor):
        """
        Remembers the arrival time of the job

        Args:
            now: Current time

        Returns: Reference to self
        """
        self.arrived_at[self.current_step_idx] = now

        return self

    def with_current_operation_start_time(self, now: torch.FloatTensor):
        """
        Remembers the start time of the job

        Args:
            now: Current time

        Returns: Reference to self
        """
        self.started_at[self.current_step_idx] = now

        return self

    def with_current_operation_slack_upon_arrival(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Remembers slack upon arrival of the job

        Returns: Reference to self
        """
        self.slack[self.current_step_idx] = self.slack_upon_now(self.arrived_at[self.current_step_idx], strategy)

        return self

    def with_sampled_due_at(self, tightness, num_machines, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Generates due moment for the job

        Args:
            tightness: The tightness factor of the simulation. We suppose that tightness is uniformly distributed and
                the input is the upper bound of the distribution
            num_machines: The number of machines in system
            strategy: The strategy to use for calculating the estimated processing time

        Returns: Reference to self
        """
        mean, _ = self.processing_time_moments(strategy)
        tightness = torch.distributions.Uniform(1, tightness).rsample((1,))

        self.due_at = torch.round(mean * num_machines * tightness + self.created_at)

        return self

    def with_completion_time(self, time: torch.FloatTensor):
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
                return processing_times.min(axis=1)[0]
            case Job.ReductionStrategy.max:
                return processing_times.max(axis=1)[0]
            case Job.ReductionStrategy.none:
                return processing_times
            case _:
                raise ValueError(f"Unknown reduction strategy {strategy}")
