
import torch

from dataclasses import dataclass, field


@dataclass
class Job:
    # Id of the job
    id: int

    # The sequence of machines that the job has to visit
    machine_idx: torch.LongTensor
    # The processing time of the job on each machine
    processing_times: torch.LongTensor

    # The index of the current operation to complete
    current_operation_idx: int = 0

    # The creation time of the job
    created_at: torch.LongTensor = 0
    # The time of the job is completed
    completed_at: torch.LongTensor = 0
    # The due time of the job
    due_at: torch.LongTensor = 0

    # The time, when each operation arrives to the specified machine
    arrived_at: torch.LongTensor = field(default_factory=torch.LongTensor)
    # The list of the times, when operation was initiated for processing
    started_at: torch.LongTensor = field(default_factory=torch.LongTensor)
    # Slack, i.e. the amount of time, that the job can be postponed
    # It is calculated as the due_at - current_time - remaining_processing time, and it is recorded at the arrival
    # of the job on the machine
    slack: torch.LongTensor = field(default_factory=torch.LongTensor)

    def __post_init__(self):
        self.arrived_at = torch.zeros_like(self.processing_times)
        self.started_at = torch.zeros_like(self.processing_times)
        self.stack = torch.zeros_like(self.processing_times)

    @property
    def processing_time_properties(self):
        """
        Returns: The expectation and variance of processing times
        """
        return torch.mean(self.processing_times), torch.std(self.processing_times)

    @property
    def current_operation_processing_time(self):
        return self.processing_times[self.current_operation_idx]

    @property
    def remaining_processing_time(self):
        return self.processing_times[self.current_operation_idx:].sum()

    @property
    def remaining_operations_count(self):
        return len(self.processing_times) - self.current_operation_idx

    @property
    def next_machine_idx(self):
        next_idx = self.current_operation_idx + 1

        if next_idx >= len(self.machine_idx):
            return None

        return self.machine_idx[next_idx]

    @property
    def next_operation_processing_time(self):
        next_idx = self.current_operation_idx + 1

        if next_idx >= len(self.machine_idx):
            return 0

        return self.processing_times[next_idx]

    @property
    def slack_upon_arrival(self):
        return self.slack[self.current_operation_idx]

    def slack_upon_now(self, now: int):
        return self.due_at - now - self.remaining_processing_time

    def time_until_due(self, now: int):
        """
        Computes the time until the job is due

        Args:
            now: Moment to start computing time until due
        """
        return self.due_at - now

    def current_operation_waiting_time(self, now: int):
        return now - self.arrived_at[self.current_operation_idx]

    def with_arrival(self, now: int):
        return self.with_current_operation_arrival_time(now).with_current_operation_slack_upon_arrival(now)

    def with_current_operation_arrival_time(self, now: int):
        """
        Remembers the arrival time of the job

        Args:
            now: Time

        Returns: Self
        """
        self.arrived_at[self.current_operation_idx] = now

        return self

    def with_current_operation_start_time(self, now: int):
        """
        Remembers the start time of the job

        Args:
            now: Time

        Returns: Self
        """
        self.started_at[self.current_operation_idx] = now

        return self

    def with_current_operation_slack_upon_arrival(self):
        """
        Remembers slack upon arrival of the job

        Args:
            now: Slack of the job

        Returns: Self
        """
        self.slack[self.current_operation_idx] = self.slack_upon_now(self.arrived_at[self.current_operation_idx])

        return self

    def with_sampled_due_at(self, tightness, num_machines):
        """
        Generates due time for the job

        Args:
            tightness: The tightness factor of the simulation
            num_machines: The number of machines in system

        Returns: Self
        """
        mean, _ = self.processing_time_properties
        tightness = torch.distributions.Uniform(1, tightness).rsample((1,))

        self.due_at = torch.round(mean * num_machines * tightness + self.created_at)

        return self

    def with_completion_time(self, time: int):
        """
        Remembers the completion time of the job

        Args:
            time: Completion time of the job

        Returns: Self
        """
        self.completed_at = time

        return self
