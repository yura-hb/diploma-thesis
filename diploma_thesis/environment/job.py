
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache

import torch


class ReductionStrategy(Enum):
    """
    Job doesn't know in advance on which machine it will be processed inside work-center. ReductionStrategy
    defines the way, how the expected processing time on the work-center is calculated
    """
    mean = 0
    min = 1
    max = 2
    none = 3


def reduce(values, strategy: ReductionStrategy):
    values = torch.atleast_2d(values)

    match strategy:
        case ReductionStrategy.mean:
            return values.mean(axis=1)
        case ReductionStrategy.min:
            return values.min(axis=1)[0]
        case ReductionStrategy.max:
            return values.max(axis=1)[0]
        case ReductionStrategy.none:
            return values
        case _:
            raise ValueError(f"Unknown reduction strategy {strategy}")


@dataclass
class JobEvent:

    class Kind(Enum):
        creation = auto()
        dispatch = auto()
        forward = auto()
        arrival_on_work_center = auto()
        arrival_on_machine = auto()
        production_start = auto()
        production_end = auto()
        completion = auto()

    moment: torch.FloatTensor
    kind: Kind
    machine_idx: int = None
    work_center_idx: int = None


@dataclass
class Job:

    Event = JobEvent

    @dataclass
    class History:
        # The creation time of the job
        created_at: torch.FloatTensor = torch.FloatTensor([0.0])
        # The time, when job was pushed into system
        dispatched_at: torch.FloatTensor = torch.FloatTensor([0.0])
        # The creation time of the job
        completed_at: torch.FloatTensor = torch.FloatTensor([0.0])
        # The list of the times, when operation was started to be processed on the machine
        started_at: torch.FloatTensor = field(default_factory=torch.FloatTensor)
        # The list of the times, when operation has finished to be processed on the machine
        finished_at: torch.FloatTensor = field(default_factory=torch.FloatTensor)
        # The time, when each operation arrives to the specified machine
        arrived_at_work_center: torch.FloatTensor = field(default_factory=torch.FloatTensor)
        # The list of the times, when operation was selected for processing on the machine
        arrived_at_machine: torch.FloatTensor = field(default_factory=torch.FloatTensor)
        # The list of the times, when operation was selected for processing on the work center
        arrived_machine_idx: torch.LongTensor = field(default_factory=torch.LongTensor)

        def configure(self, step_idx: torch.LongTensor):
            self.started_at = torch.zeros_like(step_idx) - 1
            self.finished_at = torch.zeros_like(step_idx) - 1
            self.arrived_at_work_center = torch.zeros_like(step_idx) - 1
            self.arrived_at_machine = torch.zeros_like(step_idx) - 1
            self.arrived_machine_idx = torch.zeros_like(step_idx) - 1

        def with_event(self, event: JobEvent, step_idx: torch.LongTensor):
            def get_work_center_idx():
                return torch.argwhere(step_idx == event.work_center_idx).item()

            match event.kind:
                case JobEvent.Kind.creation:
                    self.created_at = torch.FloatTensor([event.moment])
                case JobEvent.Kind.dispatch:
                    self.dispatched_at = torch.FloatTensor([event.moment])
                case JobEvent.Kind.arrival_on_work_center:
                    idx = get_work_center_idx()

                    self.arrived_at_work_center[idx] = event.moment
                case JobEvent.Kind.arrival_on_machine:
                    idx = get_work_center_idx()

                    self.arrived_at_machine[idx] = event.moment
                    self.arrived_machine_idx[idx] = event.machine_idx
                case JobEvent.Kind.production_start:
                    idx = get_work_center_idx()

                    self.started_at[idx] = event.moment
                case JobEvent.Kind.production_end:
                    idx = get_work_center_idx()

                    self.finished_at[idx] = event.moment
                case JobEvent.Kind.completion:
                    self.completed_at = torch.FloatTensor([event.moment])
                case _:
                    pass

            return self

    # Id of the job
    id: torch.LongTensor = -1
    # The sequence of work-centers ids, which job must visit in order to be complete
    step_idx: torch.LongTensor = field(default_factory=torch.LongTensor)
    # The processing time of the job in work_center & machine,
    # i.e. the tensor of shape (num_work_centers, num_machines_per_work_center)
    processing_times: torch.LongTensor = field(default_factory=torch.LongTensor)
    # The priority of the Job
    priority: torch.FloatTensor = torch.tensor([1.0])
    # The due time of the job, i.e. deadline
    due_at: torch.FloatTensor = torch.tensor([0.0])

    # The step in job sequence, which is currently being processed
    current_step_idx: torch.LongTensor = torch.tensor(-1)
    # The index of the machine in the work-center where the job is being processed
    current_machine_idx: torch.LongTensor = torch.tensor(-1)
    # History of the job
    history: History = None

    def __post_init__(self):
        if not torch.is_tensor(self.id):
            self.id = torch.tensor(self.id)

        if self.history is None:
            self.history = Job.History()
            self.history.configure(self.step_idx)
            self.history = self.history

        assert 0.0 <= self.priority <= 1.0, "Priority must be in range [0, 1]"

    def processing_time_moments(self, reduction_strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The expectation and variance of processing times
        """
        processing_times = reduce(self.processing_times.float(), reduction_strategy)

        return torch.mean(processing_times), torch.std(processing_times)

    @property
    def n_steps(self):
        return self.step_idx.shape[0]

    @property
    def current_operation_processing_time_on_machine(self):
        """
        Returns: The processing time of the current operation in machine
        """
        return self.__processing_time_on_machine__(
            self.step_idx, self.processing_times, self.current_step_idx, self.current_machine_idx
        )

    def current_operation_processing_time_in_work_center(self, strategy: ReductionStrategy = ReductionStrategy.none):
        """
        Returns: Returns the processing time of the current operation in work center
        """
        return self.__processing_time_on_work_center__(
            self.step_idx, self.processing_times, self.current_step_idx, strategy
        )

    def operation_processing_time_in_work_center(
        self, work_center_idx: int, strategy: ReductionStrategy = ReductionStrategy.mean
    ):
        """
        Returns: Returns the processing time of the current operation in workcenter
        """
        step_idx = torch.argwhere(self.step_idx == work_center_idx).item()

        return self.__processing_time_on_work_center__(
            self.step_idx, self.processing_times, step_idx, strategy
        )

    def remaining_processing_time(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The total processing time of the remaining operations
        """
        return self.__remaining_processing_time__(
            self.step_idx, self.processing_times, self.current_step_idx, self.current_machine_idx, strategy
        )

    def next_remaining_processing_time(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The remaining processing time of the operation excluding processing time on current machine
        """
        return self.__next_remaining_processing_time__(self.processing_times, self.current_step_idx, strategy)

    def total_processing_time(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The total processing time of the job
        """
        return reduce(self.processing_times.float(), strategy).sum()

    @property
    def release_moment_on_machine(self):
        """
        Returns: Release time of job from machine or None if the job isn't processed at this moment
        """
        if not self.is_being_processed_on_machine:
            return None

        started_at = self.history.started_at[self.current_step_idx]

        return started_at + self.current_operation_processing_time_on_machine

    @property
    def is_being_processed_on_machine(self):
        """
        Returns: True if the job is currently being processed on machine
        """
        return (self.current_machine_idx >= 0
                and not self.is_completed
                and self.is_dispatched
                and self.history.started_at[self.current_step_idx] > 0)

    @property
    def is_completed(self):
        return self.history.completed_at > 0

    @property
    def is_dispatched(self):
        return self.current_step_idx >= 0

    @property
    def tardiness_upon_completion(self) -> torch.FloatTensor:
        """
        Returns: The tardiness of the job
        """
        assert self.is_completed, "Job must be completed in order to compute tardiness"

        return torch.tensor([max(self.history.completed_at - self.due_at, 0.0)])

    @property
    def flow_time_upon_completion(self) -> torch.FloatTensor:
        """
        Returns: The flow time of the job
        """
        assert self.is_completed, "Job must be completed in order to compute flow time"

        return self.history.completed_at - self.history.dispatched_at

    @property
    def is_tardy_upon_completion(self) -> bool:
        """
        Returns: True if the job is tardy, False otherwise
        """
        assert self.is_completed, "Job must be completed in order to compute tardiness"

        return self.due_at < self.history.completed_at

    @property
    def earliness_upon_completion(self) -> torch.FloatTensor:
        """
        Returns: The earliness of the job
        """
        assert self.is_completed, "Job must be completed in order to compute earliness"

        result = max(self.due_at - self.history.completed_at, 0.0)

        return result if torch.is_tensor(result) else torch.tensor(result)

    @property
    def remaining_operations_count(self):
        """
        Returns: The number of remaining operations
        """
        result = max(self.step_idx.shape[0] - self.current_step_idx, 0.0)

        return result if torch.is_tensor(result) else torch.tensor(result)

    @property
    def processed_operations_count(self):
        """
        Returns: The number of processed operations
        """
        return self.current_step_idx

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
        return self.__next_processing_time__(self.step_idx, self.processing_times, self.current_step_idx, strategy)

    def slack_upon_moment(self, now: torch.FloatTensor, strategy: ReductionStrategy = ReductionStrategy.mean):
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
        if self.is_completed:
            return self.due_at - self.history.completed_at

        return self.due_at - now

    def is_tardy_at(self, now: torch.FloatTensor):
        """
        Args:
            now: Current time

        Returns: True if the job is tardy at the moment, False otherwise
        """
        return self.time_until_due(now) <= 0

    def is_expected_to_be_tardy_at(self, now: torch.FloatTensor, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Args:
            now: Current time
            strategy: The strategy to use for calculating the slack

        Returns: True if the job is expected to be tardy at the moment, False otherwise
        """
        return self.slack_upon_moment(now, strategy) < 0

    def current_operation_waiting_time_on_work_center(self, now: torch.FloatTensor):
        """
        Args:
            now: Current time

        Returns: The time that the current operation has been waiting for processing on current work center
        """
        return now - self.history.arrived_at_work_center[self.current_step_idx]

    def current_operation_waiting_time_on_machine(self, now: torch.FloatTensor):
        """
        Args:
            now: Current time

        Returns: The time that the current operation has been waiting for processing on current machine
        """
        return now - self.history.arrived_at_machine[self.current_step_idx]

    def wait_time_on_machine(self, step_idx: int):
        """
        Args:
            step_idx: The index of the operation

        Returns: The time that the operation has been waiting for processing on machine
        """
        assert step_idx < self.current_step_idx or self.is_completed,\
              "Operation must be started on machine to compute wait time"

        return self.history.started_at[step_idx] - self.history.arrived_at_machine[step_idx]

    def slack_upon_arrival_on_machine(self, step_idx):
        """
        Args:
            step_idx: The index of the operation

        Returns: The slack upon arrival on machine at specific step
        """
        assert step_idx <= self.current_step_idx, "Operation must be started on machine to compute slack time"

        if step_idx == self.current_step_idx:
            assert self.current_machine_idx >= 0 or self.is_completed, \
                "Job must be processed on machine to compute slack"

        machine_idx = self.history.arrived_machine_idx[step_idx]
        arrival_time = self.history.arrived_at_machine[step_idx]
        remaining_processing_time = self.__remaining_processing_time__(
            self.step_idx, self.processing_times, step_idx, machine_idx
        )

        return self.due_at - arrival_time - remaining_processing_time

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

    def estimated_completion_time(self, now: torch.FloatTensor):
        """
        Args:
            now: Current time
        """

        return now + self.remaining_processing_time(strategy=ReductionStrategy.min)

    # State Update

    def with_event(self, event: Event):
        # Clone tensors to avoid in-place modification
        match event.kind:
            case JobEvent.Kind.dispatch:
                self.current_step_idx = torch.tensor(0, dtype=torch.long)
            case JobEvent.Kind.forward:
                self.current_step_idx = self.current_step_idx + 1
                self.current_machine_idx = torch.tensor(-1, dtype=torch.long)
            case JobEvent.Kind.arrival_on_machine:
                self.current_machine_idx = event.machine_idx
            case _:
                pass

        self.history = self.history.with_event(event, self.step_idx)

        return self

    def with_due_at(self, due_at: torch.FloatTensor):
        self.due_at = torch.FloatTensor([due_at])

        return self

    # Utils

    @classmethod
    @lru_cache
    def __processing_time_on_work_center__(cls,
                                           steps: torch.LongTensor,
                                           processing_times: torch.LongTensor,
                                           step_idx: int, 
                                           strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The processing time of the operation in work center
        """
        if step_idx < 0 or step_idx >= len(steps):
            return torch.tensor(0.0, dtype=torch.float)

        pt = processing_times[step_idx]

        return reduce(pt.float(), strategy)

    @classmethod
    @lru_cache
    def __processing_time_on_machine__(cls,
                                       steps: torch.LongTensor,
                                       processing_times: torch.LongTensor, 
                                       step_idx: int,
                                       machine_idx: int):
        """
        Returns: The processing time of the operation in machine
        """
        if (step_idx < 0 or step_idx >= len(steps) or
                machine_idx < 0 or machine_idx >= processing_times.shape[1]):
            return torch.tensor(0.0, dtype=torch.float)

        return processing_times[step_idx][machine_idx]

    @classmethod
    @lru_cache
    def __next_processing_time__(cls,
                                 steps: torch.LongTensor,
                                 processing_times: torch.LongTensor,
                                 step_idx: int, 
                                 strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The processing time of the next operation
        """
        return cls.__processing_time_on_work_center__(steps, processing_times, step_idx + 1, strategy)

    @classmethod
    @lru_cache
    def __remaining_processing_time__(cls,
                                      steps: torch.LongTensor,
                                      processing_times: torch.LongTensor,
                                      step_idx: int,
                                      machine_idx: int,
                                      strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The remaining processing time of the operation
        """
        result = torch.FloatTensor([0.0])

        if step_idx >= len(steps):
            return result

        if machine_idx >= 0:
            result += cls.__processing_time_on_machine__(steps, processing_times, step_idx, machine_idx)
        else:
            result += cls.__processing_time_on_work_center__(steps, processing_times, step_idx, strategy)

        result += cls.__next_remaining_processing_time__(processing_times, step_idx, strategy)

        return result
    
    @classmethod
    @lru_cache
    def __next_remaining_processing_time__(cls,
                                           processing_times: torch.LongTensor, 
                                           step_idx: int, 
                                           strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The remaining processing time of the next operation
        """
        result = torch.tensor(0.0, dtype=torch.float)
        expected_processing_time = processing_times[max(step_idx + 1, 0):]

        if expected_processing_time.numel() == 0:
            return result

        expected_processing_time = reduce(expected_processing_time.float(), strategy)
        result += expected_processing_time.sum()

        return result
