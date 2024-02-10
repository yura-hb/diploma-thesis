
import torch

from dataclasses import dataclass, field
from enum import Enum, auto


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
        created_at: torch.FloatTensor = 0
        # The time, when job was pushed into system
        dispatched_at: torch.FloatTensor = 0
        # The creation time of the job
        completed_at: torch.FloatTensor = 0
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
            self.started_at = torch.zeros_like(step_idx)
            self.finished_at = torch.zeros_like(step_idx)
            self.arrived_at_work_center = torch.zeros_like(step_idx)
            self.arrived_at_machine = torch.zeros_like(step_idx)
            self.arrived_machine_idx = torch.zeros_like(step_idx)

        def with_event(self, event: JobEvent, step_idx: torch.LongTensor):
            def get_work_center_idx():
                return torch.argwhere(step_idx == event.work_center_idx).item()

            match event.kind:
                case JobEvent.Kind.creation:
                    self.created_at = event.moment
                case JobEvent.Kind.dispatch:
                    self.dispatched_at = event.moment
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
                    self.completed_at = event.moment
                case _:
                    pass

            return self

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
    # The priority of the Job
    priority: float = 1.0
    # The due time of the job, i.e. deadline
    due_at: torch.FloatTensor = 0
    # History of the job
    history: History = None

    def __post_init__(self):
        self.history = self.History()
        self.history.configure(self.step_idx)

        assert 0.0 <= self.priority <= 1.0, "Priority must be in range [0, 1]"

    def processing_time_moments(self, reduction_strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The expectation and variance of processing times
        """
        processing_times = reduce(self.processing_times.float(), reduction_strategy)

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

        return reduce(processing_times, strategy)

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

        return reduce(processing_times, strategy)

    def remaining_processing_time(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The total processing time of the remaining operations
        """
        # Since we don't know, to which machine inside work-center the job will be dispatched next, we
        # approximate it with the average
        if self.is_completed:
            return 0

        expected_processing_time = self.processing_times[max(self.current_step_idx, 0):]
        expected_processing_time = reduce(expected_processing_time.float(), strategy)

        return expected_processing_time.sum()

    def total_processing_time(self, strategy: ReductionStrategy = ReductionStrategy.mean):
        """
        Returns: The total processing time of the job
        """
        return reduce(self.processing_times.float(), strategy).sum()

    @property
    def is_completed(self):
        return self.history.completed_at > 0

    @property
    def is_dispatched(self):
        return self.current_step_idx >= 0

    @property
    def tardiness(self) -> torch.FloatTensor:
        """
        Returns: The tardiness of the job
        """
        assert self.is_completed, "Job must be completed in order to compute tardiness"

        return torch.tensor([max(self.history.completed_at - self.due_at, 0.0)])

    @property
    def flow_time(self) -> torch.FloatTensor:
        """
        Returns: The flow time of the job
        """
        assert self.is_completed, "Job must be completed in order to compute flow time"

        return self.history.completed_at - self.history.dispatched_at

    @property
    def is_tardy(self) -> bool:
        """
        Returns: True if the job is tardy, False otherwise
        """
        assert self.is_completed, "Job must be completed in order to compute tardiness"

        return self.due_at < self.history.completed_at

    @property
    def earliness(self) -> torch.FloatTensor:
        """
        Returns: The earliness of the job
        """
        assert self.is_completed, "Job must be completed in order to compute earliness"

        return torch.tensor([max(self.due_at - self.history.completed_at, 0)])

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

        return reduce(pt.float(), strategy)

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
        return self.due_at - now

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

    def with_event(self, event: Event):
        match event.kind:
            case JobEvent.Kind.dispatch:
                self.current_step_idx = 0
            case JobEvent.Kind.forward:
                self.current_step_idx += 1
            case JobEvent.Kind.arrival_on_machine:
                self.current_machine_idx = event.machine_idx
            case _:
                pass

        self.history = self.history.with_event(event, self.step_idx)

        return self

    def with_due_at(self, due_at):
        self.due_at = due_at

        return self
