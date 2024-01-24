
import pytest
import torch

from diploma_thesis.environment.job import Job

from dataclasses import dataclass, field
from typing import Tuple, Dict, List


@dataclass
class Expectation:
    id: int = 0
    created_at: int = 0
    due_at: int = 800
    completed_at: int = 1000
    arrives_at: List[int] = field(default_factory=list)
    step_idx: torch.LongTensor = field(default_factory=torch.LongTensor)
    machine_sequence: torch.LongTensor = field(default_factory=torch.LongTensor)
    processing_times: torch.LongTensor = field(default_factory=torch.LongTensor)
    moments: Dict[Job.ReductionStrategy, Tuple[float, float]] = field(default_factory=lambda: (0, 0))
    # Expected processing times returned from `current_operation_processing_time_on_machine`
    machine_processing_times: torch.LongTensor = field(default_factory=torch.LongTensor)
    # Expected processing times returned from `current_operation_processing_time_in_work_center`
    work_center_processing_times: List[Dict[Job.ReductionStrategy, float]] = field(default_factory=lambda: list)
    # Remaining processing count
    remaining_processing_time: List[Dict[Job.ReductionStrategy, float]] = field(default_factory=lambda: list)
    # Expected work center idx returned from `next_work_center_idx`
    next_work_center_idx: int = None
    # Expected next operation processing time returned from `current_operation_processing_time_on_machine`
    next_operation_processing_time: List[Dict[Job.ReductionStrategy, float]] = field(default_factory=lambda: list)
    # Expected due times returned from `time_until_due(arrived_at)`
    due_times_from_arrival: List[int] = field(default_factory=lambda: list)
    # Expected waiting time returned from `current_operation_waiting_time` at moment of tuple[0] with expected value of
    # tuple[1]
    current_operation_waiting_time: List[Tuple[int, int]] = field(default_factory=lambda: list)
    # Slack upon arrival
    slack_upon_arrival: List[Dict[Job.ReductionStrategy, float]] = field(default_factory=lambda: list)

    def initial_job(self):
        return Job(
            id=self.id,
            step_idx=self.step_idx,
            processing_times=self.processing_times,
            created_at=self.created_at
        )


@pytest.fixture
def simple_job():
    expectation = Expectation(
        id=1234,
        step_idx=torch.LongTensor([0, 2, 1]),
        created_at=0,
        arrives_at=[0, 100, 500],
        machine_sequence=torch.LongTensor([0, 2, 1]),
        processing_times=torch.LongTensor([[10, 20, 30], [40, 50, 60], [70, 80, 90]]),
        moments={
            Job.ReductionStrategy.mean: (50, 30),
            Job.ReductionStrategy.min: (40, 30),
            Job.ReductionStrategy.max: (60, 30)
        },
        machine_processing_times=torch.LongTensor([10, 60, 80]),
        work_center_processing_times=[
            {
                Job.ReductionStrategy.mean: 20,
                Job.ReductionStrategy.min: 10,
                Job.ReductionStrategy.max: 30
            },
            {
                Job.ReductionStrategy.mean: 50,
                Job.ReductionStrategy.min: 40,
                Job.ReductionStrategy.max: 60
            },
            {
                Job.ReductionStrategy.mean: 80,
                Job.ReductionStrategy.min: 70,
                Job.ReductionStrategy.max: 90
            }
        ],
        remaining_processing_time=[
            # Time -1
            {
                Job.ReductionStrategy.mean: 150,
                Job.ReductionStrategy.min: 120,
                Job.ReductionStrategy.max: 180
            },
            {
                Job.ReductionStrategy.mean: 150,
                Job.ReductionStrategy.min: 120,
                Job.ReductionStrategy.max: 180
            },
            {
                Job.ReductionStrategy.mean: 130,
                Job.ReductionStrategy.min: 110,
                Job.ReductionStrategy.max: 150
            },
            {
                Job.ReductionStrategy.mean: 80,
                Job.ReductionStrategy.min: 70,
                Job.ReductionStrategy.max: 90
            },
            # Job Completed
            {
                Job.ReductionStrategy.mean: 0,
                Job.ReductionStrategy.min: 0,
                Job.ReductionStrategy.max: 0
            },
        ],
        next_operation_processing_time=[
            # Time -1
            {
                Job.ReductionStrategy.mean: 20,
                Job.ReductionStrategy.min: 10,
                Job.ReductionStrategy.max: 30
            },
            {
                Job.ReductionStrategy.mean: 50,
                Job.ReductionStrategy.min: 40,
                Job.ReductionStrategy.max: 60
            },
            {
                Job.ReductionStrategy.mean: 80,
                Job.ReductionStrategy.min: 70,
                Job.ReductionStrategy.max: 90
            },
            {
                Job.ReductionStrategy.mean: 0,
                Job.ReductionStrategy.min: 0,
                Job.ReductionStrategy.max: 0
            }
        ],
        next_work_center_idx=[0, 2, 1, None],
        due_times_from_arrival=[800, 700, 300],
        current_operation_waiting_time=[
            (50, 50),
            (200, 100),
            (700, 200)
        ],
        slack_upon_arrival=[
            {
                Job.ReductionStrategy.mean: 650,
                Job.ReductionStrategy.min: 680,
                Job.ReductionStrategy.max: 620
            },
            {
                Job.ReductionStrategy.mean: 570,
                Job.ReductionStrategy.min: 590,
                Job.ReductionStrategy.max: 550
            },
            {
                Job.ReductionStrategy.mean: 220,
                Job.ReductionStrategy.min: 230,
                Job.ReductionStrategy.max: 210
            },
        ]
    )

    job = expectation.initial_job()

    return job, expectation


def test_job(simple_job):
    job, expectation = simple_job
    current_step = -1

    job.due_at = expectation.due_at

    assert job.current_machine_idx == -1

    while True:
        history_index = current_step + 1

        assert job.id == expectation.id
        assert job.step_idx.tolist() == expectation.step_idx.tolist()
        assert job.processing_times.tolist() == expectation.processing_times.tolist()
        assert job.current_step_idx == current_step
        assert job.created_at == expectation.created_at
        assert not job.is_completed
        assert job.next_work_center_idx == expectation.next_work_center_idx[history_index]

        for strategy, moments in expectation.moments.items():
            assert job.processing_time_moments(strategy) == moments

        for strategy, value in expectation.remaining_processing_time[history_index].items():
            assert job.remaining_processing_time(strategy) == value

        for strategy, value in expectation.next_operation_processing_time[history_index].items():
            assert job.next_operation_processing_time(strategy) == value

        current_step += 1

        if job.next_work_center_idx is None:
            job.with_completion_time(expectation.completed_at)
            break

        job.with_next_step()

        for strategy, value in expectation.slack_upon_arrival[current_step].items():
            job.with_arrival(expectation.arrives_at[current_step], strategy)

            assert job.slack_upon_arrival == value

        job.with_arrival(expectation.arrives_at[current_step], Job.ReductionStrategy.mean)
        job.with_assigned_machine(expectation.machine_sequence[current_step])

        assert job.current_machine_idx == (expectation.machine_sequence[current_step])
        assert job.current_operation_processing_time_on_machine == expectation.machine_processing_times[current_step]
        assert job.remaining_operations_count == len(expectation.step_idx) - current_step
        assert job.time_until_due(expectation.arrives_at[current_step]) == expectation.due_times_from_arrival[current_step]
        assert (job.current_operation_waiting_time(expectation.current_operation_waiting_time[current_step][0]) ==
                expectation.current_operation_waiting_time[current_step][1])

        for strategy, value in expectation.work_center_processing_times[current_step].items():
            assert job.current_operation_processing_time_in_work_center(strategy) == value
            assert job.operation_processing_time_in_work_center(job.step_idx[job.current_step_idx], strategy) == value

    assert job.is_completed
    assert job.completed_at == expectation.completed_at

    for strategy, value in expectation.remaining_processing_time[current_step + 1].items():
        assert job.remaining_processing_time(strategy) == value
