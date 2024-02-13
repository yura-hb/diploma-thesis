from dataclasses import dataclass
from typing import Dict

import torch

from agents.base.state import TensorState
from environment import JobReductionStrategy, Job, Machine
from .encoder import StateEncoder


class DEEPMARLMinimumRepetitionStateEncoder(StateEncoder):
    """
    Encoded state is a tensor of dimension (5, 5) where:
    1. First 4 rows contain the following information:
        1. Current operation processing time on machine
        2. Next remaining processing time
        3. Slack upon moment
        4. Average waiting in next queue
        5. Work center index
       Depending on the number of jobs in queue, the state is represented in the following way:
        1. If there are 0 jobs, then the state is a tensor of zeros
        2. If there is 1 job, then the state is a tensor of shape (4, 5) where the first row repeated
        3. If there are more than 1 job, then the information of job minimum values of first 4 criterias are stored
    2. Arriving job info represents information of the job that is about to arrive at the machine
    """

    @dataclass
    class State(TensorState):
        job_idx: torch.LongTensor

    def __init__(self, strategy: JobReductionStrategy = JobReductionStrategy.mean):
        super().__init__()

        self.reduction_strategy = strategy

    def encode(self, parameters: StateEncoder.Input) -> State:
        state, job_idx = self.__make_initial_state(parameters)
        arriving_job_state, _ = self.__make_arriving_job_state__(parameters.machine, parameters.now)

        state = torch.vstack([state, arriving_job_state])

        return self.State(state, job_idx)

    def __make_initial_state(self, parameters: StateEncoder.Input) -> torch.FloatTensor:
        machine = parameters.machine
        queue_size = machine.queue_size
        state = None

        match queue_size:
            case 0:
                return torch.zeros((4, 5)), torch.zeros(4)
            case 1:
                job = machine.queue[0]
                state = self.__make_machine_state_for_single_job__(job, machine, parameters.now)
                state = state.repeat(4, 1)
            case _:
                candidates = torch.vstack([
                    self.__make_machine_state_for_single_job__(job, machine, parameters.now)
                    for job in machine.queue
                ])
                pool = candidates.clone()

                state = []

                for i in range(4):
                    is_no_candidates = pool.numel() == 0
                    store = candidates if is_no_candidates else pool

                    idx = torch.argmin(store[:, i])

                    state += [store[idx]]

                    if not is_no_candidates:
                        pool = pool[torch.arange(pool.size(0)) != idx]

                state = torch.vstack(state)

        return state[:, 1:], state[:, 0]

    def __make_arriving_job_state__(self, machine: Machine, now: float) -> torch.FloatTensor:
        arriving_jobs = machine.arriving_jobs

        average_waiting_time = machine.work_center.average_waiting_time

        if len(arriving_jobs) == 0:
            state = torch.FloatTensor([0, 0, 0, average_waiting_time, 0])

            return state, None

        job = min(arriving_jobs, key=lambda job: job.release_moment_on_machine)

        time_till_arrival = job.release_moment_on_machine - now

        if time_till_arrival < -1:
            self.log_error(f"Arriving job release moment is in the past: {time_till_arrival}")

        time_till_arrival = max(time_till_arrival, 0)

        state = [
            job.next_operation_processing_time(self.reduction_strategy),
            job.next_remaining_processing_time(self.reduction_strategy),
            job.slack_upon_moment(now, self.reduction_strategy),
            average_waiting_time,
            time_till_arrival
        ]

        state = self.__to_list_of_tensors__(state)

        return torch.hstack(state), job.id

    def __make_machine_state_for_single_job__(self, job: Job, machine: Machine, now) -> torch.FloatTensor:
        state = [
            job.id,
            job.current_operation_processing_time_on_machine,
            job.next_remaining_processing_time(self.reduction_strategy),
            job.slack_upon_moment(now, self.reduction_strategy),
            machine.shop_floor.average_waiting_in_next_queue(job),
            machine.work_center_idx
        ]

        state = self.__to_list_of_tensors__(state)

        return torch.hstack(state)

    @staticmethod
    def from_cli(parameters: Dict):
        return DEEPMARLMinimumRepetitionStateEncoder()
