from dataclasses import dataclass
from typing import Dict
from environment import JobReductionStrategy
from functools import reduce

import torch

from .encoder import StateEncoder


class DEEPMARLIndirectStateEncoder(StateEncoder):

    @dataclass
    class State:
        state: torch.FloatTensor

    def __init__(self, strategy: JobReductionStrategy = JobReductionStrategy.mean):
        super().__init__()

        self.reduction_strategy = strategy

    def encode(self, parameters: StateEncoder.Input) -> State:
        state = [
            self.__make_job_number_state__(parameters),
            self.__make_system_state__(parameters),
            self.__make_arriving_job_info__(parameters),
            self.__make_processing_time_info__(parameters),
            self.__make_average_waiting_in_next_queue_info__(parameters),
            self.__make_time_till_due_state__(parameters)
        ]

        state = reduce(lambda x, y: x + y, state, [])
        state = self.__to_list_of_tensors__(state)
        state = torch.hstack(state).reshape(-1)
        state = torch.nan_to_num(state, nan=0.0, posinf=1, neginf=-1)

        return self.State(state)

    def __make_job_number_state__(self, parameters: StateEncoder.Input):
        state = [
            len(parameters.machine.shop_floor.in_system_jobs),
            parameters.machine.queue_size,
        ]

        return state

    def __make_system_state__(self, parameters: StateEncoder.Input):
        available_times = torch.FloatTensor([
            machine.time_till_available for machine in parameters.machine.shop_floor.machines
        ])

        available_time_ratio = parameters.machine.time_till_available / available_times.sum()

        processing_time_heterogeneity = torch.FloatTensor([
            machine.cumulative_processing_time for machine in parameters.machine.shop_floor.machines
        ])

        processing_time_heterogeneity = processing_time_heterogeneity.std() / processing_time_heterogeneity.mean()

        state = [
            parameters.machine.shop_floor.completion_rate,
            parameters.machine.shop_floor.tardy_rate(parameters.now),
            parameters.machine.shop_floor.expected_tardy_rate(parameters.now, self.reduction_strategy),
            available_time_ratio,
            processing_time_heterogeneity
        ]

        return state

    def __make_arriving_job_info__(self, parameters: StateEncoder.Input):
        arriving_jobs = parameters.machine.arriving_jobs

        time_till_arrival = torch.FloatTensor([
            max(job.release_moment_on_machine - parameters.now, 0)
            for job in arriving_jobs
        ])

        slack = torch.FloatTensor([
            job.slack_upon_moment(parameters.now, self.reduction_strategy)
            for job in arriving_jobs
        ])

        state = [
            len(arriving_jobs),
            time_till_arrival.mean(),
            slack.mean()
        ]

        return state

    def __make_processing_time_info__(self, parameters: StateEncoder.Input):
        processing_times = torch.FloatTensor([
            job.current_operation_processing_time_on_machine for job in parameters.machine.queue
        ])
        next_remaining_processing_time = torch.FloatTensor([
            job.next_remaining_processing_time(self.reduction_strategy) for job in parameters.machine.queue
        ])

        state = []

        state += self.__moments__(processing_times)
        state += self.__moments__(next_remaining_processing_time)

        return state

    def __make_average_waiting_in_next_queue_info__(self, parameters: StateEncoder.Input):
        avlm = torch.FloatTensor([
            parameters.machine.shop_floor.average_waiting_in_next_queue(job) for job in parameters.machine.queue
        ])

        state = self.__moments__(avlm)

        return state

    def __make_time_till_due_state__(self, parameters: StateEncoder.Input):
        time_till_due = torch.FloatTensor([
            job.time_until_due(parameters.now) for job in parameters.machine.queue
        ])

        state = self.__moments__(time_till_due)

        return state

    def __moments__(self, values: torch.FloatTensor):
        mean = values.mean()

        return [
            values.sum(),
            mean,
            values.min(),
            values.std() / mean
        ]

    @staticmethod
    def from_cli(parameters: Dict):
        return DEEPMARLIndirectStateEncoder()
