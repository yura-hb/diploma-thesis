from abc import ABCMeta
from typing import List

import torch
import torch_geometric as pyg

from agents.base import Encoder, GraphEncoder, Graph, State
from agents.machine import MachineInput
from environment import Job


class StateEncoder(Encoder[MachineInput], metaclass=ABCMeta):

    Input = MachineInput
    State = State

    @staticmethod
    def __to_list_of_tensors__(parameters: List) -> List[torch.FloatTensor]:
        return [parameter if torch.is_tensor(parameter) else torch.tensor(parameter) for parameter in parameters]


class GraphStateEncoder(GraphEncoder, metaclass=ABCMeta):

    def __localize__(self, parameters: StateEncoder.Input, graph: Graph):
        job_ids = torch.cat([job.id.view(-1) for job in parameters.machine.queue])

        arriving_jobs = parameters.machine.will_arrive_jobs

        if len(arriving_jobs) > 0:
            job_ids = torch.cat([job_ids, torch.cat([job.id.view(-1) for job in arriving_jobs])])

        return super().__localize_with_job_ids__(graph, job_ids)

    def __post_encode__(self, graph: pyg.data.HeteroData, parameters: StateEncoder.Input) -> pyg.data.HeteroData:
        job_index = graph[Graph.JOB_INDEX_MAP]

        queued_jobs = torch.hstack([job.id for job in parameters.machine.queue])
        is_in_queue = torch.isin(job_index[:, 0].view(-1), queued_jobs, assume_unique=False)

        index = torch.hstack([parameters.machine.work_center_idx, parameters.machine.machine_idx])
        is_target = torch.all(job_index[:, [2, 3]] == index, dim=1)

        graph[Graph.OPERATION_KEY][Graph.TARGET_KEY] = torch.logical_and(is_in_queue.view(-1), is_target.view(-1))

        return graph

    def __estimate_completion_times__(self, job, now):
        lower_completions_times = self.__fill_job_matrix__(job, job.history.finished_at)
        mean_completion_time = lower_completions_times.clone()
        upper_completions_times = lower_completions_times.clone()

        for j in range(job.current_step_idx, len(job.step_idx)):
            if j == 0:
                moment = job.history.dispatched_at
            else:
                moment = lower_completions_times[j - 1]

            lower_completions_times[j] = moment + job.processing_times[j].min()
            mean_completion_time[j] = moment + job.processing_times[j].mean()
            upper_completions_times[j] = moment + job.processing_times[j].max()

        return lower_completions_times - now, mean_completion_time - now, upper_completions_times - now
