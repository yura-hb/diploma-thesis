from abc import ABCMeta
from typing import List, TypeVar

import torch

from agents.base.encoder import Input
from environment import Job

from agents.base import Encoder, GraphEncoder, Graph
from agents.machine import MachineInput

State = TypeVar('State')


class StateEncoder(Encoder[MachineInput, State], metaclass=ABCMeta):

    Input = MachineInput

    @staticmethod
    def __to_list_of_tensors__(parameters: List) -> List[torch.FloatTensor]:
        return [parameter if torch.is_tensor(parameter) else torch.tensor(parameter) for parameter in parameters]


class GraphStateEncoder(GraphEncoder, metaclass=ABCMeta):

    def __fill_job_matrix__(self, job: Job, tensor):
        result = torch.zeros_like(job.processing_times)

        idx = job.current_step_idx + (1 if job.is_completed else 0)

        result[
            torch.arange(idx, dtype=torch.long),
            job.history.arrived_machine_idx[:idx].int()
        ] = tensor[:idx].float()

        return result

    def __localize__(self, parameters: StateEncoder.Input, graph: Graph):
        job_ids = graph[Graph.JOB_INDEX_MAP][:, 0]
        machine_job_ids = torch.cat([job.id.view(-1) for job in parameters.machine.queue])
        machine_job_ids = torch.cat([machine_job_ids, parameters.machine.history.processed_job_ids])
        mask = torch.isin(job_ids, machine_job_ids, assume_unique=True)
        idx = torch.nonzero(mask).view(-1)

        if idx.numel() == 0:
            return graph

        graph = graph.subgraph({Graph.OPERATION_KEY: idx})
        graph[Graph.JOB_INDEX_MAP] = graph[Graph.JOB_INDEX_MAP][mask]

        return graph

    # TODO: - Implement
    def __post_encode__(self, graph: Graph, parameters: StateEncoder.Input) -> Graph:
        index = torch.tensor(parameters.machine.work_center_idx, parameters.machine.machine_idx)
        is_target = graph[Graph.JOB_INDEX_MAP][[2, 3], :] == index


        return graph

