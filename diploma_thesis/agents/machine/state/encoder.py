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
        job_ids = torch.cat([job_ids, parameters.machine.history.processed_job_ids])

        return super().__localize_with_job_ids__(graph, job_ids)

    def __post_encode__(self, graph: pyg.data.HeteroData, parameters: StateEncoder.Input) -> pyg.data.HeteroData:
        index = torch.hstack([parameters.machine.work_center_idx, parameters.machine.machine_idx])
        is_target = torch.all(graph[Graph.JOB_INDEX_MAP][:, [2, 3]] == index, dim=1)

        graph[Graph.OPERATION_KEY][Graph.TARGET_KEY] = is_target.view(-1)

        return graph

