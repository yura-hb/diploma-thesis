from abc import ABCMeta, abstractmethod
from typing import List, TypeVar

import torch

from torch_geometric.data import Batch

from agents.base import Encoder, GraphEncoder, Graph
from agents.base.encoder import Input
from agents.machine import MachineInput

State = TypeVar('State')


class StateEncoder(Encoder[MachineInput, State], metaclass=ABCMeta):

    Input = MachineInput

    @staticmethod
    def __to_list_of_tensors__(parameters: List) -> List[torch.FloatTensor]:
        return [parameter if torch.is_tensor(parameter) else torch.tensor(parameter) for parameter in parameters]


class GraphStateEncoder(GraphEncoder, metaclass=ABCMeta):

    def __localize__(self, parameters: StateEncoder.Input, graph: Graph):
        job_ids = graph.data[Graph.JOB_INDEX_MAP][:, 0]
        queued_job_ids = torch.cat([job.id.view(-1) for job in parameters.machine.queue])
        mask = torch.isin(job_ids, queued_job_ids, assume_unique=True)
        idx = torch.nonzero(mask).view(-1)

        graph.data = graph.data.subgraph({Graph.OPERATION_KEY: idx})
        graph.data[Graph.JOB_INDEX_MAP] = graph.data[Graph.JOB_INDEX_MAP][mask]

        return graph
