from abc import ABCMeta
from typing import List, TypeVar

import torch

from agents.base import Encoder, GraphEncoder, Graph
from agents.machine import MachineInput

State = TypeVar('State')


class StateEncoder(Encoder[MachineInput, State], metaclass=ABCMeta):

    Input = MachineInput

    @staticmethod
    def __to_list_of_tensors__(parameters: List) -> List[torch.FloatTensor]:
        return [parameter if torch.is_tensor(parameter) else torch.tensor(parameter) for parameter in parameters]


class GraphStateEncoder(GraphEncoder, metaclass=ABCMeta):

    def __fill_job_matrix__(self, job, tensor):
        result = torch.zeros_like(job.processing_times)

        result[
            torch.arange(job.current_step_idx, dtype=torch.long),
            job.history.arrived_machine_idx[:job.current_step_idx].int()
        ] = tensor[:job.current_step_idx].float()

        return result

    def __localize__(self, parameters: StateEncoder.Input, graph: Graph):
        job_ids = graph.data[Graph.JOB_INDEX_MAP][:, 0]
        machine_job_ids = torch.cat([job.id.view(-1) for job in parameters.machine.queue])
        machine_job_ids = torch.cat([machine_job_ids, parameters.machine.history.processed_job_ids])
        mask = torch.isin(job_ids, machine_job_ids, assume_unique=True)
        idx = torch.nonzero(mask).view(-1)

        graph.data = graph.data.subgraph({Graph.OPERATION_KEY: idx})
        graph.data[Graph.JOB_INDEX_MAP] = graph.data[Graph.JOB_INDEX_MAP][mask]

        return graph
