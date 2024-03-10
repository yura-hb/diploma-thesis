
from typing import Dict

from tensordict.prototype import tensorclass

from agents.base.state import GraphState
from .encoder import *


class AuxiliaryGraphEncoder(GraphStateEncoder):

    @tensorclass
    class State(GraphState):
        pass

    def __encode__(self, parameters: StateEncoder.Input) -> State:
        if parameters.graph is None:
            raise ValueError("Graph is not provided")

        graph = parameters.graph

        job_ids = graph.data[Graph.JOB_INDEX_MAP][:, 0].unique()

        processing_times = []

        for job_id in job_ids:
            processing_times += [parameters.machine.shop_floor.job(job_id).processing_times.view(-1)]

        processing_times = torch.cat(processing_times, dim=0).view(-1, 1)
        graph.data[Graph.OPERATION_KEY].x = processing_times

        return self.State(graph, batch_size=[])

    @classmethod
    def from_cli(cls, parameters: dict):
        return AuxiliaryGraphEncoder(**cls.base_parameters_from_cli(parameters))
