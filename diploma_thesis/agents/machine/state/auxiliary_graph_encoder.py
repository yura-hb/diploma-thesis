import torch

from agents.base.state import GraphState, Graph

from tensordict.prototype import tensorclass
from .encoder import *


class AuxiliaryGraphEncoder(StateEncoder):

    @tensorclass
    class State(GraphState):
        pass

    def encode(self, parameters: StateEncoder.Input) -> State:
        if parameters.graph is None:
            raise ValueError("Graph is not provided")

        graph = parameters.graph

        job_ids = graph.data[Graph.JOB_INDEX_KEY][:, 0].unique()

        processing_times = []

        for job_id in job_ids:
            processing_times += [parameters.machine.shop_floor.job(job_id).processing_times.view(-1)]

        processing_times = torch.cat(processing_times, dim=0).view(-1, 1)
        graph.data[Graph.OPERATION_KEY].x = processing_times

        return self.State(parameters.graph, batch_size=[])

    @staticmethod
    def from_cli(parameters: dict):
        return AuxiliaryGraphEncoder()
