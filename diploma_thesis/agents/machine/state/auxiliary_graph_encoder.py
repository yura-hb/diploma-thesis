from tensordict.prototype import tensorclass

from agents.base.state import GraphState, Graph
from .encoder import *


class AuxiliaryGraphEncoder(GraphStateEncoder):

    def __init__(self, is_homogeneous: bool):
        super().__init__()

        self.is_homogeneous = is_homogeneous

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

        if self.is_homogeneous:
            graph.data = graph.data.to_homogeneous(node_attrs=['x'])

        del graph.data[Graph.JOB_INDEX_MAP]
        del graph.data[Graph.MACHINE_INDEX_KEY]

        return self.State(graph, batch_size=[])

    @staticmethod
    def from_cli(parameters: dict):
        return AuxiliaryGraphEncoder(parameters.get('is_homogeneous', False))
