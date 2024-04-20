import torch

from abc import abstractmethod
from typing import TypeVar, Generic

import torch_geometric as pyg

from torch_geometric.transforms import ToUndirected, RemoveIsolatedNodes, RemoveDuplicatedEdges

from utils import Loggable
from .state import State, Graph

from environment import Job

Input = TypeVar('Input')


class Encoder(Loggable, Generic[Input]):

    @abstractmethod
    def encode(self, parameters: Input) -> State:
        pass


class GraphEncoder(Encoder, Generic[Input]):

    def __init__(self, is_homogeneous: False, is_undirected: False, is_local: False, append_target_mask: bool):
        super().__init__()

        self.is_homogeneous = is_homogeneous
        self.is_undirected = is_undirected
        self.is_local = is_local
        self.append_target_mask = append_target_mask
        self.to_undirected = ToUndirected()
        self.remove_isolated_nodes = RemoveIsolatedNodes()
        self.remove_duplicated_edges = RemoveDuplicatedEdges()

    def encode(self, parameters: Input) -> State:
        parameters.graph = parameters.graph.to_pyg_graph()

        if self.is_local:
            parameters.graph = self.__localize__(parameters, parameters.graph)

        result = self.__encode__(parameters)
        result.graph = self.__post_encode__(result.graph, parameters)

        if self.is_homogeneous:
            result.graph = result.graph.to_homogeneous(node_attrs=[Graph.X, Graph.TARGET_KEY])
            result.info[Graph.OPERATION_KEY] = torch.tensor(
                [i for i, name in enumerate(result.graph._node_type_names) if name == Graph.OPERATION_KEY][0]
            )

        if self.append_target_mask:
            store = result.graph if self.is_homogeneous else result.graph[Graph.OPERATION_KEY]
            store[Graph.X] = torch.cat([store[Graph.X], store[Graph.TARGET_KEY].view(-1, 1)], dim=1)

        del result.graph[Graph.MACHINE_INDEX_KEY]

        if self.is_undirected:
            result.graph = self.to_undirected(result.graph)

        result.graph = self.remove_isolated_nodes(result.graph)
        result.graph = self.remove_duplicated_edges(result.graph)

        result.graph = Graph.from_pyg_graph(result.graph)

        return result

    @abstractmethod
    def __encode__(self, parameters: Input) -> State:
        pass

    @abstractmethod
    def __localize__(self, graph: Graph, parameters: Input):
        pass

    def __post_encode__(self, graph: pyg.data.HeteroData, parameters: Input) -> pyg.data.HeteroData:
        return graph

    @staticmethod
    def __fill_job_matrix__(job: Job, tensor, initial_matrix=None, until_current_step: bool = True):
        result = initial_matrix if initial_matrix is not None else torch.zeros_like(job.processing_times)

        idx = job.current_step_idx + (1 if job.is_completed else 0)

        result[
            torch.arange(idx, dtype=torch.long), job.history.arrived_machine_idx[:idx].int()
        ] = tensor[:idx].float()

        if not job.is_completed:
            result[idx, job.current_machine_idx] = tensor[idx].float()

            if not until_current_step:
                result[idx + 1:, :] = tensor[idx + 1:].unsqueeze(-1).float()

        return result

    @staticmethod
    def __localize_with_job_ids__(graph: Graph, job_ids: torch.Tensor):
        job_ids_ = graph[Graph.JOB_INDEX_MAP][:, 0]
        mask = torch.isin(job_ids_, job_ids, assume_unique=False)
        idx = torch.nonzero(mask).view(-1)

        if idx.numel() == 0:
            return graph

        graph = graph.subgraph({Graph.OPERATION_KEY: idx})
        graph[Graph.JOB_INDEX_MAP] = graph[Graph.JOB_INDEX_MAP][mask]

        return graph

    @classmethod
    def base_parameters_from_cli(cls, parameters):
        return dict(
            is_homogeneous=parameters.get('is_homogeneous', False),
            is_undirected=parameters.get('is_undirected', False),
            is_local=parameters.get('is_local', False),
            append_target_mask=parameters.get('append_target_mask', False)
        )
