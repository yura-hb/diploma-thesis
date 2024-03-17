from abc import abstractmethod
from typing import TypeVar, Generic

from torch_geometric.transforms import ToUndirected

from utils import Loggable
from .state import State, Graph

Input = TypeVar('Input')


class Encoder(Loggable, Generic[Input]):

    @abstractmethod
    def encode(self, parameters: Input) -> State:
        pass


class GraphEncoder(Encoder, Generic[Input]):

    def __init__(self, is_homogeneous: False, is_undirected: False, is_local: False):
        super().__init__()

        self.is_homogeneous = is_homogeneous
        self.is_undirected = is_undirected
        self.is_local = is_local
        self.to_undirected = ToUndirected()

    def encode(self, parameters: Input) -> State:
        parameters.graph = parameters.graph.to_pyg_graph()

        if self.is_local:
            parameters.graph = self.__localize__(parameters, parameters.graph)

        result = self.__encode__(parameters)
        result.graph = self.__post_encode__(result.graph, parameters)

        if self.is_homogeneous:
            result.graph = result.graph.to_homogeneous(node_attrs=['x'])

        del result.graph[Graph.JOB_INDEX_MAP]
        del result.graph[Graph.MACHINE_INDEX_KEY]

        if self.is_undirected:
            result.graph = self.to_undirected(result.graph)

        result.graph = Graph.from_pyg_graph(result.graph)

        return result

    @abstractmethod
    def __encode__(self, parameters: Input) -> State:
        pass

    def __post_encode__(self, graph: Graph, parameters: Input) -> Graph:
        return graph

    @abstractmethod
    def __localize__(self, parameters: Input, graph: Graph):
        pass

    @classmethod
    def base_parameters_from_cli(cls, parameters):
        return dict(
            is_homogeneous=parameters.get('is_homogeneous', False),
            is_undirected=parameters.get('is_undirected', False),
            is_local=parameters.get('is_local', False)
        )
