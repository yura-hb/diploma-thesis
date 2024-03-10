from abc import abstractmethod
from typing import TypeVar, Generic

from torch_geometric.data import Batch
from torch_geometric.transforms import ToUndirected

from utils import Loggable
from .state import GraphState, Graph

Input = TypeVar('Input')
State = TypeVar('State')


class Encoder(Loggable, Generic[Input, State]):

    @abstractmethod
    def encode(self, parameters: Input) -> State:
        pass


class GraphEncoder(Encoder, Generic[Input, State]):

    def __init__(self, is_homogeneous: False, is_undirected: False, is_local: False):
        super().__init__()

        self.is_homogeneous = is_homogeneous
        self.is_undirected = is_undirected
        self.is_local = is_local
        self.to_undirected = ToUndirected()

    def encode(self, parameters: Input) -> State:
        if self.is_local:
            parameters.graph = self.__localize__(parameters, parameters.graph)

        result = self.__encode__(parameters)

        if self.is_homogeneous:
            result.graph.data = result.graph.data.to_homogeneous(node_attrs=['x'])

        del result.graph.data[Graph.JOB_INDEX_MAP]
        del result.graph.data[Graph.MACHINE_INDEX_KEY]

        if self.is_undirected:
            result.graph.data = self.to_undirected(result.graph.data)

        result.graph.data = Batch.from_data_list([result.graph.data])

        return result

    @abstractmethod
    def __encode__(self, parameters: Input) -> State | GraphState:
        pass

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
