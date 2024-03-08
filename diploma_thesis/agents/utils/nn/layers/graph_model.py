import torch

from .layer import *
from typing import Dict, List, Tuple
from dataclasses import dataclass
from .graph_layer import GraphLayer

from enum import StrEnum

import torch_geometric as pyg


class OutputType(StrEnum):
    NODE = 'node'
    EDGE = 'edge'
    GLOBAL = 'global'


class GraphModel(Layer):

    @dataclass
    class Configuration:

        @dataclass
        class Output:
            node_key: str | None
            kind: OutputType

            @staticmethod
            def from_cli(parameters: Dict) -> 'GraphModel.Configuration.Output':
                return GraphModel.Configuration.Output(
                    node_key=parameters.get('node_key'),
                    kind=OutputType(parameters['kind'])
                )

        layers: List[Tuple[Layer, str | None]]
        hetero_aggregation: str = 'mean'
        output: Output = None

        @staticmethod
        def from_cli(parameters: Dict) -> 'GraphModel.Configuration':
            from .cli import from_cli

            return GraphModel.Configuration(
                layers=[
                    (from_cli(layer), layer.get('parameters', {}).get('signature'))
                    for layer in parameters['layers']
                ],
                hetero_aggregation=parameters.get('hetero_aggregation', 'mean'),
                output=GraphModel.Configuration.Output.from_cli(parameters.get('output', {}))
            )

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.is_configured = False
        self.model: pyg.nn.Sequential = None
        self.configuration = configuration

        self.__build__()

    def forward(self, batch: pyg.data.Batch) -> torch.Tensor:
        self.__configure_if_needed__(batch)

        if isinstance(batch, pyg.data.HeteroData):
            # Result is the dict for each edge_type
            hidden = self.model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)

            return hidden

        hidden = self.model(batch.x, batch.edge_index, batch.batch)

        return hidden

    def __build__(self):
        def encode_layer(layer, signature):
            if isinstance(layer, GraphLayer) and layer.signature is not None:
                return layer, layer.signature

            if signature is not None:
                return layer, signature

            return layer

        layers = [encode_layer(layer, signature) for layer, signature in self.configuration.layers]

        self.model = pyg.nn.Sequential('x, edge_index, batch', layers)

    def __configure_if_needed__(self, graph: pyg.data.Data | pyg.data.HeteroData):
        if not self.is_configured:
            if isinstance(graph, pyg.data.HeteroData):
                self.model = pyg.nn.to_hetero(self.model, graph.metadata(), aggr=self.configuration.hetero_aggregation)

            self.is_configured = True

    def __process_heterogeneous_output__(self, graph: pyg.data.Data, output: torch.Tensor):
        match self.configuration.output.kind:
            case OutputType.EDGE:
                return output
            case OutputType.GLOBAL:
                pass
            case OutputType.NODE:
                pass


    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return cls(cls.Configuration.from_cli(parameters))

