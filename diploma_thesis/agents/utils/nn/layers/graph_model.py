from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch_geometric as pyg

from agents.base import Graph
from .graph_layer import GraphLayer
from .layer import *


class GraphModel(Layer):

    @dataclass
    class Configuration:
        layers: List[Tuple[Layer, str | None]]

        hetero_aggregation: str = 'mean'
        hetero_aggregation_key: str = 'operation'

        @staticmethod
        def from_cli(parameters: Dict) -> 'GraphModel.Configuration':
            from .cli import from_cli

            return GraphModel.Configuration(
                layers=[
                    (from_cli(layer), layer.get('parameters', {}).get('signature'))
                    for layer in parameters['layers']
                ],
                hetero_aggregation=parameters.get('hetero_aggregation', 'mean'),
                hetero_aggregation_key=parameters.get('hetero_aggregation_key', 'operation')
            )

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.is_configured = False
        self.model: pyg.nn.Sequential = None
        self.configuration = configuration

        self.__build__()

    def forward(self, graph: Graph | pyg.data.Batch) -> torch.Tensor:
        batch: pyg.data.Batch = None

        if isinstance(graph, Graph):
            batch = graph.to_pyg_batch()
        else:
            batch = graph

        self.__configure_if_needed__(batch)

        if isinstance(batch, pyg.data.HeteroData):
            # Result is the dict for each edge_type
            hidden = self.model(batch.x_dict, batch.edge_index_dict, batch.batch_dict)

            return self.__process_heterogeneous_output__(hidden)

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

    def __process_heterogeneous_output__(self, output: Dict[Tuple[str, str, str], torch.Tensor]) -> torch.Tensor:
        result = []

        for key, embeddings in output.items():
            if key[0] == self.configuration.hetero_aggregation_key:
                result += [embeddings]

        # TODO: Use aggregation
        return torch.stack(result).mean(dim=0)

    @classmethod
    def from_cli(cls, parameters: dict) -> 'Layer':
        return cls(cls.Configuration.from_cli(parameters))

