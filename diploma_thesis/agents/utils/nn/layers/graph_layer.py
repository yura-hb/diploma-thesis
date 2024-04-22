from typing import Dict

import torch.nn
import torch_geometric as pyg

from agents.base.state import Graph
from .layer import *


class GraphLayer(Layer):
    pass


class BaseWrapper(GraphLayer):

    def __init__(self, configuration):
        self._signature = configuration.get('signature')

        if 'signature' in configuration:
            del configuration['signature']

        super().__init__(self._signature)

    @property
    def signature(self):
        return self._signature

    @classmethod
    def from_cli(cls, parameters: Dict):
        return cls(parameters)


class GraphModuleWrapper(BaseWrapper):

    def __init__(self, layer, configuration):
        super().__init__(configuration)

        self.model = layer(**configuration)

    @property
    def signature(self):
        return self._signature or 'x, edge_index -> x'

    def forward(self, x, edge_index):
        return self.model(x, edge_index)


class GIN(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.GIN, configuration)


class SAGEConv(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.GraphSAGE, configuration)


class GATConv(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.GAT, configuration)


class GCNConv(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.GCN, configuration)


class DeepGCNConv(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.DeepGCNLayer, configuration)


class GraphInstanceNorm(GraphModuleWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.InstanceNorm, configuration)

    @property
    def signature(self):
        return self._signature or 'x, batch -> x'


class GraphFunctionWrapper(BaseWrapper):

    def __init__(self, fn, configuration):
        super().__init__(configuration)

        self.fn = fn

    @property
    def signature(self):
        return self._signature or 'x, batch -> x'

    def forward(self, x, batch):
        if isinstance(batch, tuple):
            return self.fn(x, batch=[element for element in batch if element.shape[0] == x.shape[0]][0])

        return self.fn(x, batch=batch)


class AddPool(GraphFunctionWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.global_add_pool, configuration)


class MaxPool(GraphFunctionWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.global_max_pool, configuration)


class MeanPool(GraphFunctionWrapper):

    def __init__(self, configuration):
        super().__init__(pyg.nn.global_mean_pool, configuration)


class SelectTarget(BaseWrapper):

    def __init__(self, configuration):
        super().__init__(configuration)

    def forward(self, graph: Graph | pyg.data.Batch, embeddings: torch.FloatTensor):
        storage = graph

        if isinstance(graph, Graph):
            storage = graph.data

        return embeddings[storage[Graph.TARGET_KEY]]


class MaskTarget(BaseWrapper):

    def __init__(self, configuration):
        super().__init__(configuration)

    def forward(self, x: torch.FloatTensor, edge_index, batch: Graph | pyg.data.Batch, target):
        edge_index, _ = pyg.utils.subgraph(edge_index=edge_index, subset=target, relabel_nodes=True, num_nodes=x.shape[0])

        return x[target], edge_index, batch[target]

    @property
    def signature(self):
        return self._signature or 'x, edge_index, batch, target -> x, edge_index, batch'


class SAGPool(BaseWrapper):

    def __init__(self, configuration):
        from .cli import from_cli

        super().__init__(configuration)

        if 'layer' in configuration:
            layer = from_cli(configuration['layer'])

            configuration['GNN'] = layer

            del configuration['layer']

        self.model = pyg.nn.SAGPooling(**configuration)

    def forward(self, x, edge_index, batch):
        x, edge_index, _, batch, _, _ = self.model(x, edge_index=edge_index, batch=batch)

        return tuple([x, edge_index, batch])

    @property
    def signature(self):
        return self._signature or 'x, edge_index, batch -> x, edge_index, batch'


class GraphAttention(BaseWrapper):

    def __init__(self, configuration):
        super().__init__(configuration)

        self.pos_encoding = pyg.nn.PositionalEncoding(**configuration['pos_encoding'])
        layer = torch.nn.TransformerEncoderLayer(**configuration['layer'])
        self.model = torch.nn.TransformerEncoder(layer, **configuration['encoder'])

    def forward(self, x, batch):
        x = x + self.pos_encoding(batch)

        sequences = [x[batch == b] for b in torch.unique(batch)]
        sequences = torch.nn.utils.rnn.pack_sequence(sequences, enforce_sorted=False)
        sequences, lengths = torch.nn.utils.rnn.pad_packed_sequence(sequences, batch_first=False)

        out = self.model(sequences)
        out = torch.nn.utils.rnn.unpad_sequence(out, lengths)

        out = torch.cat(out, dim=0)

        return out

    @property
    def signature(self):
        return self._signature or 'x, batch -> x'
