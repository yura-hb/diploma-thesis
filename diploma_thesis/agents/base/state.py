import torch

from tensordict.prototype import tensorclass


@tensorclass
class TensorState:
    state: torch.FloatTensor


@tensorclass
class Graph:
    node_features: torch.FloatTensor
    edge_features: torch.FloatTensor
    edges: torch.LongTensor


@tensorclass
class GraphState:
    graph: Graph

