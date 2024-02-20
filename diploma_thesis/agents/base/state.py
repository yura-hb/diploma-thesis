import torch

from tensordict.prototype import tensorclass


@tensorclass
class TensorState:
    state: torch.FloatTensor


@tensorclass
class GraphState:
    graph: torch.FloatTensor

