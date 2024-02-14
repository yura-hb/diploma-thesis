import torch

from dataclasses import dataclass
from tensordict.prototype import tensorclass


@tensorclass
class TensorState:
    state: torch.FloatTensor


@tensorclass
class GraphState:
    graph: torch.FloatTensor

