import torch

from dataclasses import dataclass


@dataclass
class TensorState:
    state: torch.FloatTensor


@dataclass
class GraphState:
    graph: torch.FloatTensor

