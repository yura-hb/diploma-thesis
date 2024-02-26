from dataclasses import dataclass
from typing import Dict

import torch
from tensordict.prototype import tensorclass
from torch_geometric.data import HeteroData


@dataclass
class Graph:
    data: HeteroData
    job_operation_map: Dict


@tensorclass
class TensorState:
    state: torch.FloatTensor


@tensorclass
class GraphState:
    graph: Graph

