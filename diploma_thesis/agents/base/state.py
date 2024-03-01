from dataclasses import dataclass, field

import torch
from tensordict.prototype import tensorclass
from torch_geometric.data import HeteroData


@dataclass
class Graph:
    MACHINE_KEY = "machine"
    WORK_CENTER_KEY = "work_center"
    MACHINE_INDEX_KEY = 'machine_index'
    JOB_KEY = "job"
    GROUP_KEY = "group"
    OPERATION_KEY = "operation"
    SCHEDULED_KEY = "scheduled"
    PROCESSED_KEY = "processed"

    OPERATION_JOB_MAP_KEY = "operation_job_map"
    JOB_INDEX_KEY = "job_index"

    SCHEDULED_GRAPH_KEY = "scheduled_graph"
    PROCESSED_GRAPH_KEY = "processed_graph"
    FORWARD_GRAPH_KEY = "forward_graph"

    FORWARD_RELATION_KEY = "forward"
    SCHEDULED_RELATION_KEY = "scheduled"
    PROCESSED_RELATION_KEY = "processed"
    IN_WORK_CENTER_RELATION_KEY = "in_work_center"
    IN_SHOP_FLOOR_RELATION_KEY = "in_shop_floor"

    @dataclass(frozen=True)
    class OperationKey:
        job_id: int
        work_center_id: int
        machine_id: int

    data: HeteroData = field(default_factory=HeteroData)


@tensorclass
class TensorState:
    state: torch.FloatTensor


@tensorclass
class GraphState:
    graph: Graph

