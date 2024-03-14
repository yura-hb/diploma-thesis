from dataclasses import dataclass, field

import torch
import torch_geometric as pyg

from tensordict import TensorDictBase, TensorDict
from tensordict.prototype import tensorclass



@tensorclass
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
    JOB_INDEX_MAP = "job_index_map"

    SCHEDULED_GRAPH_KEY = "scheduled_graph"
    PROCESSED_GRAPH_KEY = "processed_graph"
    FORWARD_GRAPH_KEY = "forward_graph"

    FORWARD_RELATION_KEY = "forward"
    SCHEDULED_RELATION_KEY = "scheduled"
    PROCESSED_RELATION_KEY = "processed"
    IN_WORK_CENTER_RELATION_KEY = "in_work_center"
    IN_SHOP_FLOOR_RELATION_KEY = "in_shop_floor"

    X = 'x'
    NODE_TYPE = 'node_type'
    EDGE_TYPE = 'edge_type'
    EDGE_ATTR = 'edge_attr'
    EDGE_INDEX = 'edge_index'

    @dataclass(frozen=True)
    class OperationKey:
        job_id: str
        work_center_id: int
        machine_id: int

    data: TensorDictBase = field(default_factory=lambda: TensorDict({}, batch_size=[]))

    @staticmethod
    def from_pyg_graph(data: pyg.data.Data | pyg.data.HeteroData):
        result = TensorDict({}, batch_size=[])

        if isinstance(data, pyg.data.Data):
            result[Graph.X] = data.x
            result[Graph.EDGE_INDEX] = data.edge_index
            result[Graph.NODE_TYPE] = data.node_type
            result[Graph.EDGE_TYPE] = data.edge_type

            if data.edge_attr is not None:
                result[Graph.EDGE_ATTR] = data.edge_attr
        elif isinstance(data, pyg.data.HeteroData):
            for key, store in data.node_items():
                for nested_key, value in store.items():
                    result[key, nested_key] = value

            for edge, store in data.edge_items():
                for nested_key, value in store.items():
                    result[edge, nested_key] = value
        else:
            raise ValueError(f'Unknown graph type {data}')

        return Graph(result, batch_size=[])

    def to_pyg_graph(self) -> pyg.data.Data | pyg.data.HeteroData:
        keys = self.data.keys(include_nested=True, leaves_only=True)

        if Graph.X in keys:
            data = pyg.data.Data()

            for key in [Graph.X, Graph.EDGE_INDEX, Graph.NODE_TYPE, Graph.EDGE_ATTR]:
                if key in self.data.keys():
                    data[key] = self.data[key]

            return data

        data = pyg.data.HeteroData()

        for key in keys:
            if isinstance(key, str):
                data[key] = self.data[key]

            if len(key) == 2:
                data[key[0]][key[1]] = self.data[key]

            if len(key) == 4:
                data[key[:3]][key[3]] = self.data[key]

        return data

    def to_pyg_batch(self):
        graph = self.to_pyg_graph()

        return pyg.data.Batch.from_data_list([graph]).to(self.device)


@tensorclass
class TensorState:
    state: torch.FloatTensor


@tensorclass
class GraphState:
    graph: Graph
