from dataclasses import dataclass
from typing import Dict

import torch

from agents.base import Graph
from environment import ShopFloor


@dataclass
class Configuration:
    pass


class Encoder:

    def encode(self,
               graph: Graph,
               shop_floor: ShopFloor,
               job_operation_map: Dict[torch.LongTensor, Dict[Graph.OperationKey, int]]) -> Graph:
        result = self.__construct_initial_graph__()

        return result

    # Initial

    def __construct_initial_graph__(self):
        result = Graph()
        t = torch.tensor([], dtype=torch.long)

        # Nodes
        result.data[Graph.OPERATION_KEY] = t.view(1, 0)
        result.data[Graph.GROUP_KEY] = t.view(1, 0)
        result.data[Graph.MACHINE_KEY] = t.view(1, 0)
        result.data[Graph.WORK_CENTER_KEY] = t.view(1, 0)

        # Indices
        result.data['operation_job_map'] = ...
        result.data['job_index'] = ...

        # All Possible Relations
        result.data[Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY] = t.view(2, 0)
        result.data[Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.GROUP_KEY] = t.view(2, 0)
        result.data[Graph.GROUP_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY] = t.view(2, 0)

        result.data[Graph.OPERATION_KEY, Graph.SCHEDULED_RELATION_KEY, Graph.OPERATION_KEY] = t.view(2, 0)
        result.data[Graph.OPERATION_KEY, Graph.PROCESSED_RELATION_KEY, Graph.OPERATION_KEY] = t.view(2, 0)

        result.data[Graph.MACHINE_KEY, Graph.SCHEDULED_KEY, Graph.OPERATION_KEY] = t.view(2, 0)
        result.data[Graph.MACHINE_KEY, Graph.PROCESSED_KEY, Graph.OPERATION_KEY] = t.view(2, 0)

        result.data[Graph.WORK_CENTER_KEY, Graph.IN_WORK_CENTER_RELATION_KEY, Graph.MACHINE_KEY] = t.view(2, 0)
        result.data[Graph.WORK_CENTER_KEY, Graph.IN_SHOP_FLOOR_RELATION_KEY, Graph.WORK_CENTER_KEY] = t.view(2, 0)

        return result

    # Append Forward Edges

    def __append_jobs__(self,
                        result: Graph,
                        source: Graph,
                        shop_floor: ShopFloor, job_operation_map: Dict[torch.LongTensor, Dict[Graph.OperationKey, int]]):
        job_ids = source[Graph.JOB_KEY].keys()
        job_ids = sorted(job_ids)
        job_ids = torch.tensor(job_ids)

        result.data['job_index'] = job_ids
        result.data['']



