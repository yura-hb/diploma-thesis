
from simulator.graph.graph_model import *


class BatchGraphEncoder:
    """
    Encodes all shop-floor jobs into disjunctive graph.
        1. Nodes are distinct operations in the graph.
        2. Edges are the following:
           1. Processed:
           2. Scheduled:
           3. To:
    """

    def encode(self, shop_floor: ShopFloor, target_jobs: List[Job]):
        jobs = shop_floor.in_system_jobs

        data = HeteroData()

        data['operation'].processing_time = ...
        data['operation'].job_id = ...
        data['operation'].work_center_id = ...
        data['operation'].machine_id = ...

        data['operation', 'to', 'operation'].edge_index = ...
        data['operation', 'scheduled', 'operation'].edge_index = ...
        data['operation', 'processed', 'operation'].edge_index = ...

        return data