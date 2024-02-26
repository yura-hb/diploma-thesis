from abc import ABCMeta, abstractmethod
from typing import List, Tuple, Dict

from torch_geometric.data import HeteroData
from agents.base import Graph
from environment import ShopFloor, Job, MachineKey, WorkCenterKey, Delegate, Context
from tensordict.prototype import tensorclass


class GraphModel(Delegate):

    def graph(self, context: Context, key: WorkCenterKey | MachineKey) -> Graph | None:
        return None

    @staticmethod
    def from_cli(parameters: Dict) -> 'GraphModel':
        return GraphModel()

    #
    # @abstractmethod
    # def encode(self, shop_floor: ShopFloor, key: WorkCenterKey | MachineKey, moment: float) -> Tuple[HeteroData, Dict]:
    #     """
    #     Encodes shop floor state into graph.
    #
    #     Args:
    #         - shop_floor: ShopFloor
    #         - key: WorkCenterKey | MachineKey
    #             Key of the work center or machine, which performs the decision
    #         - moment: float
    #             Moment of the time, when the decision is made
    #
    #     Returns:
    #         - HeteroData: Encoded graph
    #         - Dict: Job operation map
    #     """
    #     pass
    #
    # @staticmethod
    # def __make_job_operation_map__(jobs: List[Job]) -> Tuple[Dict, int]:
    #     result = {}
    #     operations_count = 0
    #
    #     for job in jobs:
    #         result[job.id] = dict()
    #
    #         for step_id, work_center_id in enumerate(job.step_idx):
    #             for _, machine_id in enumerate(job.processing_times[step_id]):
    #                 key = MachineKey(work_center_id, machine_id)
    #
    #                 result[key] = operations_count
    #                 operations_count += 1
    #
    #     return result, operations_count
