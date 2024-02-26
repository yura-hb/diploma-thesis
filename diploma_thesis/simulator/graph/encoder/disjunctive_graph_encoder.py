import itertools

import torch

from simulator.graph.graph_model import *


class DisjunctiveGraphEncoder:
    """
    Encodes all shop-floor jobs into disjunctive graph.
        1. Nodes represent job operation. For each node the following information is filled:
            * job_id: int - job id
            * work_center_id: int - work center id
            * machine_id: int - machine id within the work center
            * is_target: bool - whether the operation participates in the decision
        2. There are several relations between operations:
           1. Processed:
           2. Scheduled:
           3. Next: An edge from operation to operation representing the order of operations
    """

    def encode(self, shop_floor: ShopFloor, key: WorkCenterKey | MachineKey, moment: float):
        jobs = shop_floor.in_system_jobs
        job_ids = [job.id for job in jobs]

        assert job_ids == sorted(job_ids), "Jobs must be sorted by id."

        job_operation_map, n_operations = self.__make_job_operation_map__(jobs)

        data = HeteroData()

        self.__encode_operations__(job_operation_map, n_operations, data)
        self.__encode_target_operations__(shop_floor, key, data)
        self.__encode_scheduling_progress__(job_operation_map, shop_floor, data)

        return data

    @staticmethod
    def __encode_operations__(job_operation_map: Dict, n_operations: int, data: HeteroData):
        data['operation'].job_id = torch.zeros(n_operations, dtype=torch.long)
        data['operation'].work_center_id = torch.zeros(n_operations, dtype=torch.long)
        data['operation'].machine_id = torch.zeros(n_operations, dtype=torch.long)

        for job_id, operations in job_operation_map.items():
            for machine_key, operation_idx in operations.items():
                data['operation'].job_id[operation_idx] = job_id
                data['operation'].work_center_id[operation_idx] = machine_key.work_center_id
                data['operation'].machine_id[operation_idx] = machine_key.machine_id

        return data

    @staticmethod
    def __encode_target_operations__(shop_floor: ShopFloor, key: WorkCenterKey | MachineKey, data: HeteroData):
        data['operation'].is_target = torch.zeros_like(data['operation'].job_id, dtype=torch.bool)

        match key:
            case WorkCenterKey(work_center_id):
                jobs = shop_floor.work_center(work_center_id).state.queue

                for job in jobs:
                    is_target = data['operation'].job_id == job.id & data['operation'].work_center_id == work_center_id

                    data['operation'].is_target |= is_target
            case MachineKey(work_center_id, machine_id):
                jobs = shop_floor.work_center(work_center_id).machines[machine_id].state.queue

                for job in jobs:
                    is_target = (data['operation'].job_id == job.id &
                                 data['operation'].work_center_id == work_center_id &
                                 data['operation'].machine_id == machine_id)

                    data['operation'].is_target |= is_target

    @staticmethod
    def __encode_next_operation_edges__(job_operation_map: Dict, shop_floor: ShopFloor, data: HeteroData):
        edges = torch.LongTensor([]).view(2, 0)

        for job_id, key_to_operation in job_operation_map.items():
            job = shop_floor.job(job_id)

            result = []

            for index, work_center_id in enumerate(job.step_idx):
                if index + 1 >= len(job.step_idx):
                    break

                result = itertools.product(range(len(job.processing_times[index])),
                                           range(len(job.processing_times[index + 1])))
                result = list(result)

            edges = torch.cat([edges, torch.LongTensor(result).T], dim=1)

        data['operation', 'next', 'operation'].edge_index = edges

    @staticmethod
    def __encode_scheduling_progress__(job_operation_map: Dict, shop_floor: ShopFloor, data: HeteroData):
        pass
        # data['operation'].scheduled = torch.zeros_like(data['operation'].job_id, dtype=torch.bool)
        # data['operation', 'scheduled', 'operation'].edge_index = ...