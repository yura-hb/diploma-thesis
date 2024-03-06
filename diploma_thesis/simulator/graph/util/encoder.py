import itertools
from dataclasses import dataclass
from typing import Dict

import torch

from agents.base import Graph
from environment import ShopFloor


@dataclass
class Configuration:
    is_machine_set_in_work_center_connected: bool = True
    is_work_center_set_in_shop_floor_connected: bool = True


JOB_OPERATION_MAP_TYPE = Dict[torch.LongTensor, Dict[Graph.OperationKey, int]]


class Encoder:

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    def encode(self,
               previous: Graph | None,
               did_change_jobs: bool,
               graph: Graph,
               shop_floor: ShopFloor,
               job_operation_map: JOB_OPERATION_MAP_TYPE) -> Graph | None:
        if graph is None:
            return None

        result = self.__construct_initial_graph__() if previous is None else previous

        if previous is None:
            self.__append_machine_nodes__(result, graph, shop_floor)

        if did_change_jobs:
            self.__update_job_index__(result, graph, job_operation_map)

        self.__update_forward_graph__(result, graph, shop_floor, job_operation_map)
        self.__update_schedule_graphs__(result, graph)

        return result

    # Initial

    def __construct_initial_graph__(self):
        result = Graph()
        t = torch.tensor([], dtype=torch.long)

        # Nodes
        result.data[Graph.OPERATION_KEY].x = t.view(1, 0)
        result.data[Graph.GROUP_KEY].x = t.view(1, 0)
        result.data[Graph.MACHINE_KEY].x = t.view(1, 0)
        result.data[Graph.WORK_CENTER_KEY].x = t.view(1, 0)

        # Indices
        result.data[Graph.JOB_INDEX_KEY] = t.view(4, 0)

        def __init_rel__(key):
            result.data[*key].edge_index = t.view(2, 0)

        # All Possible Relations
        __init_rel__([Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY])
        __init_rel__([Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.GROUP_KEY])
        __init_rel__([Graph.GROUP_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY])

        __init_rel__([Graph.OPERATION_KEY, Graph.SCHEDULED_RELATION_KEY, Graph.OPERATION_KEY])
        __init_rel__([Graph.OPERATION_KEY, Graph.PROCESSED_RELATION_KEY, Graph.OPERATION_KEY])

        __init_rel__([Graph.MACHINE_KEY, Graph.SCHEDULED_KEY, Graph.OPERATION_KEY])
        __init_rel__([Graph.MACHINE_KEY, Graph.PROCESSED_KEY, Graph.OPERATION_KEY])

        __init_rel__([Graph.MACHINE_KEY, Graph.IN_WORK_CENTER_RELATION_KEY, Graph.MACHINE_KEY])
        __init_rel__([Graph.WORK_CENTER_KEY, Graph.IN_WORK_CENTER_RELATION_KEY, Graph.MACHINE_KEY])
        __init_rel__([Graph.WORK_CENTER_KEY, Graph.IN_SHOP_FLOOR_RELATION_KEY, Graph.WORK_CENTER_KEY])

        return result

    # Append Forward Edges

    def __update_job_index__(self, result: Graph, source: Graph, job_operation_map: JOB_OPERATION_MAP_TYPE):
        job_ids = source.data[Graph.JOB_KEY].keys()
        job_ids = sorted(job_ids)

        n_all_ops = 0

        result.data[Graph.JOB_INDEX_KEY] = torch.tensor([], dtype=torch.long).view(4, 0)

        for job_id in job_ids:
            if job_id not in job_operation_map:
                continue

            index = job_operation_map[job_id]
            index = sorted(index.items(), key=lambda x: x[1])

            min_operation_id = index[0][1]
            max_operation_id = index[-1][1]
            n_ops = max_operation_id - min_operation_id + 1
            n_all_ops += n_ops

            index = torch.vstack(
                [
                    torch.full((n_ops,), fill_value=job_id),
                    torch.arange(n_ops),
                    torch.tensor([k.work_center_id for k, _ in index]),
                    torch.tensor([k.machine_id for k, _ in index])
                ]
            )

            result.data[Graph.JOB_INDEX_KEY] = torch.cat([result.data[Graph.JOB_INDEX_KEY], index], dim=1)

        result.data[Graph.OPERATION_KEY].x = torch.zeros(n_all_ops, dtype=torch.long).view(1, -1)

    def __update_forward_graph__(
         self, result: Graph, source: Graph, shop_floor: ShopFloor, job_operation_map: JOB_OPERATION_MAP_TYPE
    ):
        self.__reset_forward_graph__(result)

        job_ids = source.data[Graph.JOB_KEY].keys()
        job_ids = sorted(job_ids)

        if len(job_ids) == 0:
            return

        has_groups = (source.data[Graph.JOB_KEY][job_ids[0]][Graph.FORWARD_GRAPH_KEY] < 0).sum() > 0
        n_groups = 0

        for job_id in job_ids:
            if job_id not in job_operation_map:
                continue

            job = shop_floor.job(job_id)

            min_operation_id = job_operation_map[job_id][
                Graph.OperationKey(job_id=job_id.item(),
                                   work_center_id=job.step_idx[0].item(),
                                   machine_id=0)
            ]
            max_operation_id = job_operation_map[job_id][
                Graph.OperationKey(job_id=job_id.item(),
                                   work_center_id=job.step_idx[-1].item(),
                                   machine_id=len(job.processing_times[-1]) - 1)
            ]
            n_steps = len(job.step_idx)

            forward_graph = source.data[Graph.JOB_KEY][job_id][Graph.FORWARD_GRAPH_KEY]
            local_operation_to_global_map = torch.arange(min_operation_id, max_operation_id + 1)

            if has_groups:
                # Group nodes are encoded as -step_id -1, which means that they access items in the end of the list
                local_operation_to_global_map = torch.cat(
                    [local_operation_to_global_map,
                     torch.flip(torch.arange(n_groups, n_groups + n_steps, 1), dims=(0,))]
                )

                n_groups += n_steps
                to_group_edges = forward_graph[1, :] < 0

                forward_graph = local_operation_to_global_map[forward_graph]

                s = result.data[Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.GROUP_KEY]
                s.edge_index = torch.cat([s.edge_index, forward_graph[:, to_group_edges]], dim=1)

                s = result.data[Graph.GROUP_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY]
                s.edge_index = torch.cat([s.edge_index, forward_graph[:, ~to_group_edges]], dim=1)
            else:
                s = result.data[Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY]
                s.edge_index = torch.cat([s.edge_index, local_operation_to_global_map[forward_graph]], dim=1)

        result.data[Graph.GROUP_KEY].x = torch.zeros(n_groups, dtype=torch.long).view(1, -1)

    def __append_machine_nodes__(self, result: Graph, source: Graph, shop_floor: ShopFloor):
        result.data[Graph.MACHINE_INDEX_KEY] = source.data[Graph.MACHINE_INDEX_KEY]
        result.data[Graph.MACHINE_KEY].x = torch.zeros(len(shop_floor.machines), dtype=torch.long).view(1, -1)
        result.data[Graph.WORK_CENTER_KEY].x = torch.zeros(len(shop_floor.work_centers), dtype=torch.long).view(1, -1)

        if self.configuration.is_machine_set_in_work_center_connected:
            for work_center in shop_floor.work_centers:
                machine_index = torch.vstack([
                    torch.full((len(work_center.machines),), fill_value=work_center.work_center_idx.item()),
                    torch.arange(len(work_center.machines))
                ])

                machine_node_id = self.__get_node_ids__(machine_index, result.data[Graph.MACHINE_INDEX_KEY])

                edges = itertools.combinations(machine_node_id, 2)
                edges = torch.tensor(list(edges)).view(-1, 2).T

                s = result.data[Graph.MACHINE_KEY, Graph.IN_WORK_CENTER_RELATION_KEY, Graph.MACHINE_KEY]
                s.edge_index = torch.cat([s.edge_index, edges], dim=1)

                s = result.data[Graph.WORK_CENTER_KEY, Graph.IN_WORK_CENTER_RELATION_KEY, Graph.MACHINE_KEY]
                s.edge_index = torch.cat([s.edge_index, torch.vstack([
                    torch.full((len(work_center.machines),), fill_value=work_center.work_center_idx.item()),
                    machine_node_id
                ])], dim=1)

        if self.configuration.is_work_center_set_in_shop_floor_connected:
            edges = itertools.combinations(range(len(shop_floor.work_centers)), 2)
            edges = torch.tensor(list(edges)).T

            s = result.data[Graph.WORK_CENTER_KEY, Graph.IN_SHOP_FLOOR_RELATION_KEY, Graph.WORK_CENTER_KEY]
            s.edge_index = edges

    def __update_schedule_graphs__(self, result: Graph, source: Graph):
        self.__reset_schedule_graph__(result)

        index = result.data[Graph.JOB_INDEX_KEY][[0, 1], :]

        for machine_id in range(len(result.data[Graph.MACHINE_KEY])):
            has_graph = Graph.SCHEDULED_GRAPH_KEY in source.data[Graph.MACHINE_KEY][machine_id]

            if has_graph:
                graph_key = [Graph.SCHEDULED_GRAPH_KEY, Graph.PROCESSED_GRAPH_KEY]
                relation_key = [Graph.SCHEDULED_RELATION_KEY, Graph.PROCESSED_RELATION_KEY]

                for graph, relation in zip(graph_key, relation_key):
                    graph = source.data[Graph.MACHINE_KEY][machine_id][graph]
                    s = result.data[Graph.OPERATION_KEY, relation, Graph.OPERATION_KEY]
                    s.edge_index = torch.cat([s.edge_index, self.__encode_graph__(graph, index)], dim=1)
            else:
                op_key = [Graph.SCHEDULED_KEY, Graph.PROCESSED_KEY]
                relation_key = [Graph.SCHEDULED_RELATION_KEY, Graph.PROCESSED_RELATION_KEY]

                for op, relation in zip(op_key, relation_key):
                    s = result.data[Graph.MACHINE_KEY, relation, Graph.OPERATION_KEY]

                    scheduled_operations = source.data[Graph.MACHINE_KEY][machine_id][op]
                    scheduled_operations = self.__get_node_ids__(scheduled_operations, index)

                    s.edge_index = torch.cat([s.edge_index, torch.vstack([
                        torch.full((scheduled_operations.shape[1],), fill_value=machine_id),
                        scheduled_operations
                    ])], dim=1)

    def __reset_forward_graph__(self, result: Graph):
        t = torch.tensor([], dtype=torch.long).view(2, 0)

        result.data[Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.GROUP_KEY].edge_index = t
        result.data[Graph.GROUP_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY].edge_index = t
        result.data[Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY].edge_index = t

    def __reset_schedule_graph__(self, result: Graph):
        t = torch.tensor([], dtype=torch.long).view(2, 0)

        result.data[Graph.OPERATION_KEY, Graph.SCHEDULED_RELATION_KEY, Graph.OPERATION_KEY].edge_index = t
        result.data[Graph.OPERATION_KEY, Graph.PROCESSED_RELATION_KEY, Graph.OPERATION_KEY].edge_index = t

        result.data[Graph.MACHINE_KEY, Graph.SCHEDULED_RELATION_KEY, Graph.OPERATION_KEY].edge_index = t
        result.data[Graph.MACHINE_KEY, Graph.PROCESSED_RELATION_KEY, Graph.OPERATION_KEY].edge_index = t

    def __encode_graph__(self, graph: torch.Tensor, index: torch.Tensor):
        src_nodes = self.__get_node_ids__(graph[[0, 1], :], index)
        dst_nodes = self.__get_node_ids__(graph[[2, 3], :], index)

        return torch.vstack([src_nodes, dst_nodes])

    def __get_node_ids__(self, values, store):
        if values.numel() == 0:
            return torch.tensor([], dtype=torch.long).view(1, 0)

        result = torch.abs(store - values.unsqueeze(dim=0))

        return (result.sum(axis=-2) == 0).nonzero()[:, 1]
