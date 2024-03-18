import itertools
from dataclasses import dataclass
from typing import Dict, TypeVar

import torch

from agents.base import Graph
from environment import ShopFloor

from .edge import edge
from ..transition.utils import key, unkey


@dataclass
class Configuration:
    is_machine_set_in_work_center_connected: bool = True
    is_work_center_set_in_shop_floor_connected: bool = True


OperationKey = TypeVar('OperationKey')


JOB_OPERATION_MAP_TYPE = Dict[str, Dict[OperationKey, int]]


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
        result = Graph(batch_size=[])
        t = torch.tensor([], dtype=torch.int32)

        def __init_store__(key):
            result.data[key, Graph.X] = t.to(torch.float32).view(0, 1)

        # Nodes
        __init_store__(Graph.OPERATION_KEY)
        __init_store__(Graph.GROUP_KEY)
        __init_store__(Graph.MACHINE_KEY)
        __init_store__(Graph.WORK_CENTER_KEY)

        # Indices
        result.data[Graph.JOB_INDEX_MAP] = t.view(0, 4)

        def __init_rel__(key):
            result.data[key, Graph.EDGE_INDEX] = t.view(2, 0)

        # All Possible Relations
        __init_rel__(edge(Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY))
        __init_rel__(edge(Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.GROUP_KEY))
        __init_rel__(edge(Graph.GROUP_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY))

        __init_rel__(edge(Graph.OPERATION_KEY, Graph.SCHEDULED_RELATION_KEY, Graph.OPERATION_KEY))
        __init_rel__(edge(Graph.OPERATION_KEY, Graph.PROCESSED_RELATION_KEY, Graph.OPERATION_KEY))

        __init_rel__(edge(Graph.MACHINE_KEY, Graph.SCHEDULED_KEY, Graph.OPERATION_KEY))
        __init_rel__(edge(Graph.MACHINE_KEY, Graph.PROCESSED_KEY, Graph.OPERATION_KEY))

        __init_rel__(edge(Graph.MACHINE_KEY, Graph.IN_WORK_CENTER_RELATION_KEY, Graph.MACHINE_KEY))
        __init_rel__(edge(Graph.WORK_CENTER_KEY, Graph.IN_WORK_CENTER_RELATION_KEY, Graph.MACHINE_KEY))
        __init_rel__(edge(Graph.WORK_CENTER_KEY, Graph.IN_SHOP_FLOOR_RELATION_KEY, Graph.WORK_CENTER_KEY))

        return result

    # Append Forward Edges

    def __update_job_index__(self, result: Graph, source: Graph, job_operation_map: JOB_OPERATION_MAP_TYPE):
        job_ids = self.__get_job_ids__(source)

        n_all_ops = 0

        result.data[Graph.JOB_INDEX_MAP] = torch.tensor([], dtype=torch.long).view(0, 4)

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
                    torch.full((n_ops,), fill_value=unkey(job_id, is_tensor=True)),
                    torch.arange(n_ops),
                    torch.tensor([k.work_center_id for k, _ in index]),
                    torch.tensor([k.machine_id for k, _ in index])
                ]
            )

            result.data[Graph.JOB_INDEX_MAP] = torch.cat([result.data[Graph.JOB_INDEX_MAP], index.T], dim=0)

        result.data[Graph.OPERATION_KEY, Graph.X] = torch.zeros(n_all_ops, dtype=torch.float32).view(-1, 1)

    def __append_machine_nodes__(self, result: Graph, source: Graph, shop_floor: ShopFloor):
        result.data[Graph.MACHINE_INDEX_KEY] = source.data[Graph.MACHINE_INDEX_KEY].T
        result.data[Graph.MACHINE_KEY, Graph.X] = torch.zeros(len(shop_floor.machines), dtype=torch.float32).view(-1, 1)
        result.data[Graph.WORK_CENTER_KEY, Graph.X] = torch.zeros(len(shop_floor.work_centers), dtype=torch.float32).view(-1, 1)

        if self.configuration.is_machine_set_in_work_center_connected:
            for work_center in shop_floor.work_centers:
                machine_index = torch.vstack([
                    torch.full((len(work_center.machines),), fill_value=work_center.work_center_idx.item()),
                    torch.arange(len(work_center.machines))
                ])

                machine_node_id = self.__get_node_ids__(machine_index, result.data[Graph.MACHINE_INDEX_KEY].T)

                edges = itertools.combinations(machine_node_id, 2)
                edges = torch.tensor(list(edges), dtype=torch.long).view(-1, 2).T

                s = result.data[edge(Graph.MACHINE_KEY, Graph.IN_WORK_CENTER_RELATION_KEY, Graph.MACHINE_KEY)]
                s[Graph.EDGE_INDEX] = torch.cat([s[Graph.EDGE_INDEX], edges], dim=1)

                s = result.data[edge(Graph.WORK_CENTER_KEY, Graph.IN_WORK_CENTER_RELATION_KEY, Graph.MACHINE_KEY)]
                s[Graph.EDGE_INDEX] = torch.cat([s[Graph.EDGE_INDEX], torch.vstack([
                    torch.full((len(work_center.machines),), fill_value=work_center.work_center_idx.item()),
                    machine_node_id
                ])], dim=1)

        if self.configuration.is_work_center_set_in_shop_floor_connected:
            edges = itertools.combinations(range(len(shop_floor.work_centers)), 2)
            edges = torch.tensor(list(edges), dtype=torch.long).T

            s = result.data[edge(Graph.WORK_CENTER_KEY, Graph.IN_SHOP_FLOOR_RELATION_KEY, Graph.WORK_CENTER_KEY)]
            s[Graph.EDGE_INDEX] = edges

    def __update_forward_graph__(
        self, result: Graph, source: Graph, shop_floor: ShopFloor, job_operation_map: JOB_OPERATION_MAP_TYPE
    ):
        self.__reset_forward_graph__(result)

        job_ids = self.__get_job_ids__(source)

        if len(job_ids) == 0:
            return

        has_groups = (source.data[Graph.JOB_KEY][job_ids[0]][Graph.FORWARD_GRAPH_KEY] < 0).sum() > 0
        n_groups = 0

        for job_id in job_ids:
            if job_id not in job_operation_map.keys():
                continue

            job = shop_floor.job(unkey(job_id, is_tensor=True))

            min_operation_id = job_operation_map[job_id][
                Graph.OperationKey(job_id=job_id,
                                   work_center_id=job.step_idx[0].item(),
                                   machine_id=0)
            ]
            max_operation_id = job_operation_map[job_id][
                Graph.OperationKey(job_id=job_id,
                                   work_center_id=job.step_idx[-1].item(),
                                   machine_id=len(job.processing_times[-1]) - 1)
            ]
            n_steps = len(job.step_idx)

            forward_graph = source.data[Graph.JOB_KEY][job_id][Graph.FORWARD_GRAPH_KEY]
            local_operation_to_global_map = torch.arange(min_operation_id, max_operation_id + 1, dtype=torch.long)

            if has_groups:
                # Group nodes are encoded as -step_id -1, which means that they access items in the end of the list
                local_operation_to_global_map = torch.cat(
                    [local_operation_to_global_map,
                     torch.flip(torch.arange(n_groups, n_groups + n_steps, 1), dims=(0,))]
                )

                n_groups += n_steps
                to_group_edges = forward_graph[1, :] < 0

                forward_graph = local_operation_to_global_map[forward_graph]

                s = result.data[edge(Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.GROUP_KEY)]
                s[Graph.EDGE_INDEX] = torch.cat([s[Graph.EDGE_INDEX], forward_graph[:, to_group_edges]], dim=1)

                s = result.data[edge(Graph.GROUP_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY)]
                s[Graph.EDGE_INDEX] = torch.cat([s[Graph.EDGE_INDEX], forward_graph[:, ~to_group_edges]], dim=1)
            else:
                s = result.data[edge(Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY)]
                s[Graph.EDGE_INDEX] = torch.cat(
                    [s[Graph.EDGE_INDEX], local_operation_to_global_map[forward_graph]], dim=1
                )

        result.data[Graph.GROUP_KEY][Graph.X] = torch.zeros(n_groups, dtype=torch.float32).view(1, -1)

    def __update_schedule_graphs__(self, result: Graph, source: Graph):
        self.__reset_schedule_graph__(result)

        index = result.data[Graph.JOB_INDEX_MAP][:, [0, 1]].T

        for machine_id in range(result.data[Graph.MACHINE_KEY, Graph.X].shape[0]):
            machine_id = key(machine_id)

            has_graph = Graph.SCHEDULED_GRAPH_KEY in source.data[Graph.MACHINE_KEY, machine_id].keys()

            if has_graph:
                graph_key = [Graph.SCHEDULED_GRAPH_KEY, Graph.PROCESSED_GRAPH_KEY]
                relation_key = [Graph.SCHEDULED_RELATION_KEY, Graph.PROCESSED_RELATION_KEY]

                for graph, relation in zip(graph_key, relation_key):
                    graph = source.data[Graph.MACHINE_KEY, machine_id, graph]
                    s = result.data[edge(Graph.OPERATION_KEY, relation, Graph.OPERATION_KEY)]
                    s[Graph.EDGE_INDEX] = torch.cat([s[Graph.EDGE_INDEX], self.__encode_graph__(graph, index)], dim=1)
            else:
                op_key = [Graph.SCHEDULED_KEY, Graph.PROCESSED_KEY]
                relation_key = [Graph.SCHEDULED_RELATION_KEY, Graph.PROCESSED_RELATION_KEY]

                for op, relation in zip(op_key, relation_key):
                    s = result.data[(Graph.MACHINE_KEY, relation, Graph.OPERATION_KEY)]

                    scheduled_operations = source.data[Graph.MACHINE_KEY, machine_id, op]
                    scheduled_operations = self.__get_node_ids__(scheduled_operations, index)

                    s[Graph.EDGE_INDEX] = torch.cat([s[Graph.EDGE_INDEX], torch.vstack([
                        torch.full((scheduled_operations.shape[1],), fill_value=machine_id),
                        scheduled_operations
                    ])], dim=1)

    @staticmethod
    def __reset_forward_graph__(result: Graph):
        t = torch.tensor([], dtype=torch.long).view(2, 0)

        edges = [
            edge(Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.GROUP_KEY),
            edge(Graph.GROUP_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY),
            edge(Graph.OPERATION_KEY, Graph.FORWARD_RELATION_KEY, Graph.OPERATION_KEY)
        ]

        for edge_ in edges:
            if edge_ in result.data.keys(include_nested=True):
                del result.data[edge_]

            result.data[edge_, Graph.EDGE_INDEX] = t

    @staticmethod
    def __reset_schedule_graph__(result: Graph):
        t = torch.tensor([], dtype=torch.long).view(2, 0)

        edges = [
            edge(Graph.OPERATION_KEY, Graph.SCHEDULED_RELATION_KEY, Graph.OPERATION_KEY),
            edge(Graph.OPERATION_KEY, Graph.PROCESSED_RELATION_KEY, Graph.OPERATION_KEY),
            edge(Graph.MACHINE_KEY, Graph.SCHEDULED_RELATION_KEY, Graph.OPERATION_KEY),
            edge(Graph.MACHINE_KEY, Graph.PROCESSED_RELATION_KEY, Graph.OPERATION_KEY)
        ]

        for edge_ in edges:
            if edge_ in result.data.keys(include_nested=True):
                del result.data[edge_]

            result.data[edge_, Graph.EDGE_INDEX] = t

    def __encode_graph__(self, graph: torch.Tensor, index: torch.Tensor):
        src_nodes = self.__get_node_ids__(graph[[0, 1], :], index)
        dst_nodes = self.__get_node_ids__(graph[[2, 3], :], index)

        return torch.vstack([src_nodes, dst_nodes])

    def __get_node_ids__(self, values, store):
        if values.numel() == 0:
            return torch.tensor([], dtype=torch.long).view(1, 0)

        result = torch.abs(store - values.unsqueeze(dim=0).T)

        return (result.sum(axis=-2) == 0).nonzero()[:, 1]

    def __get_job_ids__(self, graph) -> [torch.Tensor]:
        job_ids = graph.data[Graph.JOB_KEY].keys()
        job_ids = sorted(job_ids, key=lambda x: unkey(x))

        return job_ids
