
from typing import Dict

import torch

from agents.base import Graph
from environment import Job, ShopFloor
from .forward_graph import ForwardTransition
from .schedule_graph import ScheduleTransition


class GraphTransition:

    def __init__(self, forward_transition: ForwardTransition, schedule_transition: ScheduleTransition):
        self.forward_transition = forward_transition
        self.schedule_transition = schedule_transition

    def append(self, job: Job, shop_floor: ShopFloor, graph: Graph):
        if Graph.JOB_KEY not in graph.data:
            self.__build_initial_graph__(graph, shop_floor)

        self.__append_job__(job, graph)

        return self.schedule_transition.schedule_implicit(job, graph)

    def update_on_dispatch(self, job: Job, shop_floor: ShopFloor, graph: Graph):
        self.__update_next_operation_edges_on_dispatch__(job, graph)
        self.schedule_transition.schedule(job, graph)

        return graph

    def update_on_will_produce(self, job: Job, shop_floor: ShopFloor, graph: Graph):
        self.schedule_transition.process(job, graph)

        return graph

    def remove(self, job: Job, shop_floor: ShopFloor, graph: Graph):
        try:
            self.__remove_job__(job, graph)
        except:
            return graph

        self.schedule_transition.remove(job, graph)

        return graph

    # Initial Graph

    @classmethod
    def __build_initial_graph__(cls, graph: Graph, shop_floor: ShopFloor):
        graph.data[Graph.JOB_KEY] = dict()
        graph.data[Graph.MACHINE_KEY] = dict()

        machine_index = torch.tensor([], dtype=torch.long).view(2, 0)

        for index, machine in enumerate(shop_floor.machines):
            machine_index = torch.cat([
                machine_index,
                torch.vstack([machine.work_center_idx, machine.machine_idx])
            ], dim=1)

            graph.data[Graph.MACHINE_KEY][index] = dict()

            t = torch.tensor([], dtype=torch.long)

            graph.data[Graph.MACHINE_KEY][index][Graph.SCHEDULED_KEY] = t.clone().view(2, 0)
            graph.data[Graph.MACHINE_KEY][index][Graph.PROCESSED_KEY] = t.clone().view(2, 0)

            graph.data[Graph.MACHINE_KEY][index][Graph.SCHEDULED_GRAPH_KEY] = t.clone().view(4, 0)
            graph.data[Graph.MACHINE_KEY][index][Graph.PROCESSED_GRAPH_KEY] = t.clone().view(4, 0)

        graph.data[Graph.MACHINE_INDEX_KEY] = machine_index

    # Append

    def __append_job__(self, job: Job, graph: Graph):
        operations_count = 0

        for step_id, work_center_id in enumerate(job.step_idx):
            operations_count += len(job.processing_times[step_id])

        graph.data[Graph.JOB_KEY][job.id] = dict()
        graph.data[Graph.JOB_KEY][job.id]['job_id'] = job.id
        graph.data[Graph.JOB_KEY][job.id][Graph.FORWARD_GRAPH_KEY] = self.forward_transition.construct(job)
        graph.data[Graph.JOB_KEY][job.id][Graph.GROUP_KEY] = torch.tensor([], dtype=torch.long).view(2, 0)

    # Update

    def __update_next_operation_edges_on_dispatch__(self, job: Job, graph: Graph):
        if job.id not in graph.data[Graph.JOB_KEY]:
            return

        graph.data[Graph.JOB_KEY][job.id][Graph.FORWARD_GRAPH_KEY] = self.forward_transition.construct(job)

    # Remove

    @classmethod
    def __remove_job__(cls, job: Job, graph: Graph):
        if job.id in graph.data[Graph.JOB_KEY]:
            graph.data[Graph.JOB_KEY].pop(job.id)
            return
        
        raise ValueError(f'Job with id {job.id} not found in graph')

    # Utils

    @classmethod
    def from_cli(cls, parameters: Dict):
        from .forward_graph import from_cli as forward_from_cli
        from .schedule_graph import from_cli as schedule_from_cli

        return GraphTransition(
            forward_transition=forward_from_cli(parameters['forward']),
            schedule_transition=schedule_from_cli(parameters['schedule']),
        )
