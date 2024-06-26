
from typing import Dict

import torch

from .transition import *


class MachineCompressedTransition(ScheduleTransition):
    """
    Compressed schedule transition constructs a operation-machine graph. Let consider a machine with operations in
    scheduling queue and processed queue (schedule). Then the graph is constructed as:

    1. If there is an operation in schedule queue, then there is an edge from machine to that operation
    2. If there is an operation in processed queue, then there is an edge from that operation to machine
    """

    def schedule_implicit(self, job: Job, graph: Graph) -> Graph:
        self.__schedule_implicit__(job, graph)

        return graph

    def schedule(self, job: Job, graph: Graph) -> Graph:
        self.__schedule__(job, graph)

        return graph

    def process(self, job: Job, graph: Graph) -> Graph:
        self.__process__(job, graph)

        return graph

    def remove(self, job: Job, graph: Graph) -> Graph:
        self.__remove__(job, graph)

        return graph

    def __schedule_implicit__(self, job: Job, graph: Graph):
        for step_id, _ in enumerate(job.step_idx):
            if len(job.processing_times[step_id]) > 1:
                continue

            operation_id = self.__operation_id__(graph, job, step_id, 0)
            machine_index = self.__get_machine_index__(graph, job.step_idx[step_id], 0)

            if machine_index is None or operation_id is None:
                continue

            self.__append__(graph.data[Graph.MACHINE_KEY, machine_index], Graph.SCHEDULED_KEY, operation_id, dim=1)

    def __schedule__(self, job: Job, graph: Graph):
        operation_id = self.__current_operation_id__(graph, job)
        machine_index = self.__get_current_machine_index__(graph, job)

        if operation_id is None or machine_index is None:
            return

        self.__append__(graph.data[Graph.MACHINE_KEY, machine_index], Graph.SCHEDULED_KEY, operation_id, dim=1)

    def __process__(self, job: Job, graph: Graph):
        operation_id = self.__current_operation_id__(graph, job)
        machine_index = self.__get_current_machine_index__(graph, job)

        if operation_id is None or machine_index is None:
            return

        s = graph.data[Graph.MACHINE_KEY, machine_index]

        self.__append__(s, Graph.PROCESSED_KEY, operation_id, dim=1)

        s[Graph.SCHEDULED_KEY] = self.__delete_by_first_row__(value=job.id, tensor=s[Graph.SCHEDULED_KEY])

    def __remove__(self, job: Job, graph: Graph):
        for index in range(graph.data[Graph.MACHINE_INDEX_KEY].shape[1]):
            index = key(index)
            s = graph.data[Graph.MACHINE_KEY, index]

            s[Graph.SCHEDULED_KEY] = self.__delete_by_first_row__(value=job.id, tensor=s[Graph.SCHEDULED_KEY])
            s[Graph.PROCESSED_KEY] = self.__delete_by_first_row__(value=job.id, tensor=s[Graph.PROCESSED_KEY])

    @staticmethod
    def from_cli(parameters: Dict) -> ScheduleTransition:
        return MachineCompressedTransition()
