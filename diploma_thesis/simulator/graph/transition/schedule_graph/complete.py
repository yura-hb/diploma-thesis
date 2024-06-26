from itertools import combinations
from typing import Dict

import torch

from .transition import *


class CompleteTransition(ScheduleTransition):
    """
    Complete schedule transition constructs a disjunction graph. Let consider a machine with
    scheduling queue of operations and processed queue of operations (schedule). Then the graph is constructed as:

    1. If there is no operation in schedule, then all scheduling operations are connected with each other in a complete
       graph
    2. If there is at least one operation in schedule, then the last processed operation is connected with each
       operation in the schedule
    """

    def schedule_implicit(self, job: Job, graph: Graph) -> Graph:
        self.__append_operations_processed_on_single_machine__(job, graph)

        return graph

    def schedule(self, job: Job, graph: Graph) -> Graph:
        self.__append_current_operation_to_scheduled_graph__(job, graph)

        return graph

    def process(self, job: Job, graph: Graph) -> Graph:
        self.__process_current_operation_in_scheduled_graph__(job, graph)

        return graph

    def remove(self, job: Job, graph: Graph) -> Graph:
        self.__remove_operation_record__(job, graph)

        return graph

    # Append

    def __append_current_operation_to_scheduled_graph__(self, job: Job, graph: Graph):
        operation_id = self.__current_operation_id__(graph, job)
        machine_index = self.__get_current_machine_index__(graph, job)

        if operation_id is None or machine_index is None:
            return

        self.__append_operations_to_scheduled_graph__(operation_id, machine_index, graph)

    def __append_operations_processed_on_single_machine__(self, job: Job, graph: Graph):
        for step_id, _ in enumerate(job.step_idx):
            if len(job.processing_times[step_id]) > 1:
                continue

            operation_id = self.__operation_id__(graph, job, step_id, 0)
            machine_index = self.__get_machine_index__(graph, job.step_idx[step_id], 0)

            if machine_index is None or operation_id is None:
                continue

            self.__append_operations_to_scheduled_graph__(operation_id, machine_index, graph)

    def __append_operations_to_scheduled_graph__(self, operations: torch.LongTensor, machine_index: int, graph: Graph):
        super().__append_operations_to_scheduled_graph__(operations, machine_index, graph)

        self.__update_scheduled_graph__(machine_index, graph)

    # Update

    def __process_current_operation_in_scheduled_graph__(self, job: Job, graph: Graph):
        operation_id = self.__current_operation_id__(graph, job)
        machine_index = self.__get_current_machine_index__(graph, job)

        if operation_id is None or machine_index is None:
            return

        s = graph.data[Graph.MACHINE_KEY, machine_index]

        # Remove operation from disjunctive graph
        disjunctive_operations = s[Graph.SCHEDULED_KEY]
        mask = (disjunctive_operations == operation_id).all(dim=0)
        s[Graph.SCHEDULED_KEY] = disjunctive_operations[:, ~mask]

        # Append operation to history
        self.__append__(s, Graph.PROCESSED_KEY, operation_id, dim=1)

        # Update disjunction graph
        self.__update_scheduled_graph__(machine_index, graph)
        self.__update_processed_graph__(machine_index, graph)

    def __update_scheduled_graph__(self, machine_index, graph: Graph):
        dj = graph.data[Graph.MACHINE_KEY, machine_index, Graph.SCHEDULED_KEY]

        if dj.shape[-1] > 1:
            range = torch.arange(dj.shape[-1])
            combinations = torch.combinations(range, 2)

            src = dj[:, combinations[:, 0]]
            dst = dj[:, combinations[:, 1]]

            edges = torch.hstack([
                torch.vstack([src, dst]),
                torch.vstack([dst, src])
            ])
        else:
            edges = torch.tensor([], dtype=torch.long).view(4, 0)

        # Replace disjunctive edges
        graph.data[Graph.MACHINE_KEY, machine_index, Graph.SCHEDULED_GRAPH_KEY] = edges

    def __update_processed_graph__(self, machine_index, graph: Graph):
        history = graph.data.get(
            (Graph.MACHINE_KEY, machine_index, Graph.PROCESSED_KEY), default=torch.tensor([]).view(4, 0)
        )

        if history.shape[1] <= 1:
            new_edges = torch.tensor([], dtype=torch.long).view(4, 0)
        else:
            # Construct a path
            new_edges = torch.vstack([history[:, :-1], history[:, 1:]])

        graph.data[Graph.MACHINE_KEY, machine_index, Graph.PROCESSED_GRAPH_KEY] = new_edges

    # Remove

    def __remove_operation_record__(self, job: Job, graph: Graph):
        for machine_index in range(graph.data[Graph.MACHINE_INDEX_KEY].shape[1]):
            machine_index = key(machine_index)
            s = graph.data[Graph.MACHINE_KEY, machine_index]

            s[Graph.PROCESSED_KEY] = self.__delete_by_first_row__(job.id, s[Graph.PROCESSED_KEY])
            s[Graph.SCHEDULED_KEY] = self.__delete_by_first_row__(job.id, s[Graph.SCHEDULED_KEY])

            self.__update_scheduled_graph__(machine_index, graph)
            self.__update_processed_graph__(machine_index, graph)

    @staticmethod
    def from_cli(parameters: Dict) -> ScheduleTransition:
        return CompleteTransition()
