from itertools import combinations
from typing import Dict

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

        # Remove operation from disjunctive graph
        disjunctive_operations = graph.data[Graph.MACHINE_KEY][machine_index][Graph.SCHEDULED_KEY]
        mask = (disjunctive_operations == operation_id).all(dim=0)
        graph.data[Graph.MACHINE_KEY][machine_index][Graph.SCHEDULED_KEY] = disjunctive_operations[:, ~mask]

        # Append operation to history
        graph.data[Graph.MACHINE_KEY][machine_index][Graph.PROCESSED_KEY] = torch.cat([
            graph.data[Graph.MACHINE_KEY][machine_index][Graph.PROCESSED_KEY],
            operation_id
        ], dim=1)

        # Update disjunction graph
        self.__update_scheduled_graph__(machine_index, graph)
        self.__update_processed_graph__(machine_index, graph)

    def __update_scheduled_graph__(self, machine_index, graph: Graph):
        dj = graph.data[Graph.MACHINE_KEY][machine_index][Graph.SCHEDULED_KEY]

        # Construct new disjunctive edges, i.e complete graph
        history = graph.data[Graph.MACHINE_KEY][machine_index][Graph.PROCESSED_KEY]

        if history.numel() == 0 and dj.shape[1] > 1:
            # If no operation was processed on machine, then it can start with any of them in the queue. Hence, we
            # need to construct a complete graph
            edges = [dj[:, i] for i in range(dj.shape[1])]
            edges = list(combinations(edges, 2))
            edges = [torch.atleast_2d(torch.cat([edge[0], edge[1]])).T for edge in edges]
            edges = torch.hstack(edges)
        elif (history.numel() == 0 and dj.shape[1] <= 1) or dj.shape[1] == 0:
            # If there is number of nodes in graph is less than two, then there is no edges
            edges = torch.tensor([], dtype=torch.long).view(4, 0)
        else:
            # Otherwise, we need to select one edge from last processed operation to all operations in the disjunction
            edges = torch.zeros((2, dj.shape[1]), dtype=torch.long)
            edges += history[:, -1].view(2, 1)
            edges = torch.vstack([edges, dj])

        # Replace disjunctive edges
        graph.data[Graph.MACHINE_KEY][machine_index][Graph.SCHEDULED_GRAPH_KEY] = edges

    def __update_processed_graph__(self, machine_index, graph: Graph):
        history = graph.data[Graph.MACHINE_KEY][machine_index][Graph.PROCESSED_KEY]

        if history.shape[1] <= 1:
            new_edges = torch.tensor([], dtype=torch.long).view(2, 0)
        else:
            new_edges = torch.vstack([history[:, :-1], history[:, 1:]])

        graph.data[Graph.MACHINE_KEY][machine_index][Graph.PROCESSED_GRAPH_KEY] = new_edges

    # Remove

    def __remove_operation_record__(self, job: Job, graph: Graph):
        for machine_index, _ in enumerate(graph.data[Graph.MACHINE_INDEX_KEY]):
            machine_index = key(machine_index)

            graph.data[Graph.MACHINE_KEY][machine_index][Graph.PROCESSED_KEY] = self.__delete_by_first_row__(
                job.id, graph.data[Graph.MACHINE_KEY][machine_index][Graph.PROCESSED_KEY]
            )

            graph.data[Graph.MACHINE_KEY][machine_index][Graph.SCHEDULED_KEY] = self.__delete_by_first_row__(
                job.id, graph.data[Graph.MACHINE_KEY][machine_index][Graph.SCHEDULED_KEY]
            )

            self.__update_scheduled_graph__(machine_index, graph)
            self.__update_processed_graph__(machine_index, graph)

    @staticmethod
    def from_cli(parameters: Dict) -> ScheduleTransition:
        return CompleteTransition()
