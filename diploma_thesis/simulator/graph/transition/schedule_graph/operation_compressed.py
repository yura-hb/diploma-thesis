import torch

from .complete import *


class OperationCompressedTransition(CompleteTransition):
    # Instread of constructing disjunctive graph, the last operation is connected to all operations in the machine
    # queue

    def __init__(self, skip_no_history_phase: bool = False):
        super().__init__()

        self.skip_no_history_phase = skip_no_history_phase

    def __update_scheduled_graph__(self, machine_index, graph: Graph):
        dj = graph.data[Graph.MACHINE_KEY, machine_index, Graph.SCHEDULED_KEY]

        # Construct new disjunctive edges, i.e complete graph
        history = graph.data.get(
            (Graph.MACHINE_KEY, machine_index, Graph.PROCESSED_KEY), default=torch.tensor([]).view(4, 0)
        )

        # edges = [torch.atleast_2d(torch.cat([edge[0], edge[1]])).T for edge in edges]

        if history.numel() == 0 and dj.shape[-1] > 1 and not self.skip_no_history_phase:
            range = torch.arange(dj.shape[-1])
            combinations = torch.combinations(range, 2)

            src = dj[:, combinations[:, 0]]
            dst = dj[:, combinations[:, 1]]

            edges = torch.hstack([
                torch.vstack([src, dst]),
                torch.vstack([dst, src])
            ])
        elif history.numel() > 0:
            # Otherwise, we need to select one edge from last processed operation to all operations in the disjunction
            edges = torch.zeros((2, dj.shape[1]), dtype=torch.long)
            edges += history[:, -1].view(2, 1)
            edges = torch.hstack([torch.vstack([edges, dj]), torch.vstack([dj, edges])])
        else:
            edges = torch.tensor([], dtype=torch.long).view(4, 0)

        # Replace disjunctive edges
        graph.data[Graph.MACHINE_KEY, machine_index, Graph.SCHEDULED_GRAPH_KEY] = edges

    @staticmethod
    def from_cli(parameters: Dict) -> ScheduleTransition:
        return OperationCompressedTransition(
            skip_no_history_phase=parameters.get('skip_no_history_phase', False)
        )