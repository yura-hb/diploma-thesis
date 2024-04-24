from .action_policy import *
from agents.base.state import State, Graph

import torch_geometric as pyg


class FlexibleAction(ActionPolicy):

    def __init__(self, base_parameters):
        super().__init__(action_layer=lambda: self.make_linear_layer(1),
                         value_layer=lambda: self.make_linear_layer(1),
                         **base_parameters)

    def configure(self, configuration: RunConfiguration):
        super().configure(configuration)

    def encode(self, state):
        output = super().encode(state)

        values, actions, _ = self.__fetch_values__(output)

        # Unpack node embeddings obtained from graph batch
        if state.graph is not None and isinstance(state.graph, pyg.data.Batch):
            result = []
            lengths = []

            store = state.graph[Graph.OPERATION_KEY] if isinstance(state.graph, pyg.data.HeteroData) else state.graph

            prev_count = 0

            for i, j in zip(state.graph.ptr, state.graph.ptr[1:]):
                target = store[Graph.TARGET_KEY][i:j]
                target_nodes_count = target.sum()

                result += [actions[prev_count:prev_count + target_nodes_count].view(-1)]
                lengths += [target_nodes_count]

                prev_count += target_nodes_count

            actions = torch.nn.utils.rnn.pad_sequence(result, batch_first=True, padding_value=-float('inf'))
            lengths = torch.tensor(lengths).to(actions.device)

            output[Keys.ACTIONS] = (actions, lengths)

        return output

    @classmethod
    def from_cli(cls, parameters: Dict) -> 'Policy':
        return FlexibleAction(cls.base_parameters_from_cli(parameters))

