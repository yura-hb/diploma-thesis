from .action_policy import *
from agents.base.state import State, Graph

import torch_geometric as pyg


class FlexibleAction(ActionPolicy):

    def __init__(self, base_parameters):
        super().__init__(**base_parameters)

        self.action_layer = self.make_linear_layer(1)

    @property
    def is_recurrent(self):
        return False

    def configure(self, configuration: RunConfiguration):
        super().configure(configuration)

        self.action_layer.to(configuration.device)

    def post_encode(self, state: State, value: torch.FloatTensor, actions: torch.FloatTensor):
        actions = self.action_layer(actions)

        # Unpack node embeddings obtained from graph batch
        if state.graph is not None and isinstance(state.graph, pyg.data.Batch):
            result = []
            lengths = []

            store = state.graph[Graph.OPERATION_KEY] if isinstance(state.graph, pyg.data.HeteroData) else state.graph

            prev_count = 0

            for i, j in zip(state.graph.ptr, state.graph.ptr[1:]):
                target = store[Graph.TARGET_KEY][i:j]
                target_nodes_count = target.sum()

                result += [actions[prev_count:prev_count+target_nodes_count].view(-1)]
                lengths += [target_nodes_count]

                prev_count += target_nodes_count

            actions = torch.nn.utils.rnn.pad_sequence(result, batch_first=True, padding_value=torch.nan)
            lengths = torch.tensor(lengths)

            return super().post_encode(state, value, (actions, lengths))

        return super().post_encode(state, value, actions)

    def __estimate_policy__(self, value, actions):
        if isinstance(actions, tuple):
            # Encode as logits with zero probability
            min_value = torch.finfo(torch.float32).min
            post_process = lambda x: torch.nan_to_num(x, nan=min_value)

            match self.policy_estimation_method:
                case PolicyEstimationMethod.DUELING_ARCHITECTURE:
                    if isinstance(actions, tuple):
                        actions, lengths = actions
                        means = torch.nan_to_num(actions, nan=0.0).sum(dim=-1) / lengths

                        return value, post_process(value + actions - means)
                case _:
                    return value, post_process(actions[0])

        return super().__estimate_policy__(value, actions)

    @classmethod
    def from_cli(cls, parameters: Dict) -> 'Policy':
        return FlexibleAction(cls.base_parameters_from_cli(parameters))

