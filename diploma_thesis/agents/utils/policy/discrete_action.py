
from .action_policy import *


class DiscreteAction(ActionPolicy):

    def __init__(self, n_actions: int, base_parameters):
        self.n_actions = n_actions

        super().__init__(**base_parameters)

        self.action_layer = self.make_linear_layer(self.n_actions)
        self.value_layer = self.make_linear_layer(1)

    def configure(self, configuration: RunConfiguration):
        super().configure(configuration)

        self.action_layer.to(configuration.device)
        self.value_layer.to(configuration.device)

    def post_encode(self, state, value: torch.FloatTensor, actions: torch.FloatTensor):
        value = self.value_layer(value)
        actions = self.action_layer(actions)

        return value, actions

    @classmethod
    def from_cli(cls, parameters: Dict) -> 'Policy':
        n_actions = parameters['n_actions']
        base_parameters = cls.base_parameters_from_cli(parameters)

        return DiscreteAction(n_actions, base_parameters)
