
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

    def post_encode(self, state, output):
        value, action, _ = self.__fetch_values__(output)

        output[Keys.VALUE] = self.value_layer(value)
        output[Keys.ACTIONS] = self.action_layer(action)

        return self.__estimate_policy__(output)

    @classmethod
    def from_cli(cls, parameters: Dict) -> 'Policy':
        n_actions = parameters['n_actions']
        base_parameters = cls.base_parameters_from_cli(parameters)

        return DiscreteAction(n_actions, base_parameters)
