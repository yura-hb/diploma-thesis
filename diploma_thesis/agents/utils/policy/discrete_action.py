
from .action_policy import *


class DiscreteAction(ActionPolicy):

    def __init__(self, n_actions: int, base_parameters):
        self.n_actions = n_actions

        super().__init__(action_layer=lambda: self.make_linear_layer(self.n_actions),
                         value_layer=lambda: self.make_linear_layer(1),
                         **base_parameters)

    def configure(self, configuration: RunConfiguration):
        super().configure(configuration)

    def encode(self, state: State):
        output = super().encode(state)

        return output

    @classmethod
    def from_cli(cls, parameters: Dict) -> 'Policy':
        n_actions = parameters['n_actions']
        base_parameters = cls.base_parameters_from_cli(parameters)

        return DiscreteAction(n_actions, base_parameters)
