from .flexible_action import *

from agents.utils.nn.layers import Linear


class DiscreteAction(FlexibleAction):

    def __init__(self, n_actions: int, base_parameters):
        self.n_actions = n_actions

        super().__init__(**base_parameters)

    def __get_values__(self, state):
        values = self.value_model(state)

        return values.expand(-1, self.n_actions)

    # Utilities

    def __configure__(self):
        if self.action_model is not None:
            self.action_model.append_output_layer(Linear(dim=self.n_actions, activation='none', dropout=0))

        if self.value_model is not None:
            self.value_model.append_output_layer(Linear(dim=1, activation='none', dropout=0))

        super().__configure__()

    @classmethod
    def from_cli(cls, parameters: Dict) -> 'Policy':
        n_actions = parameters['n_actions']
        base_parameters = cls.base_parameters_from_cli(parameters)

        return DiscreteAction(n_actions, base_parameters)
