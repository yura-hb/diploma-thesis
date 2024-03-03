import copy
from typing import Dict

from agents.utils import NeuralNetwork, Phase
from agents.utils.nn.layers.linear import Linear
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli
from .policy import *


class DiscreteAction(Policy[Input]):

    def __init__(self,
                 n_actions: int,
                 q_model: NeuralNetwork,
                 advantage_model: NeuralNetwork | None,
                 action_selector: ActionSelector):
        super().__init__()

        self.n_actions = n_actions
        self.q_model = q_model
        self.advantage_model = advantage_model
        self.action_selector = action_selector

        self.__configure__()

    def update(self, phase: Phase):
        self.phase = phase

        for module in [self.q_model, self.advantage_model, self.action_selector]:
            if isinstance(module, PhaseUpdatable):
                module.update(phase)

    def __call__(self, state: State, parameters: Input) -> Record:
        values = self.predict(state).view(-1)
        action, policy = self.action_selector(values)
        action = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.long)

        info = TensorDict({"policy": policy, "values": values.detach().clone()}, batch_size=[])

        return Record(state, action, info, batch_size=[])

    def predict(self, state: State) -> torch.FloatTensor:
        values = self.q_model(state)

        if self.advantage_model is not None:
            advantages = self.advantage_model(state)
            values = values + advantages - advantages.mean(dim=1, keepdim=True)

        return values

    def clone(self):
        return copy.deepcopy(self)

    # Utilities

    def __configure__(self):
        value_output_layer = Linear(dim=1, activation='none', dropout=0)
        action_output_layer = Linear(dim=self.n_actions, activation='none', dropout=0)

        if self.advantage_model is not None:
            self.q_model.append_output_layer(value_output_layer)
            self.advantage_model.append_output_layer(action_output_layer)
        else:
            self.q_model.append_output_layer(action_output_layer)

    @staticmethod
    def from_cli(parameters: Dict) -> 'Policy':
        n_actions = parameters['n_actions']
        q_model = NeuralNetwork.from_cli(parameters['q_model'])
        advantage_model = NeuralNetwork.from_cli(parameters['advantage_model']) if parameters.get('advantage_model') else None
        action_selector = action_selector_from_cli(parameters['action_selector'])

        return DiscreteAction(n_actions, q_model, advantage_model, action_selector)
