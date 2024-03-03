from typing import Dict

from agents.utils import NN, Phase
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli
from .policy import *


class DiscreteAction(Generic[Rule, Input, Record], Policy[Rule, Input, Record], metaclass=ABCMeta):

    def __init__(self, n_actions: int, q_model: NN, advantage_model: NN | None, action_selector: ActionSelector):
        super().__init__()

        self.n_actions = n_actions
        self.q_model = q_model
        self.advantage_model = advantage_model
        self.action_selector = action_selector

        self.__configure_model_output_layers__()

    def update(self, phase: Phase):
        self.phase = phase

        for module in [self.q_model, self.advantage_model, self.action_selector]:
            if isinstance(module, PhaseUpdatable):
                module.update(phase)

    def __call__(self, state: State, parameters: Input) -> Record:
        values = self.predict(state).view(-1)
        action, policy = self.action_selector(values)
        action = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.long)

        info = TensorDict(batch_size=[])
        info['policy'] = policy
        info['values'] = values

        return Record(state, action, info)

    def predict(self, state: State) -> torch.FloatTensor:
        values = self.q_model(state)

        if self.advantage_model is not None:
            advantages = self.advantage_model(state)
            values = values + advantages - advantages.mean(dim=1, keepdim=True)

        return values

    def parameters(self, recurse: bool = True):
        result = [
            {'params': self.q_model.parameters(recurse)}
        ]

        if self.advantage_model:
            result.append({'params': self.advantage_model.parameters(recurse)})

        return result

    def copy_parameters(self, other: 'DiscreteAction', decay: float = 1.0):
        self.q_model.copy_parameters(other.q_model, decay)

        if self.advantage_model and other.advantage_model:
            self.advantage_model.copy_parameters(other.advantage_model, decay)

    # Utilities

    def __configure_model_output_layers__(self):
        value_output_layer = NN.Linear(dim=1, activation='none', dropout=0)
        action_output_layer = NN.Linear(dim=self.n_actions, activation='none', dropout=0)

        if self.advantage_model is not None:
            self.q_model.append(value_output_layer)
            self.advantage_model.append(action_output_layer)
        else:
            self.q_model.append(action_output_layer)

    @staticmethod
    def from_cli(parameters: Dict) -> 'Policy':
        n_actions = parameters['n_actions']
        q_model = NN.from_cli(parameters['q_model'])
        advantage_model = NN.from_cli(parameters['advantage_model']) if parameters.get('advantage_model') else None
        action_selector = action_selector_from_cli(parameters['action_selector'])

        return DiscreteAction(n_actions, q_model, advantage_model, action_selector)
