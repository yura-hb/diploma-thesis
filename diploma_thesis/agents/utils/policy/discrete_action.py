from typing import Dict

from agents.utils import NNCLI, Phase
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli
from .policy import *


class DiscreteAction(Generic[Rule, Input, Record], Policy[Rule, Input, Record], metaclass=ABCMeta):

    def __init__(self, n_actions: int, q_model: NNCLI, advantage_model: NNCLI | None, action_selector: ActionSelector):
        super().__init__()

        self.n_actions = n_actions
        self.q_model = q_model
        self.advantage_model = advantage_model
        self.action_selector = action_selector

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
        # assert isinstance(state, TensorState), f"State must conform to TensorState"
        #
        # if not self.model.is_connected:
        #     self.__connect__(len(self.rules), self.model, state.state.shape)
        #
        # tensor = torch.atleast_2d(state.state)
        #
        # return self.model(tensor)
        pass

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

    def __connect__(self, model: NNCLI, input_shape: torch.Size):
        pass
        # output_layer = NNCLI.Configuration.Linear(dim=self.n_actions, activation='none', dropout=0)
        #
        # model.connect(input_shape, output_layer)

    @staticmethod
    def from_cli(parameters: Dict) -> 'Policy':
        n_actions = parameters['n_actions']
        q_model = NNCLI.from_cli(parameters['q_model'])
        advantage_model = NNCLI.from_cli(parameters['advantage_model']) if parameters.get('advantage_model') else None
        action_selector = action_selector_from_cli(parameters['action_selector'])

        return DiscreteAction(n_actions, q_model, advantage_model, action_selector)
