import copy
from typing import Dict

from agents.utils import NeuralNetwork, Phase
from agents.utils.nn.layers.linear import Linear
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli
from .policy import *
from enum import StrEnum


class PolicyEstimationMethod(StrEnum):
    DUELING_ARCHITECTURE = "dueling_architecture"
    INDEPENDENT = "independent"


class DiscreteAction(Policy[Input]):

    def __init__(self,
                 n_actions: int,
                 value_model: NeuralNetwork,
                 action_model: NeuralNetwork,
                 action_selector: ActionSelector,
                 value_estimation_method: PolicyEstimationMethod = PolicyEstimationMethod.INDEPENDENT):
        super().__init__()

        self.n_actions = n_actions
        self.value_model = value_model
        self.action_model = action_model
        self.action_selector = action_selector
        self.policy_estimation_method = value_estimation_method

        self.__configure__()

    def update(self, phase: Phase):
        self.phase = phase

        for module in [self.value_model, self.advantage_model, self.action_selector]:
            if isinstance(module, PhaseUpdatable):
                module.update(phase)

    def __call__(self, state: State, parameters: Input) -> Record:
        values, actions = self.predict(state).view(-1)
        action, policy = self.action_selector(actions)
        action = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.long)

        info = TensorDict({
            "policy": policy,
            "values": values.detach().clone(),
            "actions": actions.detach().clone()
        }, batch_size=[])

        return Record(state, action, info, batch_size=[])

    def predict(self, state: State):
        values = None
        actions = None

        if self.value_model is not None:
            values = self.value_model(state)

        if self.advantage_model is not None:
            actions = self.action_model(state)

        match self.policy_estimation_method:
            case PolicyEstimationMethod.INDEPENDENT:
                return values, actions
            case PolicyEstimationMethod.DUELING_ARCHITECTURE:
                return values, values + (actions - actions.mean(dim=1, keepdim=True))
            case _:
                raise ValueError(f"Policy estimation method {self.policy_estimation_method} is not supported")

    def clone(self):
        return copy.deepcopy(self)

    # Utilities

    def __configure__(self):
        if self.action_model is not None:
            self.action_model.append_output_layer(Linear(dim=self.n_actions, activation='none', dropout=0))

        if self.value_model is not None:
            self.value_model.append_output_layer(Linear(dim=1, activation='none', dropout=0))

    @staticmethod
    def from_cli(parameters: Dict) -> 'Policy':
        n_actions = parameters['n_actions']
        value_model = NeuralNetwork.from_cli(parameters['value_model']) if parameters.get('value_model') else None
        action_model = NeuralNetwork.from_cli(parameters['action_model']) if parameters.get('action_model') else None
        action_selector = action_selector_from_cli(parameters['action_selector'])

        return DiscreteAction(n_actions, value_model, action_model, action_selector)
