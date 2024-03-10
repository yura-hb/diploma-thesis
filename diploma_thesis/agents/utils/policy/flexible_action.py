from enum import StrEnum
from typing import Dict

from agents.utils import NeuralNetwork, Phase
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli
from .policy import *


class PolicyEstimationMethod(StrEnum):
    DUELING_ARCHITECTURE = "dueling_architecture"
    INDEPENDENT = "independent"


class FlexibleAction(Policy[Input]):

    def __init__(self,
                 value_model: NeuralNetwork,
                 action_model: NeuralNetwork,
                 action_selector: ActionSelector,
                 policy_method: PolicyEstimationMethod = PolicyEstimationMethod.INDEPENDENT):
        super().__init__()

        self.value_model = value_model
        self.action_model = action_model
        self.action_selector = action_selector
        self.policy_estimation_method = policy_method

    def update(self, phase: Phase):
        self.phase = phase

        for module in [self.value_model, self.action_model, self.action_selector]:
            if isinstance(module, PhaseUpdatable):
                module.update(phase)

    def __get_values__(self, state):
        return self.value_model(state)

    def __get_actions__(self, state):
        return self.action_model(state)

    def __call__(self, state: State, parameters: Input) -> Record:
        values, actions = self.predict(state)
        values, actions = values.squeeze(), actions.squeeze()
        action, policy = self.action_selector(actions)
        action = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.long)

        info = TensorDict({
            "policy": policy,
            "values": values.detach().clone(),
            "actions": actions.detach().clone()
        }, batch_size=[])

        return Record(state, action, info, batch_size=[])

    def predict(self, state: State):
        actions = torch.tensor(0, dtype=torch.long)

        if self.action_model is not None:
            actions = self.__get_actions__(state)

        if self.value_model is not None:
            values = self.__get_values__(state)
        else:
            values = actions

        match self.policy_estimation_method:
            case PolicyEstimationMethod.INDEPENDENT:
                return values, actions
            case PolicyEstimationMethod.DUELING_ARCHITECTURE:
                return values, values + (actions - actions.mean(dim=1, keepdim=True))
            case _:
                raise ValueError(f"Policy estimation method {self.policy_estimation_method} is not supported")

    @classmethod
    def from_cli(cls, parameters: Dict) -> 'Policy':
        return FlexibleAction(**cls.base_parameters_from_cli(parameters))

    @staticmethod
    def base_parameters_from_cli(parameters: Dict):
        return dict(
            value_model=NeuralNetwork.from_cli(parameters['value_model']) if parameters.get('value_model') else None,
            action_model=NeuralNetwork.from_cli(parameters['action_model']) if parameters.get('action_model') else None,
            action_selector=action_selector_from_cli(parameters['action_selector']),
            policy_method=PolicyEstimationMethod(parameters['policy_method']) if parameters.get('policy_method')
            else PolicyEstimationMethod.INDEPENDENT
        )
