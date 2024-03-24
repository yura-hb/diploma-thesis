import logging
import torch
from typing import Dict

from agents.utils import NeuralNetwork
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli
from .policy import *


class PolicyEstimationMethod(StrEnum):
    DUELING_ARCHITECTURE = "dueling_architecture"
    INDEPENDENT = "independent"


class ActionPolicy(Policy[Input], metaclass=ABCMeta):

    def __init__(self,
                 model: NeuralNetwork | None,
                 action_selector: ActionSelector,
                 policy_method: PolicyEstimationMethod = PolicyEstimationMethod.INDEPENDENT,
                 noise_parameters: Dict = None):
        super().__init__()

        self.model = model
        self.action_selector = action_selector
        self.policy_estimation_method = policy_method
        self.noise_parameters = noise_parameters
        self.run_configuration = None

        self.__post_init__()

    def __post_init__(self):
        if self.noise_parameters is not None:
            self.model.to_noisy(self.noise_parameters)

    def with_logger(self, logger: logging.Logger):
        super().with_logger(logger)

        for module in [self.model, self.action_selector]:
            if isinstance(module, Loggable):
                module.with_logger(logger)

        return self

    def configure(self, configuration: RunConfiguration):
        self.run_configuration = configuration

        if configuration.compile:
            for model in [self.actor, self.critic]:
                if model is not None:
                    model.compile()

        if self.model is not None:
            self.model = self.model.to(configuration.device)

    def update(self, phase: Phase):
        super().update(phase)

        for module in [self.model, self.action_selector]:
            if isinstance(module, PhaseUpdatable):
                module.update(phase)

    def forward(self, state: State):
        if state.device != self.run_configuration.device:
            state = state.to(self.run_configuration.device)

        output = self.encode(state)

        return self.post_encode(state, output)

    def encode(self, state: State):
        output = self.model(state)

        return output

    def post_encode(self, state: State, output: TensorDict):
        return self.__estimate_policy__(output)

    def select(self, state: State) -> Record:
        output = self.__call__(state)
        value, actions, memory = self.__fetch_values__(output)
        value, actions = value.squeeze(), actions.squeeze()

        if memory is not None:
            for key, item in memory.items(include_nested=True, leaves_only=True):
                memory[key] = item.squeeze()

        action, policy = self.action_selector(actions)
        action = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.long)

        info = TensorDict({
            Keys.POLICY: policy,
            Keys.VALUE: value,
            Keys.ACTIONS: actions
        }, batch_size=[])

        return Record(state, action, memory, info, batch_size=[]).detach().cpu()

    def __estimate_policy__(self, output):
        match self.policy_estimation_method:
            case PolicyEstimationMethod.INDEPENDENT:
                return output
            case PolicyEstimationMethod.DUELING_ARCHITECTURE:
                actions = output[Keys.ACTIONS]
                value = output.get(Keys.VALUE, actions)

                if isinstance(actions, tuple):
                    actions, lengths = actions

                    output[Keys.ACTIONS] = value + actions - actions.sum(dim=-1) / lengths

                    return output
                else:
                    output[Keys.ACTIONS] = value + actions - actions.mean(dim=-1, keepdim=True)

                    return output
            case _:
                raise ValueError(f"Policy estimation method {self.policy_estimation_method} is not supported")

    @staticmethod
    def __fetch_values__(output: TensorDict):
        actions = output[Keys.ACTIONS]
        values = output.get(Keys.VALUE, actions)
        memory = output.get(Keys.MEMORY, None)

        return values, actions, memory

    @staticmethod
    def base_parameters_from_cli(parameters: Dict):
        return dict(
            model=NeuralNetwork.from_cli(parameters['model']),
            action_selector=action_selector_from_cli(parameters['action_selector']),
            policy_method=PolicyEstimationMethod(parameters['policy_method']) if parameters.get('policy_method')
            else PolicyEstimationMethod.INDEPENDENT,
            noise_parameters=parameters.get('noise')
        )
