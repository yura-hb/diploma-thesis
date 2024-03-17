from enum import StrEnum
from typing import Dict

from agents.utils import NeuralNetwork, Phase
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli
from .policy import *


class PolicyEstimationMethod(StrEnum):
    DUELING_ARCHITECTURE = "dueling_architecture"
    INDEPENDENT = "independent"


class ActionPolicy(Policy[Input], metaclass=ABCMeta):

    def __init__(self,
                 actor: NeuralNetwork | None,
                 critic: NeuralNetwork | None,
                 action_selector: ActionSelector,
                 policy_method: PolicyEstimationMethod = PolicyEstimationMethod.INDEPENDENT,
                 noise_parameters: Dict = None):
        super().__init__()

        self.actor = actor
        self.critic = critic
        self.action_selector = action_selector
        self.policy_estimation_method = policy_method
        self.noise_parameters = noise_parameters
        self.run_configuration = None

        self.__post_init__()

    def __post_init__(self):
        if self.noise_parameters is not None:
            self.actor.to_noisy(self.noise_parameters)

            if self.critic is not None:
                self.critic.to_noisy(self.noise_parameters)

    def configure(self, configuration: RunConfiguration):
        self.run_configuration = configuration

        if configuration.compile:
            for model in [self.actor, self.critic]:
                if model is not None:
                    model.compile()

        if self.actor is not None:
            self.actor = self.actor.to(configuration.device)

        if self.critic is not None:
            self.critic = self.critic.to(configuration.device)

    def update(self, phase: Phase):
        self.phase = phase

        for module in [self.critic, self.actor, self.action_selector]:
            if isinstance(module, PhaseUpdatable):
                module.update(phase)

    def forward(self, state: State):
        if state.device != self.run_configuration.device:
            state = state.to(self.run_configuration.device)

        values, actions = self.encode(state)
        values, actions = self.post_encode(state, values, actions)

        return self.__estimate_policy__(values, actions)

    def encode(self, state: State, return_values: bool = True, return_actions: bool = True):
        values, actions = None, None

        if self.actor is not None:
            actions = self.actor(state)

        if self.critic is not None:
            values = self.critic(state)
        else:
            values = actions

        return values, actions

    def select(self, state: State) -> Record:
        value, actions = self.__call__(state)
        value, actions = value.squeeze(), actions.squeeze()
        action, policy = self.action_selector(actions)
        action = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.long)

        info = TensorDict({
            "policy": policy,
            "value": value,
            "actions": actions
        }, batch_size=[])

        return Record(state, action, info, batch_size=[]).detach().cpu()

    def __estimate_policy__(self, value, actions):
        match self.policy_estimation_method:
            case PolicyEstimationMethod.INDEPENDENT:
                return value, actions
            case PolicyEstimationMethod.DUELING_ARCHITECTURE:
                if isinstance(actions, tuple):
                    actions, lengths = actions

                    return value, value + actions - actions.sum(dim=-1) / lengths
                else:
                    return value, value + actions - actions.mean(dim=-1, keepdim=True)
            case _:
                raise ValueError(f"Policy estimation method {self.policy_estimation_method} is not supported")

    @staticmethod
    def base_parameters_from_cli(parameters: Dict):
        return dict(
            actor=NeuralNetwork.from_cli(parameters['actor']),
            critic=NeuralNetwork.from_cli(parameters['critic']) if parameters.get('critic') else None,
            action_selector=action_selector_from_cli(parameters['action_selector']),
            policy_method=PolicyEstimationMethod(parameters['policy_method']) if parameters.get('policy_method')
            else PolicyEstimationMethod.INDEPENDENT,
            noise_parameters=parameters.get('noise')
        )
