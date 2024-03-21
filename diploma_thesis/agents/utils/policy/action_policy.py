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

    def configure(self, configuration: RunConfiguration):
        self.run_configuration = configuration

        if configuration.compile:
            for model in [self.actor, self.critic]:
                if model is not None:
                    model.compile()

        if self.model is not None:
            self.model = self.model.to(configuration.device)

    def update(self, phase: Phase):
        self.phase = phase

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
        values, actions = self.__fetch_values_and_actions__(output)

        return self.__estimate_policy__(values, actions)

    def select(self, state: State) -> Record:
        value, actions = self.__call__(state)
        value, actions = value.squeeze(), actions.squeeze()
        action, policy = self.action_selector(actions)
        action = action if torch.is_tensor(action) else torch.tensor(action, dtype=torch.long)

        info = TensorDict({
            Keys.POLICY: policy,
            Keys.VALUE: value,
            Keys.ACTIONS: actions
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
    def __fetch_values_and_actions__(output: TensorDict):
        actions = output[Keys.ACTIONS]
        values = output.get(Keys.VALUE, actions)

        return values, actions

    @staticmethod
    def base_parameters_from_cli(parameters: Dict):
        return dict(
            model=NeuralNetwork.from_cli(parameters['model']),
            action_selector=action_selector_from_cli(parameters['action_selector']),
            policy_method=PolicyEstimationMethod(parameters['policy_method']) if parameters.get('policy_method')
            else PolicyEstimationMethod.INDEPENDENT,
            noise_parameters=parameters.get('noise')
        )
