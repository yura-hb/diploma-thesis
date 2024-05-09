import logging
import torch
from typing import Dict

from dataclasses import dataclass
from enum import auto

from agents.utils import NeuralNetwork
from agents.utils.action import ActionSelector, from_cli as action_selector_from_cli
from .policy import *


@dataclass
class DistributionParameters:
    # Methods from https://github.com/Kchu/DeepRL_PyTorch/tree/master
    @dataclass
    class Kind(StrEnum):
        NO = auto()
        C51 = auto()
        QR_DQN = auto()
        IQN = auto()
        QUOTA = auto()

    kind: Kind = field(default_factory=lambda: DistributionParameters.Kind.NO)
    n_values: int = 1
    min_value: float = 0
    max_value: float = 0

    @property
    def step(self):
        return self.max_value - self.min_value / (self.n_values - 1)

    @staticmethod
    def from_cli(parameters: Dict):
        return DistributionParameters(
            kind=parameters.get('kind', DistributionParameters.Kind.NO),
            n_values=parameters.get('n_values', 1),
            min_value=parameters.get('min_value', 0),
            max_value=parameters.get('max_value', 0)
        )


@dataclass
class PolicyEstimationMethod:
    is_dueling: bool = False
    distribution_parameters: DistributionParameters = field(default_factory=lambda: DistributionParameters())

    @staticmethod
    def from_cli(parameters: Dict):
        return PolicyEstimationMethod(
            is_dueling=parameters.get('is_dueling', False),
            distribution_parameters=DistributionParameters.from_cli(parameters.get('distribution', dict()))
        )


class ActionPolicy(Policy[Input], metaclass=ABCMeta):

    def __init__(self, action_layer, value_layer, **kwargs):
        super().__init__()

        self.__dict__.update(kwargs)

        self.run_configuration = None

        self.action_layer = action_layer()
        self.value_layer = value_layer()
        self.is_first_pass = True

        self.__post_init__()

    def __post_init__(self):
        if self.noise_parameters is not None:
            self.model.to_noisy(self.noise_parameters)

            self.action_layer.to_noisy(self.noise_parameters)
            self.value_layer.to_noisy(self.noise_parameters)

    @property
    def device(self):
        return next(self.parameters()).device

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

        self.action_layer.to(configuration.device)
        self.value_layer.to(configuration.device)

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

        value, action, _ = self.__fetch_values__(output)

        output[Keys.VALUE] = self.value_layer(value)
        output[Keys.ACTIONS] = self.action_layer(action)

        return output

    def post_encode(self, state: State, output: TensorDict):
        output = self.__estimate_distribution_value__(output)
        output = self.__estimate_policy__(output)

        return output

    def select(self, state: State) -> Record:
        output = self.__call__(state)
        value, actions, memory = self.__fetch_values__(output)
        value, actions = value.squeeze(), actions.squeeze()
        value, actions = torch.atleast_1d(value), torch.atleast_1d(actions)

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

    def __estimate_distribution_value__(self, output):
        kind = self.policy_method.distribution_parameters.kind

        match kind:
            case DistributionParameters.Kind.NO:
                return output
            case DistributionParameters.Kind.C51:
                ...
            case DistributionParameters.Kind.IQN:
                ...
            case DistributionParameters.Kind.QR_DQN:
                ...
            case DistributionParameters.Kind.QUOTA:
                ...
            case _:
                raise ValueError('Unknown distributional value estimation')

    def __estimate_policy__(self, output):
        # Replace padding value to such value which yields negative infinity
        min_value = torch.finfo(torch.float32).min

        if self.policy_method.is_dueling:
            actions = output[Keys.ACTIONS]
            value = output.get(Keys.VALUE, actions)

            if isinstance(actions, tuple):
                actions, lengths = actions
                mask = torch.isfinite(actions)
                means = torch.nan_to_num(actions, neginf=0.0).sum(dim=-1, keepdim=True) / torch.atleast_2d(lengths).T

                # Here we suppose that pad value ios neginf
                output[Keys.ACTIONS] = value + actions - means
                output[Keys.ACTIONS][~mask] = min_value
                output[Keys.ACTIONS] = torch.nan_to_num(output[Keys.ACTIONS], neginf=min_value)
            else:
                output[Keys.ACTIONS] = value + actions - actions.mean(dim=-1, keepdim=True)

        actions = output[Keys.ACTIONS]

        if isinstance(actions, tuple):
            output[Keys.ACTIONS] = torch.nan_to_num(actions[0], neginf=min_value)

        return output

    def with_action_selector(self, action_selector: ActionSelector):
        self.action_selector = action_selector

        if 'phase' in self.__dict__.keys():
            self.update(self.phase)

    @staticmethod
    def __fetch_values__(output: TensorDict):
        actions = output[Keys.ACTIONS]
        values = output.get(Keys.VALUE, actions)
        memory = output.get(Keys.MEMORY, None)

        return values, actions, memory

    @staticmethod
    def base_parameters_from_cli(parameters: Dict):
        return dict(
            actions_key=parameters.get('action_key', Keys.ACTIONS),
            values_key=parameters.get('values_key', Keys.VALUE),
            memory_key=parameters.get('memory_key', Keys.MEMORY),
            model=NeuralNetwork.from_cli(parameters['model']),
            action_selector=action_selector_from_cli(parameters['action_selector']),
            policy_method=PolicyEstimationMethod.from_cli(parameters.get('estimator', dict())),
            noise_parameters=parameters.get('noise')
        )
