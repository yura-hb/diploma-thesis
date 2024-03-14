from typing import Dict

from agents.base.rl_agent import RLAgent
from agents.utils.rl import from_cli as rl_trainer_from_cli
from agents.utils.run_configuration import RunConfiguration
from environment import MachineKey
from .model import DeepPolicyMachineModel, from_cli as model_from_cli
from .state import from_cli as state_encoder_from_cli


class RLMachine(RLAgent[MachineKey]):

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])
        trainer = rl_trainer_from_cli(parameters['trainer'])
        configuration = RunConfiguration.from_cli(parameters)

        assert isinstance(model, DeepPolicyMachineModel), f"Model must conform to NNModel"

        return RLMachine(model, encoder, trainer, configuration)
