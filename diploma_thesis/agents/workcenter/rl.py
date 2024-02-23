from typing import Dict

from agents.base.rl_agent import RLAgent
from agents.utils.rl import from_cli as rl_trainer_from_cli
from environment import WorkCenterKey
from .model import NNWorkCenterModel, from_cli as model_from_cli
from .state import from_cli as state_encoder_from_cli


class RLWorkCenter(RLAgent[WorkCenterKey]):

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])
        trainer = rl_trainer_from_cli(parameters['trainer'])

        assert isinstance(model, NNWorkCenterModel), f"Model must conform to NNModel"

        return RLWorkCenter(model, encoder, trainer)
