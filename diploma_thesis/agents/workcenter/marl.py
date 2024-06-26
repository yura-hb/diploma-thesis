from typing import Dict

from agents.base.marl_agent import MARLAgent
from agents.utils.rl import from_cli as rl_trainer_from_cli
from environment import WorkCenterKey, ShopFloor
from .model import DeepPolicyWorkCenterModel, from_cli as model_from_cli
from .state import from_cli as state_encoder_from_cli


class MARLWorkCenter(MARLAgent[WorkCenterKey]):

    def iterate_keys(self, shop_floor: ShopFloor):
        for work_center in shop_floor.work_centers:
            yield work_center.key

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['policy'])
        encoder = state_encoder_from_cli(parameters['encoder'])
        trainer = rl_trainer_from_cli(parameters['trainer'])

        is_model_distributed = parameters.get('is_model_distributed', True)
        is_training_centralized = parameters.get('is_training_centralized', True)

        assert isinstance(model, DeepPolicyWorkCenterModel), f"Model must conform to NNModel"

        return MARLWorkCenter(model, encoder, trainer, is_model_distributed, is_training_centralized=is_training_centralized)
