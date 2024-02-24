from typing import Dict

from agents.base.marl_agent import MARLAgent
from agents.utils.rl import from_cli as rl_trainer_from_cli
from environment import MachineKey, ShopFloor
from .model import NNMachineModel, from_cli as model_from_cli
from .state import from_cli as state_encoder_from_cli


class MARLMachine(MARLAgent[MachineKey]):

    def iterate_keys(self, shop_floor: ShopFloor):
        for work_center in shop_floor.work_centers:
            for machine in work_center.machines:
                yield machine.key

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])
        trainer = rl_trainer_from_cli(parameters['trainer'])

        is_model_distributed = parameters.get('is_model_distributed', True)

        assert isinstance(model, NNMachineModel), f"Model must conform to NNModel"

        return MARLMachine(model, encoder, trainer, is_model_distributed)
