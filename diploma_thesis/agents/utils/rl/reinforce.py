from dataclasses import dataclass
from typing import List

import torch

from typing import Dict

from agents.utils.nn import NNCLI
from agents.utils.rl.rl import *


class Reinforce(RLTrainer):

    @dataclass
    class Configuration:
        critic_networks: List[NNCLI]

    def __init__(self,
                 memory: Memory,
                 optimizer: OptimizerCLI,
                 loss: LossCLI,
                 configuration: Configuration):
        actor_loss = LossCLI(LossCLI.Configuration(
            kind='cross_entropy',
            parameters=dict()
        ))

        critics_loss = loss

        super().__init__(memory, actor_loss, optimizer)

        self.critics = None
        self.configuration = configuration

    def configure(self, model: NNModel):
        super().configure(model)

        # TODO: - Instantiate critics

    def train_step(self, model: NNModel):
        try:
            batch, info = self.memory.sample(return_info=True)
            batch: Record | torch.Tensor = torch.squeeze(batch)
        except NotReadyException:
            return

        # Perform policy step

        # Perform critics step

    @classmethod
    def from_cli(cls, parameters: Dict, memory: Memory, loss: LossCLI, optimizer: OptimizerCLI):
        critics = parameters.get('critics', {})
        models = parameters.get('models', {})
        optimizer = parameters.get('optimizer', {})

        return cls(memory, optimizer, loss, )
