
import tensordict

from agents.utils.rl.rl import *

import torch

from dataclasses import dataclass
from typing import List
from agents.utils.nn import NNCLI


class Reinforce(RLTrainer):

    @dataclass
    class Configuration:
        critic_networks: List[NNCLI]

    def __init__(self,
                 memory: Memory,
                 optimizer: OptimizerCLI,
                 loss: LossCLI,
                 configuration: Configuration):
        super().__init__(memory, loss, optimizer)

        self.critics = None
        self.configuration = configuration

    def configure(self, model: NNModel):
        super().configure(model)

        pass

    def train_step(self, model: NNModel):
        pass

    @staticmethod
    def from_cli(parameters, memory: Memory, loss: LossCLI, optimizer: OptimizerCLI):
        pass
