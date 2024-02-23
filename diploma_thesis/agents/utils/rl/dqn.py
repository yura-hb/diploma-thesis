from dataclasses import dataclass
from typing import Dict

import tensordict
import torch

from utils import filter
from agents.utils.rl.rl import *


class DeepQTrainer(RLTrainer):
    @dataclass
    class Configuration:
        gamma: float
        decay: float = 0.99
        update_steps: int = 10
        prior_eps: float = 1e-6

        @staticmethod
        def from_cli(parameters: Dict):
            return DeepQTrainer.Configuration(
                gamma=parameters['gamma'],
                decay=parameters.get('decay', 0.99),
                update_steps=parameters.get('update_steps', 10),
                prior_eps=parameters.get('prior_eps', 1e-6)
            )

    def __init__(self,
                 memory: Memory,
                 optimizer: OptimizerCLI,
                 loss: LossCLI,
                 configuration: Configuration):
        super().__init__(memory, loss, optimizer)

        self.target_model = None
        self.configuration = configuration

    def configure(self, model: NNModel):
        super().configure(model)

        self.target_model = model.clone()

    @filter(lambda self, *args: len(self.memory) > 0)
    def train_step(self, model: NNModel):
        batch, info = self.memory.sample(return_info=True)
        batch: Record | torch.Tensor = torch.squeeze(batch)

        with torch.no_grad():
            q_values, td_error = self.estimate_q(model, batch)

        values = model.values(batch.state)
        loss = self.loss(values, q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.optimizer.step_count % self.configuration.update_steps == 0:
            self.target_model.copy_parameters(model, self.configuration.decay)

        with torch.no_grad():
            td_error += self.configuration.prior_eps

            self.memory.update_priority(info['index'], td_error)

    def estimate_q(self, model: NNModel, batch: Record | tensordict.TensorDictBase):
        # Note:
        # The idea is that we compute the Q-values only for performed actions. Other actions wouldn't be updated,
        # because there will be zero loss and so zero gradient
        q_values = model.values(batch.next_state)
        orig_q = q_values.clone()[range(batch.shape[0]), batch.action]

        target = self.target_model.values(batch.next_state)
        target = target.max(dim=1).values

        q = batch.reward + self.configuration.gamma * target * (1 - batch.done)
        q_values[range(batch.shape[0]), batch.action] = q

        td_error = torch.square(orig_q - q)

        return q_values, td_error

    @classmethod
    def from_cli(cls, parameters, memory: Memory, loss: LossCLI, optimizer: OptimizerCLI):
        return cls(memory, optimizer, loss, DeepQTrainer.Configuration.from_cli(parameters))
