import tensordict

from .rl import *

import torch

from dataclasses import dataclass
from typing import Dict


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

        self.target_model = model.copy()

    def train_step(self, model: NNModel):
        batch, info = self.memory.sample(return_info=True)
        batch: Record | torch.Tensor = torch.squeeze(batch)

        # Note:
        # The idea is that we compute the Q-values only for performed actions. Other actions wouldn't be updated,
        # because there will be zero loss and so zero gradient

        with torch.no_grad():
            q_values, td_error = self.estimate_q(batch)

        if not self.optimizer.is_connected:
            self.optimizer.connect(model.parameters())

        values = model.values(batch.state)
        loss = self.loss(values, q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.optimizer.step_count % self.configuration.update_steps == 0:
            self.target_model.copy_parameters(self.model, self.configuration.decay)

        with torch.no_grad():
            td_error += self.configuration.prior_eps

            self.memory.update_priority(info['index'], td_error)

    def estimate_q(self, batch: Record | tensordict.TensorDictBase):
        q_values = self.model.values(batch.next_state)
        orig_q = q_values.clone()[range(batch.shape[0]), batch.action]

        target = self.target_model.values(batch.next_state)
        target = target.max(dim=1).values

        q = batch.reward + self.configuration.gamma * target * (1 - batch.done)
        q_values[range(batch.shape[0]), batch.action] = q

        return q_values, orig_q

    @staticmethod
    def from_cli(parameters, memory: Memory, loss: LossCLI, optimizer: OptimizerCLI):
        return DeepQTrainer(memory, optimizer, loss, DeepQTrainer.Configuration.from_cli(parameters))
