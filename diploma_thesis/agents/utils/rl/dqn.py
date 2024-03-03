from dataclasses import dataclass
from typing import Dict

import tensordict

from agents.utils.memory import NotReadyException
from agents.utils.rl.rl import *


class DeepQTrainer(RLTrainer):
    @dataclass
    class Configuration:
        decay: float = 0.99
        update_steps: int = 10
        prior_eps: float = 1e-6

        @staticmethod
        def from_cli(parameters: Dict):
            return DeepQTrainer.Configuration(
                decay=parameters.get('decay', 0.99),
                update_steps=parameters.get('update_steps', 10),
                prior_eps=parameters.get('prior_eps', 1e-6)
            )

    def __init__(self,
                 memory: Memory,
                 optimizer: Optimizer,
                 loss: Loss,
                 return_estimator: ReturnEstimator,
                 configuration: Configuration):
        super().__init__(memory, loss, optimizer, return_estimator)

        self.target_model = None
        self.configuration = configuration

    def configure(self, model: Policy):
        super().configure(model)

        self.target_model = model.clone()

    def train_step(self, model: Policy):
        try:
            batch, info = self.memory.sample(return_info=True)
            batch: Record | torch.Tensor = torch.squeeze(batch)
        except NotReadyException:
            return

        with torch.no_grad():
            q_values, td_error = self.estimate_q(model, batch)

        values = model.predict(batch.state)
        loss = self.loss(values, q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.record_loss(loss)

        if self.optimizer.step_count % self.configuration.update_steps == 0:
            self.target_model.copy_parameters(model, self.configuration.decay)

        with torch.no_grad():
            td_error += self.configuration.prior_eps

            self.memory.update_priority(info['index'], td_error)

    def estimate_q(self, model: Policy, batch: Record | tensordict.TensorDictBase):
        # Note:
        # The idea is that we compute the Q-values only for performed actions. Other actions wouldn't be updated,
        # because there will be zero loss and so zero gradient
        q_values = model.predict(batch.next_state)
        orig_q = q_values.clone()[range(batch.shape[0]), batch.action]

        target = self.target_model.predict(batch.next_state)
        target = target.max(dim=1).values

        q = batch.reward + self.return_estimator.discount_factor * target * (1 - batch.done)
        q_values[range(batch.shape[0]), batch.action] = q

        td_error = torch.square(orig_q - q)

        return q_values, td_error

    def store(self, record: Record | List[Record]):
        if isinstance(record, Record):
            self.memory.store(record.view(-1))
            return

        estimated = self.return_estimator.update_returns(record)

        self.memory.store(estimated)

    @classmethod
    def from_cli(cls,
                 parameters,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator):
        return cls(memory, optimizer, loss, return_estimator, DeepQTrainer.Configuration.from_cli(parameters))
