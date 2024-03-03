from dataclasses import dataclass
from typing import Dict

from agents.utils.nn import NN
from agents.utils.rl.rl import *


class Reinforce(RLTrainer):

    @dataclass
    class Configuration:
        critic_networks: List[NN]

    def __init__(self,
                 memory: Memory,
                 optimizer: Optimizer,
                 loss: Loss,
                 return_estimator: ReturnEstimator,
                 configuration: Configuration):
        actor_loss = Loss(Loss.Configuration(
            kind='cross_entropy',
            parameters=dict()
        ))

        critics_loss = loss

        super().__init__(memory, actor_loss, optimizer, return_estimator)

        self.critics = None
        self.configuration = configuration

    def configure(self, model: NNModel):
        super().configure(model)

        # configuration.cri
        # TODO: - Instantiate critics

    def train_step(self, model: NNModel):
        try:
            batch, info = self.memory.sample(return_info=True)
            batch: Record | torch.Tensor = torch.squeeze(batch)
        except NotReadyException:
            return

        # Perform policy step

        # Perform critics step

    def store(self, record: Record | List[Record]):
        if isinstance(record, Record):
            raise ValueError('Reinforce does not support single records')

        updated = self.return_estimator.update_returns(record)

        self.memory.store(updated)

    @classmethod
    def from_cli(cls, parameters: Dict, memory: Memory, loss: Loss, optimizer: Optimizer):
        critics = parameters.get('critics', {})
        models = parameters.get('models', {})
        optimizer = parameters.get('optimizer', {})

        return cls(memory, optimizer, loss)
