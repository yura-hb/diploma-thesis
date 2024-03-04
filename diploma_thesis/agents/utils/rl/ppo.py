
from dataclasses import dataclass
from typing import Dict

import torch

from agents.utils.memory import NotReadyException
from .episodic import *


@dataclass
class Configuration:
    value_loss: Loss
    policy_step_ratio: float
    entropy_regularization: float
    epochs: int

    @staticmethod
    def from_cli(parameters: Dict):
        return Configuration(
            value_loss=Loss.from_cli(parameters['value_loss']),
            policy_step_ratio=parameters.get('policy_step_ratio', 1.0),
            entropy_regularization=parameters.get('entropy_regularization', 0.0),
            epochs=parameters.get('epochs', 1)
        )


class PPO(EpisodicTrainer):

    def __init__(self,
                 memory: Memory,
                 optimizer: Optimizer,
                 loss: Loss,
                 return_estimator: ReturnEstimator,
                 configuration: Configuration):
        super().__init__(memory, loss, optimizer, return_estimator)

        self.is_critics_configured = False
        self.configuration = configuration
        self.episodes = 0

    def train_step(self, model: Policy):
        for i in range(self.configuration.epochs):
            self._train_step(model)

    def _train_step(self, model: Policy):
        try:
            batch = self.memory.sample(return_info=False)
            batch: Record | torch.Tensor = torch.squeeze(batch)
        except NotReadyException:
            return

        with torch.no_grad():
            batch = self.__estimate_advantage__(batch)

        advantages = batch.info[Record.ADVANTAGE_KEY]
        value, logits = model.predict(batch.state)
        distribution = torch.distributions.Categorical(logits=logits)

        loss = 0

        action_probs = batch.info[Record.POLICY_KEY][torch.arange(batch.shape[0]), batch.action.view(-1)]

        weights = distribution.log_prob(batch.action).view(-1) - torch.log(action_probs)
        weights = torch.exp(weights)

        ratio = self.configuration.policy_step_ratio
        clipped_weights = torch.clamp(weights, 1 - ratio, 1 + ratio)

        advantages = torch.min(weights * advantages, clipped_weights * advantages)

        loss = -torch.mean(advantages)
        loss += self.configuration.value_loss(value.view(-1), batch.info[Record.RETURN_KEY])
        loss -= self.configuration.entropy_regularization * distribution.entropy().mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.record_loss(loss)

    def __estimate_advantage__(self, batch: Record | torch.Tensor) -> torch.Tensor:
        result = []

        for element in batch.info['episode'].unique():
            episode = batch[batch.info['episode'] == element].unbind(dim=0)
            updated = self.return_estimator.update_returns(episode)
            updated = torch.stack(updated, dim=0)
            result.append(updated)

        return torch.cat(result, dim=0)

    def store(self, record: Record | List[Record]):
        if isinstance(record, Record):
            raise ValueError('Reinforce does not support single records')

        updated = torch.stack(record, dim=0)
        updated.info['episode'] = torch.full(updated.reward.shape, self.episodes, device=updated.reward.device)

        self.episodes += 1

        self.memory.store(updated)

    @classmethod
    def from_cli(cls,
                 parameters: Dict,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator):
        return cls(memory, optimizer, loss, return_estimator, Configuration.from_cli(parameters))
