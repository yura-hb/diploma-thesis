
from dataclasses import dataclass
from typing import Dict

from agents.utils.memory import NotReadyException
from .episodic import *


@dataclass
class Configuration:
    value_loss: Loss
    policy_step_ratio: float
    # A ratio for rolling back policy updates from
    # "Dynamic job-shop scheduling using graph reinforcement learning with auxiliary strategy" paper
    rollback_ratio: float
    entropy_regularization: float
    epochs: int

    @staticmethod
    def from_cli(parameters: Dict):
        return Configuration(
            value_loss=Loss.from_cli(parameters['value_loss']),
            policy_step_ratio=parameters.get('policy_step_ratio', 1.0),
            rollback_ratio=parameters.get('rollback_ratio', 1.0),
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

        advantages = batch.info[Record.ADVANTAGE_KEY]
        value, logits = model.predict(batch.state)
        distribution = torch.distributions.Categorical(logits=logits)

        loss = 0

        weights = distribution.log_prob(batch.action) - batch.info[Record.POLICY_KEY]
        weights = torch.exp(weights)

        ratio = self.configuration.policy_step_ratio
        clipped_weights = torch.clamp(weights, 1 - ratio, 1 + ratio)

        advantages = torch.min(weights * advantages, clipped_weights * advantages)

        loss = -torch.mean(advantages)
        loss -= self.configuration.entropy_regularization * distribution.entropy().mean()

        for critic in self.configuration.critics:
            critic_loss = critic.loss(critic.neural_network(batch.state), batch.reward)
            loss += critic_loss

            self.record_loss(critic_loss, key='critic')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.record_loss(loss, key='policy')

    @property
    def critics(self):
        return self.configuration.critics

    @classmethod
    def from_cli(cls,
                 parameters: Dict,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator):
        return cls(memory, optimizer, loss, return_estimator, Configuration.from_cli(parameters))
