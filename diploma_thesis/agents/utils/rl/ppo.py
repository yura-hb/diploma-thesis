
from dataclasses import dataclass
from typing import Dict

from agents.utils.memory import NotReadyException
from .rl import *


@dataclass
class Configuration:
    value_loss: Loss
    policy_step_ratio: float
    entropy_regularization: float
    update_advantages: bool
    epochs: int

    @staticmethod
    def from_cli(parameters: Dict):
        return Configuration(
            value_loss=Loss.from_cli(parameters['value_loss']),
            policy_step_ratio=parameters.get('policy_step_ratio', 1.0),
            entropy_regularization=parameters.get('entropy_regularization', 0.0),
            update_advantages=parameters.get('update_advantages', True),
            epochs=parameters.get('epochs', 1)
        )


class PPO(RLTrainer):

    def __init__(self, configuration: Configuration, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_critics_configured = False
        self.configuration = configuration
        self.episodes = 0

    def configure(self, model: Policy):
        super().configure(model)

    def __train__(self, model: Policy):
        for i in range(self.configuration.epochs):
            self.__step__(model)

    def __step__(self, model: Policy):
        try:
            batch = self.memory.sample(return_info=False)
            batch: Record | torch.Tensor = torch.squeeze(batch)
        except NotReadyException:
            return

        if self.configuration.update_advantages:
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

    @classmethod
    def from_cli(cls,
                 parameters: Dict,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator):
        schedule = TrainSchedule.from_cli(parameters)
        configuration = Configuration.from_cli(parameters)

        return cls(configuration=configuration,
                   memory=memory,
                   optimizer=optimizer,
                   loss=loss,
                   return_estimator=return_estimator,
                   train_schedule=schedule)
