
from dataclasses import dataclass
from typing import Dict

import torch

from agents.utils.memory import NotReadyException
from .rl import *


@dataclass
class Configuration:
    value_loss: Loss
    sample_ratio: float
    policy_step_ratio: float
    entropy_regularization: float
    update_advantages: bool
    rollback_ratio: float
    epochs: int

    @staticmethod
    def from_cli(parameters: Dict):
        return Configuration(
            value_loss=Loss.from_cli(parameters['value_loss']),
            sample_ratio=parameters.get('sample_ratio', 0.5),
            policy_step_ratio=parameters.get('policy_step_ratio', 1.0),
            entropy_regularization=parameters.get('entropy_regularization', 0.0),
            update_advantages=parameters.get('update_advantages', True),
            rollback_ratio=parameters.get('rollback_ratio', 0.1),
            epochs=parameters.get('epochs', 1)
        )


class PPO(RLTrainer):

    def __init__(self, configuration: Configuration, *args, **kwargs):
        super().__init__(*args, is_episodic=True, **kwargs)

        self.is_critics_configured = False
        self.configuration = configuration
        self.episodes = 0

    @property
    def is_episodic(self):
        return True

    def __train__(self, model: Policy):
        try:
            for slice in self.storage.sample_minibatches(update_returns=self.configuration.update_advantages,
                                                         device=self.run_configuration.device,
                                                         n=self.configuration.epochs,
                                                         sample_ratio=self.configuration.sample_ratio):
                self.__step__(slice, model)
        except NotReadyException:
            return

    def __step__(self, batch: Record, model: Policy):
        range = torch.arange(batch.shape[0], device=self.run_configuration.device)
        advantages = batch.info[Record.ADVANTAGE_KEY]
        value, logits = model(batch.state)
        value = value[range, batch.action]
        distribution = torch.distributions.Categorical(logits=logits)

        loss = 0

        action_probs = batch.info[Record.POLICY_KEY][range, batch.action.view(-1)]

        weights = distribution.log_prob(batch.action).view(-1) - torch.log(action_probs)
        weights = torch.exp(weights)

        step = self.configuration.policy_step_ratio
        phi = self.configuration.rollback_ratio
        rollback_value = - self.configuration.rollback_ratio * weights

        clipped_weights = torch.clamp(weights,
                                      rollback_value + (1 + phi) * (1 - step),
                                      rollback_value + (1 + phi) * (1 + step))

        advantages = torch.min(weights * advantages, clipped_weights * advantages)

        loss = -torch.mean(advantages)
        loss += self.configuration.value_loss(value.view(-1), batch.info[Record.RETURN_KEY])
        loss -= self.configuration.entropy_regularization * distribution.entropy().mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.record_loss(loss)

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

