
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
    epochs: int

    @staticmethod
    def from_cli(parameters: Dict):
        return Configuration(
            value_loss=Loss.from_cli(parameters['value_loss']),
            sample_ratio=parameters.get('sample_ratio', 0.5),
            policy_step_ratio=parameters.get('policy_step_ratio', 1.0),
            entropy_regularization=parameters.get('entropy_regularization', 0.0),
            update_advantages=parameters.get('update_advantages', True),
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

    def configure(self, model: Policy):
        super().configure(model)

    def __train__(self, model: Policy):
        try:
            batch, info = self.storage.sample(update_returns=self.configuration.update_advantages)
        except NotReadyException:
            return

        for i in range(self.configuration.epochs):
            mask = torch.zeros(batch.batch_size).uniform_() < self.configuration.sample_ratio

            sub_batch = batch[mask]

            if isinstance(batch.state, GraphState):
                sub_batch.state.graph = Graph(Batch.from_data_list(batch.state.graph.data[mask]))
                sub_batch.next_state.graph = Graph(Batch.from_data_list(batch.next_state.graph.data[mask]))

            self.__step__(sub_batch, model)

    def __step__(self, batch: Record, model: Policy):
        advantages = batch.info[Record.ADVANTAGE_KEY]
        value, logits = model.predict(batch.state)
        value = value[torch.arange(batch.shape[0]), batch.action]
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
