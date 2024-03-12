from dataclasses import dataclass
from typing import Dict

from agents.utils.memory import NotReadyException
from agents.utils.nn import NeuralNetwork
from agents.utils.nn.layers import Linear
from .rl import *


@dataclass
class Critic:
    neural_network: NeuralNetwork
    optimizer: Optimizer
    loss: Loss

    @staticmethod
    def from_cli(parameters: Dict):
        return Critic(
            neural_network=NeuralNetwork.from_cli(parameters['neural_network']),
            optimizer=Optimizer.from_cli(parameters['optimizer']),
            loss=Loss.from_cli(parameters['loss'])
        )


@dataclass
class Configuration:
    critics: List[Critic]

    @staticmethod
    def from_cli(parameters: Dict):
        return Configuration(
            critics=[Critic.from_cli(critic) for critic in parameters.get('critics', [])]
        )


class Reinforce(RLTrainer):

    def __init__(self, configuration: Configuration, *args, **kwargs):
        super().__init__(*args, is_episodic=True, **kwargs)

        self.is_critics_configured = False
        self.configuration = configuration

    def configure(self, model: Policy):
        super().configure(model)

        layer = Linear(1, 'none', dropout=None)

        for critic in self.configuration.critics:
            critic.neural_network.append_output_layer(layer)

    def __train__(self, model: Policy):
        try:
            batch, index = self.storage.sample(update_returns=False)
        except NotReadyException:
            return

        with torch.no_grad():
            if len(self.critics) == 0:
                baseline = torch.zeros(batch.reward.shape, device=batch.reward.device)
            else:
                baseline = torch.stack([critic.neural_network(batch.state) for critic in self.critics], dim=0)
                baseline = torch.mean(baseline, dim=0)
                baseline = torch.squeeze(baseline)

        # Perform policy step
        loss = self.loss(model(batch.state)[1], batch.action)

        if loss.numel() == 1:
            raise ValueError('Loss should not have reduction to single value')

        loss = (batch.reward - baseline) * loss
        loss = loss.mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.record_loss(loss, key='policy')

        # Perform critics step
        if not self.is_critics_configured:
            self.is_critics_configured = True

            for critic in self.critics:
                critic.optimizer.connect(critic.neural_network.parameters())

        for index, critic in enumerate(self.critics):
            critic_loss = critic.loss(torch.squeeze(critic.neural_network(batch.state)), batch.reward)
            critic.optimizer.zero_grad()
            critic_loss.backward()
            critic.optimizer.step()
            self.record_loss(critic_loss, key=f'critic_{index}')

    def compile(self):
        if not self.is_configured:
            return

        for critic in self.critics:
            critic.neural_network.compile()

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
        schedule = TrainSchedule.from_cli(parameters)
        configuration = Configuration.from_cli(parameters)

        return cls(configuration=configuration,
                   memory=memory,
                   optimizer=optimizer,
                   loss=loss,
                   return_estimator=return_estimator,
                   train_schedule=schedule)
