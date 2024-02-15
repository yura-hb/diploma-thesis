
import torch

from .machine import *
from typing import Dict
from agents.utils import TrainingPhase, OptimizerCLI, LossCLI

from dataclasses import dataclass

from utils import filter


class DeepQAgent(Machine):

    @dataclass
    class Configuration:
        gamma: float
        decay: float = 0.99
        update_steps: int = 10
        prior_eps: float = 1e-6

        @staticmethod
        def from_cli(parameters: Dict):
            return DeepQAgent.Configuration(
                gamma=parameters['gamma'],
                decay=parameters.get('decay', 0.99),
                update_steps=parameters.get('update_steps', 10),
                prior_eps=parameters.get('prior_eps', 1e-6)
            )

    def __init__(self,
                 model: NNMachineModel,
                 state_encoder: StateEncoder,
                 memory: Memory,
                 optimizer: OptimizerCLI,
                 loss: LossCLI,
                 parameters: Configuration):
        super().__init__(model, state_encoder, memory)

        self.parameters = parameters
        self.loss = loss
        self.optimizer = optimizer
        self.target_model = None

    def __post_init__(self):
        self.loss = torch.nn.SmoothL1Loss()

    @property
    def is_trainable(self):
        return True

    @filter(lambda self: self.phase == TrainingPhase())
    @filter(lambda self: len(self.memory) > 0)
    def train_step(self):
        batch, info = self.memory.sample(return_info=True)
        batch = torch.squeeze(batch)

        # Note:
        # The idea is that we compute the Q-values only for performed actions. Other actions wouldn't be updated,
        # because there will be zero loss and so zero gradient

        with torch.no_grad():
            q_values = self.model.values(batch.next_state)
            orig_q = q_values.clone()[range(batch.shape[0]), batch.action]

            target = self.target_model.values(batch.next_state)
            target = target.max(dim=1).values

            q = batch.reward + self.parameters.gamma * target * (1 - batch.done)
            q_values[range(batch.shape[0]), batch.action] = q

        if not self.optimizer.is_connected:
            self.optimizer.connect(self.model.parameters())

        values = self.model.values(batch.state)
        loss = self.loss(values, q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.optimizer.step_count % self.parameters.update_steps == 0:
            self.target_model.copy_parameters(self.model, self.parameters.decay)

        with torch.no_grad():
            td_error = torch.square(q - orig_q) + self.parameters.prior_eps

            self.memory.update_priority(info['index'], td_error)

    def schedule(self, parameters):
        result = super().schedule(parameters)

        if self.target_model is None:
            self.target_model = self.model.clone()

        return result

    @staticmethod
    def from_cli(parameters: Dict):
        model = model_from_cli(parameters['model'])
        encoder = state_encoder_from_cli(parameters['encoder'])
        memory = memory_from_cli(parameters['memory'])
        loss = LossCLI.from_cli(parameters['loss'])
        optimizer = OptimizerCLI.from_cli(parameters['optimizer'])
        parameters = DeepQAgent.Configuration.from_cli(parameters['parameters'])

        assert isinstance(model, NNMachineModel), f"Model must conform to NNModel"

        return DeepQAgent(model=model,
                          state_encoder=encoder,
                          memory=memory,
                          loss=loss,
                          optimizer=optimizer,
                          parameters=parameters)

