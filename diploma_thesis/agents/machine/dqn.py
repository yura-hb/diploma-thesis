
import torch

from .machine import *
from typing import Dict
from agents.utils import TrainingPhase, OptimizerCLI, LossCLI

from dataclasses import dataclass


class DeepQAgent(Machine):

    @dataclass
    class Configuration:
        gamma: float
        decay: float = 0.99
        update_steps: int = 10

        @staticmethod
        def from_cli(parameters: Dict):
            return DeepQAgent.Configuration(
                gamma=parameters['gamma'],
                decay=parameters.get('decay', 0.99),
                update_steps=parameters.get('update_steps', 10)
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

    def train_step(self):
        if self.phase != TrainingPhase():
            return

        batch = self.memory.sample()

        with torch.no_grad():
            q_values = self.model.values(batch.next_state)

            target = self.target_model.values(batch.next_state)
            target = target.max(dim=1).values

            q = batch.reward + self.parameters.gamma * target * (1 - batch.done)

            q_values[range(len(batch.action)), batch.action] = q

        if not self.optimizer.is_connected:
            self.optimizer.connect(self.model.parameters())

        values = self.model.values(batch.state)
        loss = self.loss(values, q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        if self.optimizer.step_count % self.parameters.update_steps == 0:
            self.target_model.copy_parameters(self.model, self.parameters.decay)

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

