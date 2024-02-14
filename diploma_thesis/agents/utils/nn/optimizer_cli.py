
import torch

from dataclasses import dataclass


class OptimizerCLI:

    @dataclass
    class Configuration:
        kind: str
        parameters: dict

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.step_idx = 0
        self._optimizer = None

    def connect(self, parameters):
        self._optimizer = self.__make_optimizer__(parameters)

    def zero_grad(self):
        self._optimizer.zero_grad()

    def step(self):
        self.step_idx += 1

        self._optimizer.step()

    @property
    def step_count(self):
        return self.step_idx

    @property
    def is_connected(self):
        return self._optimizer is not None

    def __make_optimizer__(self, parameters):
        cls = None

        match self.configuration.kind:
            case 'adam':
                cls = torch.optim.Adam
            case 'sgd':
                cls = torch.optim.SGD
            case _:
                raise ValueError(f'Unknown optimizer kind: {self.configuration.kind}')

        return cls(parameters, **self.configuration.parameters)

    @staticmethod
    def from_cli(parameters):
        return OptimizerCLI(OptimizerCLI.Configuration(
            kind=parameters['kind'],
            parameters=parameters.get('parameters', dict())
        ))
