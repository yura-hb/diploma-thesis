
import torch

from dataclasses import dataclass


class LossCLI:

    @dataclass
    class Configuration:
        kind: str
        parameters: dict

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        self.loss = self.__make_loss__()

    def __call__(self, *args, **kwargs):
        return self.loss(*args, **kwargs)

    def __make_loss__(self) -> torch.nn.Module:
        cls: torch.nn.Module = None

        match self.configuration.kind:
            case 'smooth_l1':
                cls = torch.nn.SmoothL1Loss
            case 'mse':
                cls = torch.nn.MSELoss
            case 'cross_entropy':
                cls = torch.nn.CrossEntropyLoss
            case _:
                raise ValueError(f'Unknown loss kind: {self.configuration.kind}')

        return cls(**self.configuration.parameters)

    @staticmethod
    def from_cli(parameters):
        return LossCLI(LossCLI.Configuration(
            kind=parameters['kind'],
            parameters=parameters.get('parameters', dict())
        ))
