
import torch

from dataclasses import dataclass


@dataclass
class RunConfiguration:
    compile: bool = False
    device: torch.device = torch.device('cpu')

    @staticmethod
    def from_cli(parameters):
        return RunConfiguration(
            compile=parameters.get('compile', False),
            device=torch.device(parameters.get('device', 'cpu'))
        )