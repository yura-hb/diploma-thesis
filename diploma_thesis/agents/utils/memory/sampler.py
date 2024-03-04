
from typing import Dict, Iterable

from torchrl.data import RandomSampler, SliceSampler


class Sampler:

    def __init__(self, kind: str, parameters: Dict):
        self.kind = kind
        self.parameters = parameters

    def make(self):
        match self.kind:
            case 'random':
                return RandomSampler()
            case 'slice':
                # TorchRL sampler supports only tuple for nested keys
                for key in ['traj_key', 'end_key']:
                    if key in self.parameters and isinstance(self.parameters[key], Iterable):
                        self.parameters[key] = tuple(self.parameters[key])

                return SliceSampler(**self.parameters)

    @staticmethod
    def from_cli(parameters: Dict) -> 'Sampler':
        return Sampler(kind=parameters['kind'], parameters=parameters['parameters'])