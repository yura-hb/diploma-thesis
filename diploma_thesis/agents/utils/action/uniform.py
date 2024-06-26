import torch

from .action_selector import *


class Uniform(ActionSelector):

    def __call__(self, distribution: torch.Tensor) -> Tuple[int, torch.Tensor]:
        distribution = torch.atleast_1d(distribution)

        action = torch.randint(distribution.size(0), (1,)).item()
        policy = torch.zeros_like(distribution) + 1.0 / distribution.size(0)

        return action, policy

    @staticmethod
    def from_cli(parameters: Dict):
        return Uniform()
