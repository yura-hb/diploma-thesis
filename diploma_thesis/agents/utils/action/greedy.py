import torch

from .action_selector import *


class Greedy(ActionSelector):

    def __call__(self, distribution: torch.Tensor) -> Tuple[int, torch.Tensor]:
        distribution = torch.atleast_1d(distribution)

        print(f'Greedy action {distribution} { torch.softmax(distribution, dim=0) }')

        action = torch.argmax(distribution).item()
        policy = torch.zeros_like(distribution)
        policy[action] = 1.0

        return action, policy

    @staticmethod
    def from_cli(parameters: Dict):
        return Greedy()
