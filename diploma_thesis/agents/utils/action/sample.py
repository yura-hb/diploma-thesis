import dis

import torch.distributions

from .action_selector import *


class Sample(ActionSelector):

    def __init__(self, is_distribution: bool = True):
        super().__init__()
        self.is_distribution = is_distribution

    def __call__(self, distribution: torch.Tensor) -> Tuple[int, torch.Tensor]:
        distribution = torch.atleast_1d(distribution)

        if self.is_distribution:
            distribution = torch.distributions.Categorical(probs=distribution)
        else:
            distribution = torch.distributions.Categorical(logits=distribution)

        action = distribution.sample().item()

        print("action: ", action, "entropy: ", distribution.entropy().item(), distribution.probs)

        return action, distribution.probs

    @staticmethod
    def from_cli(parameters: Dict):
        return Sample(is_distribution=parameters['is_distribution'])
