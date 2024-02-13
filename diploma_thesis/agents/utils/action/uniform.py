from .action_selector import *


class Uniform(ActionSelector):

    def __call__(self, distribution: torch.FloatTensor) -> Tuple[int, torch.FloatTensor]:
        action = torch.randint(distribution.size(0), (1,)).item()

        return action, 1.0 / distribution.size(0)

    @staticmethod
    def from_cli(parameters: Dict):
        return Uniform()
