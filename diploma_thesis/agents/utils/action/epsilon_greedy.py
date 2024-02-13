
from .action_selector import *


class EpsilonGreedy(ActionSelector):

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, distribution: torch.FloatTensor) -> Tuple[int, torch.FloatTensor]:
        if torch.rand(1) < self.epsilon:
            return torch.randint(0, distribution.size(0), (1,)).item(), self.epsilon / distribution.size(0)

        return torch.argmax(distribution).item(), 1 - self.epsilon

    @staticmethod
    def from_cli(parameters: Dict):
        return EpsilonGreedy(parameters['epsilon'])
