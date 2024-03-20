from .action_selector import *


class EpsilonGreedy(ActionSelector):

    def __init__(self, epsilon: float):
        super().__init__()
        self.epsilon = epsilon

    def __call__(self, distribution: torch.Tensor) -> Tuple[int, torch.Tensor]:
        distribution = torch.atleast_1d(distribution)

        action = torch.argmax(distribution).item()
        policy = torch.zeros_like(distribution) + self.epsilon / distribution.size(0)
        policy[action] += 1 - self.epsilon

        if torch.rand(1) < self.epsilon:
            return torch.randint(0, distribution.size(0), (1,)).item(), policy

        return action, policy

    @staticmethod
    def from_cli(parameters: Dict):
        return EpsilonGreedy(parameters['epsilon'])
