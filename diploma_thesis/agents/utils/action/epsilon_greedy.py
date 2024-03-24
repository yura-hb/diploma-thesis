from .action_selector import *


class EpsilonGreedy(ActionSelector):

    def __init__(self, epsilon: float, min_epsilon: float, decay_factor: float, decay_steps: int):
        super().__init__()
        self.min_epsilon = min_epsilon
        self.decay_factor = decay_factor
        self.decay_steps = decay_steps
        self.epsilon = epsilon
        self.steps = 0

    def __call__(self, distribution: torch.Tensor) -> Tuple[int, torch.Tensor]:
        self.decay()

        distribution = torch.atleast_1d(distribution)

        action = torch.argmax(distribution).item()
        policy = torch.zeros_like(distribution) + self.epsilon / distribution.size(0)
        policy[action] += 1 - self.epsilon

        if torch.rand(1) < self.epsilon:
            return torch.randint(0, distribution.size(0), (1,)).item(), policy

        return action, policy

    def decay(self):
        self.steps += 1

        if self.steps % self.decay_steps == 0:
            self.epsilon = max(self.epsilon * self.decay_factor, self.min_epsilon)

            self.log_info(f'Set exploration rate to { self.epsilon }. Total decay steps: { self.steps }')

    @staticmethod
    def from_cli(parameters: Dict):
        return EpsilonGreedy(
            epsilon=parameters['epsilon'],
            min_epsilon=parameters.get('min_epsilon', parameters['epsilon']),
            decay_factor=parameters.get('decay_factor', 1.0),
            decay_steps=parameters.get('decay_steps', 10000000)
        )
