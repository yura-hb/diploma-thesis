from .action_selector import *


class UpperConfidenceBound:

    def __init__(self, parameter: float):
        self.parameter = parameter
        self.counts = None

    def __call__(self, distribution: torch.FloatTensor) -> Tuple[int, torch.FloatTensor]:
        if self.counts is None:
            self.counts = torch.zeros_like(distribution)

        ucb = distribution + self.parameter * torch.sqrt(torch.log(self.counts.sum()) / self.counts)
        ucb = torch.nan_to_num(ucb, nan=float('inf'))

        action = torch.argmax(ucb).item()

        self.counts[action] += 1

        # TODO: Derive correct probability

        return action, torch.tensor(1.0)

    @staticmethod
    def from_cli(parameters: Dict):
        return UpperConfidenceBound(parameter=parameters['parameter'])
