from .action_selector import *


class Greedy(ActionSelector):

    def __call__(self, distribution: torch.FloatTensor) -> Tuple[int, torch.FloatTensor]:
        return torch.argmax(distribution).item(), torch.tensor(1.0)

    @staticmethod
    def from_cli(parameters: Dict):
        return Greedy()
