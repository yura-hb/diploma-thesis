
from .action_selector import *
from agents.utils import Phase


class PhaseSelector(ActionSelector):

    def __init__(self, default: ActionSelector, phase_to_action_selector: Dict[Phase, ActionSelector]):
        super().__init__()
        self.phase = None
        self.default = default
        self.phase_to_action_selector = phase_to_action_selector

    def update(self, phase: Phase):
        self.phase = phase

    def __call__(self, distribution: torch.FloatTensor) -> Tuple[int, torch.FloatTensor]:
        if self.phase in self.phase_to_action_selector:
            return self.phase_to_action_selector[self.phase](distribution)

        return self.default(distribution)

    @staticmethod
    def from_cli(parameters: Dict):
        from .cli import from_cli
        from agents.utils.phase import from_cli as phase_from_cli

        default = from_cli(parameters['default'])

        phase_to_action_selector = {
            phase_from_cli(info['phase']): from_cli(info['action_selector'])
            for info in parameters['phases']
        }

        return PhaseSelector(default, phase_to_action_selector)
