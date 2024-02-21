
from .cli import CLITemplate
from .simulation import Simulation
from utils.multi_value_cli import multi_value_cli


class MultiValueCLITemplate(CLITemplate):

    @classmethod
    def from_cli(cls, name: str, logger, parameters: dict) -> [Simulation]:
        return multi_value_cli(parameters, lambda _params: Simulation.from_cli(name, logger, _params))
