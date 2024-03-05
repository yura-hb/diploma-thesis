
from utils.multi_value_cli import multi_value_cli
from .cli import from_cli
from .simulation import Simulation


class MultiValueCLITemplate:

    @classmethod
    def from_cli(cls, name: str, logger, parameters: dict) -> [Simulation]:
        return multi_value_cli(parameters, lambda _params: from_cli(name, _params, logger))
