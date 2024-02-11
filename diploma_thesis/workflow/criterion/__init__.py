
from .criterion import Criterion, Direction, Scale
from .makespan import Makespan
from .tardiness import Tardiness

key_to_cls = {
    Makespan.key: Makespan,
    Tardiness.key: Tardiness
}


def from_cli(parameters: dict) -> list[Criterion]:
    cls = key_to_cls[parameters['kind']]

    return cls.from_cli(parameters['parameters'])
