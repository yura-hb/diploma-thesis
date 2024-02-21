
from .criterion import Criterion, Direction, Scale
from .makespan import Makespan
from .tardiness import Tardiness

from functools import partial
from utils import from_cli

key_to_cls = {
    Makespan.key: Makespan,
    Tardiness.key: Tardiness
}


from_cli = partial(from_cli, key_to_class=key_to_cls)