
from .transition import ScheduleTransition
from .complete import CompleteTransition
from .machine_compressed import MachineCompressedTransition
from .operation_compressed import OperationCompressedTransition
from utils import from_cli
from functools import partial

key_to_class = {
    'complete': CompleteTransition,
    'machine_compressed': MachineCompressedTransition,
    'operation_compressed': OperationCompressedTransition
}

from_cli = partial(from_cli, key_to_class=key_to_class)


