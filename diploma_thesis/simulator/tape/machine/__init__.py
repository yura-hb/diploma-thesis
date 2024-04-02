
from functools import partial

from utils import from_cli

from .reward import MachineReward
from .no import No
from .global_tardiness import GlobalTardiness
from .global_decomposed_tardiness import GlobalDecomposedTardiness
from .surrogate_tardiness import SurrogateTardiness
from .global_MDPI_reward import GlobalMDPI
from .surrogate_slack import SurrogateSlack
from .makespan import Makespan

key_to_cls = {
    'no': No,
    'global_tardiness': GlobalTardiness,
    'global_decomposed_tardiness': GlobalDecomposedTardiness,
    'global_mdpi': GlobalMDPI,
    'surrogate_tardiness': SurrogateTardiness,
    'surrogate_slack': SurrogateSlack,
    'makespan': Makespan
}


from_cli = partial(from_cli, key_to_class=key_to_cls)