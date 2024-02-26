
from functools import partial

from utils import from_cli

from .reward import MachineReward
from .no import No
from .global_tardiness_reward import GlobalTardiness
from .global_decomposed_tardiness_reward import GlobalDecomposedTardiness
from .surrogate_tardiness_reward import SurrogateTardinessReward


key_to_cls = {
    'no': No,
    'global_tardiness': GlobalTardiness,
    'global_decomposed_tardiness': GlobalDecomposedTardiness,
    'surrogate_tardiness': SurrogateTardinessReward
}


from_cli = partial(from_cli, key_to_class=key_to_cls)