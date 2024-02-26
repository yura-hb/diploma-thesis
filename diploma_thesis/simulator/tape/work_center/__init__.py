from functools import partial

from utils import from_cli
from .global_tardiness_reward import GlobalTardiness
from .no import No
from .reward import WorkCenterReward

key_to_cls = {
    'global_tardiness': GlobalTardiness,
    'no': No
}


from_cli = partial(from_cli, key_to_class=key_to_cls)
