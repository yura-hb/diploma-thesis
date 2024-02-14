
from .reward import MachineReward

from .global_tardiness_reward import GlobalTardiness
from .no import No

key_to_cls = {
    'global_tardiness': GlobalTardiness,
    'no': No
}


def from_cli(parameters) -> MachineReward:
    cls = key_to_cls[parameters['kind']]

    return cls.from_cli(parameters.get('parameters', {}))
