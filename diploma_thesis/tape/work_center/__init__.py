from .reward import WorkCenterReward
from .no import No
from .global_tardiness_reward import GlobalTardiness


key_to_cls = {
    'global_tardiness': GlobalTardiness,
    'no': No
}


def from_cli(parameters) -> WorkCenterReward:
    cls = key_to_cls[parameters['kind']]

    return cls.from_cli(parameters.get('parameters', {}))
