
from dataclasses import dataclass
from typing import Dict


class Phase:
    pass


@dataclass(frozen=True)
class WarmUpPhase(Phase):
    idx: int


@dataclass(frozen=True)
class TrainingPhase(Phase):
    pass


@dataclass(frozen=True)
class EvaluationPhase(Phase):
    pass


def from_cli(parameters: Dict) -> Phase:
    key = parameters['kind']

    match key:
        case 'warm_up':
            return WarmUpPhase(parameters['step'])
        case 'training':
            return TrainingPhase()
        case 'evaluation':
            return EvaluationPhase()
        case _:
            raise ValueError(f'Unknown phase kind: {key}')
