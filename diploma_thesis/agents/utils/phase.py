
from dataclasses import dataclass


class Phase:
    pass


@dataclass
class WarmUpPhase(Phase):
    idx: int


@dataclass
class TrainingPhase(Phase):
    pass


@dataclass
class EvaluationPhase(Phase):
    pass
