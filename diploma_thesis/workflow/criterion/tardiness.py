
from .criterion import Criterion
from environment import Statistics


class Tardiness(Criterion):

    @classmethod
    @property
    def key(cls) -> str:
        return 'tardiness'

    def compute(self, statistics):
        predicate = Statistics.Predicate()

        return statistics.total_tardiness(predicate)
