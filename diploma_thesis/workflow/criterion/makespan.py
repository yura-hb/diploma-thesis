
from .criterion import Criterion
from environment import Statistics


class Makespan(Criterion):

    @classmethod
    @property
    def key(cls) -> str:
        return 'makespan'

    def compute(self, statistics):
        predicate = Statistics.Predicate()

        return statistics.total_make_span(predicate)
