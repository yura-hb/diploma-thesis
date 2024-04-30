
from .criterion import Criterion
from environment import Statistics


class Tardiness(Criterion):

    @classmethod
    @property
    def key(cls) -> str:
        return 'tardiness'

    def compute(self, statistics):
        if self.at is None:
            predicate = Statistics.Predicate(limit=self.limit)
        else:
            tp = Statistics.Predicate.TimePredicate
            predicate = Statistics.Predicate(time_predicate=tp(self._at, kind=tp.Kind.less_than), limit=self.limit)

        return statistics.total_tardiness(predicate=predicate)
