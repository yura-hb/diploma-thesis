
from environment import Job
from torch_geometric.data import HeteroData
from typing import TypeVar, Generic, List


# There are several options to encode graph:

# 1. We can encode only jobs and their operations without introducing machines. In this case, we can mark
#    operations we are interested in by providing a separate value as hint

# 2. Additionally we can encode
#

# 1. Encode all jobs in the queue
# 2. Encode all jobs in the shopfloor

# The rest i

Unit = TypeVar('Unit')


class GraphEncoder(Generic[Unit]):

    def __init__(self):
        pass

    def encode(self, unit: Unit):
        pass

    def __encode__(self, jobs: List[Job], target_operations):

        pass