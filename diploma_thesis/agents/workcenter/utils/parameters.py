
from dataclasses import dataclass

from agents.base import Graph
from environment import WorkCenter, Job

from tensordict import TensorDict


@dataclass
class Input:
    job: Job
    work_center: WorkCenter
    graph: Graph | None
    memory: TensorDict | None
