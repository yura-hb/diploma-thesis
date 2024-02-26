
from dataclasses import dataclass

from agents.base import Graph
from environment import WorkCenter, Job


@dataclass
class Input:
    job: Job
    work_center: WorkCenter
    graph: Graph | None
