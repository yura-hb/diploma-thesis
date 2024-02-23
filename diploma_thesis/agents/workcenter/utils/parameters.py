
from dataclasses import dataclass

from environment import WorkCenter, Job


@dataclass
class Input:
    job: Job
    work_center: WorkCenter
