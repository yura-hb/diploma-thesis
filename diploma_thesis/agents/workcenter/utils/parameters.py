
from dataclasses import dataclass

from environment import WorkCenter, Job


@dataclass
class Input:
    work_center: WorkCenter
    job: Job
