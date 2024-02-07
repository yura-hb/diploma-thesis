
from dataclasses import dataclass
from environment import Job, Machine
from typing import List


@dataclass
class Input:
    job: Job
    work_center_idx: int
    machines: List[Machine]
