
import torch

from typing import Tuple
from .sampler import Sampler


class Permutation(Sampler):

    def __init__(self, uneveness: float = 0):
        super().__init__()

        self.uneveness = uneveness
        self.initial_idx = None

    def sample(self, shape: torch.Size | Tuple) -> torch.LongTensor:
        assert len(shape) == 1, "Shape must be a single number"

        idx = torch.randperm(shape[0], generator=self.generator)

        if self.initial_idx is not None:
            initial_work_center_idx = torch.LongTensor([self.initial_idx])
            idx = idx[idx != initial_work_center_idx]
            idx = torch.hstack([initial_work_center_idx, idx])

        if self.uneveness > 0:
            cut_off = torch.randint(0, self.uneveness, (1,))
            cut_off = int(cut_off)

            if cut_off > 0:
                idx = idx[:-cut_off]

        return idx

    def update(self, initial_idx: int = None):
        self.initial_idx = initial_idx

    @staticmethod
    def from_cli(parameters: dict):
        return Permutation(parameters.get('uneveness', 0))
