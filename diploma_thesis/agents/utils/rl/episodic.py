
from abc import ABCMeta
from .rl import *


class EpisodicTrainer(RLTrainer, metaclass=ABCMeta):

    def __init__(self, memory: Memory, loss: Loss, optimizer: Optimizer, return_estimator: ReturnEstimator):
        super().__init__(memory, loss, optimizer, return_estimator)
        self.episodes = 0

    def store(self, record: Record | List[Record]):
        if isinstance(record, Record):
            raise ValueError('Reinforce does not support single records')

        updated = self.return_estimator.update_returns(record)
        updated = torch.stack(updated, dim=0)
        updated.info['episode'] = torch.full(updated.reward.shape, self.episodes, device=updated.reward.device)

        self.episodes += 1

        self.memory.store(updated)
