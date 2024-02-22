import tensordict

from .dqn import DeepQTrainer
from ..memory import Record


class DoubleDeepQTrainer(DeepQTrainer):

    def estimate_q(self, batch: Record | tensordict.TensorDictBase):
        pass
