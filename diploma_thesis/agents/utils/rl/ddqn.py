import tensordict

from agents.utils.memory import Record
from agents.utils.policy import Policy
from .dqn import DeepQTrainer


class DoubleDeepQTrainer(DeepQTrainer):

    def estimate_q(self, model: Policy, batch: Record | tensordict.TensorDictBase):
        _, actions = model(batch.next_state)
        best_actions = actions.max(dim=-1).indices
        target = self.target_model(batch.next_state)[1][range(batch.shape[0]), best_actions]

        q = batch.reward + self.return_estimator.discount_factor * target * (1 - batch.done.int())

        return q
