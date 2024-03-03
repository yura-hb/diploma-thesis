import tensordict
import torch

from .dqn import DeepQTrainer
from agents.utils.memory import Record
from agents.base.model import DeepPolicyModel


class DoubleDeepQTrainer(DeepQTrainer):

    def estimate_q(self, model: DeepPolicyModel, batch: Record | tensordict.TensorDictBase):
        q_values = model.predict(batch.next_state)
        orig_q = q_values[range(batch.shape[0]), batch.action]

        best_actions = q_values.max(dim=-1).indices

        target = self.target_model.predict(batch.next_state)[range(batch.shape[0]), best_actions]

        q = batch.reward + self.return_estimator.discount_factor * target * (1 - batch.done)
        q_values[range(batch.shape[0]), batch.action] = q

        td_error = torch.square(orig_q - q)

        return q_values, td_error
