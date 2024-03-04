import tensordict
import torch

from .dqn import DeepQTrainer
from agents.utils.memory import Record
from agents.utils.policy import Policy


class DoubleDeepQTrainer(DeepQTrainer):

    def estimate_q(self, model: Policy, batch: Record | tensordict.TensorDictBase):
        _, actions = model.predict(batch.next_state)
        orig_q = actions[range(batch.shape[0]), batch.action]

        best_actions = actions.max(dim=-1).indices

        target = self.target_model.predict(batch.next_state)[range(batch.shape[0]), best_actions]

        q = batch.reward + self.return_estimator.discount_factor * target * (1 - batch.done)
        actions[range(batch.shape[0]), batch.action] = q

        td_error = torch.square(orig_q - q)

        return actions, td_error
