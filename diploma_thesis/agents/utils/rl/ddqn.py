import tensordict

from agents.utils.memory import Record
from agents.utils.policy import Policy
from .dqn import DeepQTrainer


class DoubleDeepQTrainer(DeepQTrainer):

    def estimate_q(self, model: Policy, batch: Record | tensordict.TensorDictBase):
        actions = self.__get_action_values__(model, batch.next_state, None)

        best_actions = actions.max(dim=-1).indices

        target = self.__get_action_values__(self.target_model, batch.next_state, best_actions)

        q = batch.reward.squeeze() + self.return_estimator.discount_factor * target * (1 - batch.done.squeeze().int())

        return q
