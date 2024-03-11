from dataclasses import dataclass
from typing import Dict

import tensordict
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

from agents.utils.memory import NotReadyException
from agents.utils.rl.rl import *


class DeepQTrainer(RLTrainer):
    @dataclass
    class Configuration:
        decay: float = 0.99
        update_steps: int = 10
        prior_eps: float = 1e-6

        @staticmethod
        def from_cli(parameters: Dict):
            return DeepQTrainer.Configuration(
                decay=parameters.get('decay', 0.99),
                update_steps=parameters.get('update_steps', 100),
                prior_eps=parameters.get('prior_eps', 1e-6)
            )

    def __init__(self, configuration: Configuration, *args, **kwargs):
        super().__init__(*args, is_episodic=False, **kwargs)

        self._target_models: AveragedModel | None = None
        self.configuration = configuration

    def configure(self, model: Policy):
        super().configure(model)

        self._target_model = AveragedModel(model.clone(), avg_fn=get_ema_avg_fn(self.configuration.decay))

    def __train__(self, model: Policy):
        try:
            batch, info = self.storage.sample(update_returns=False)
        except NotReadyException:
            return

        with torch.no_grad():
            q_values, td_error = self.estimate_q(model, batch)

        _, actions = model.predict(batch.state)
        loss = self.loss(actions, q_values)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        self.record_loss(loss)

        if self.optimizer.step_count % self.configuration.update_steps == 0:
            self._target_model.update_parameters(model)

        with torch.no_grad():
            td_error += self.configuration.prior_eps

            self.storage.update_priority(info['index'], td_error)

    def estimate_q(self, model: Policy, batch: Record | tensordict.TensorDictBase):
        # Note:
        # The idea is that we compute the Q-values only for performed actions. Other actions wouldn't be updated,
        # because there will be zero loss and so zero gradient
        _, actions = model.predict(batch.next_state)
        orig_q = actions.clone()[range(batch.shape[0]), batch.action]

        _, target = self.target_model.predict(batch.next_state)
        target = target.max(dim=1).values

        q = batch.reward + self.return_estimator.discount_factor * target * (1 - batch.done.int())
        actions[range(batch.shape[0]), batch.action] = q

        td_error = torch.square(orig_q - q)

        return actions, td_error

    @property
    def target_model(self):
        return self._target_model.module

    @classmethod
    def from_cli(cls,
                 parameters,
                 memory: Memory,
                 loss: Loss,
                 optimizer: Optimizer,
                 return_estimator: ReturnEstimator):
        train_schedule = TrainSchedule.from_cli(parameters)
        configuration = DeepQTrainer.Configuration.from_cli(parameters)

        return cls(
            configuration=configuration,
            memory=memory,
            loss=loss,
            optimizer=optimizer,
            return_estimator=return_estimator,
            train_schedule=train_schedule
        )
