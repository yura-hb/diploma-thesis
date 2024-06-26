import copy
from typing import Dict

import tensordict
import torch
from torch.optim.swa_utils import AveragedModel, get_ema_avg_fn

from agents.utils.memory import NotReadyException
from agents.utils.return_estimator import ValueFetchMethod
from agents.utils.rl.rl import *


class DeepQTrainer(RLTrainer):
    @dataclass
    class Configuration:
        decay: float = 0.99
        update_steps: int = 10
        epochs: int = 1

        @staticmethod
        def from_cli(parameters: Dict):
            return DeepQTrainer.Configuration(
                decay=parameters.get('decay', 0.99),
                update_steps=parameters.get('update_steps', 20),
                epochs=parameters.get('epochs', 1)
            )

    def __init__(self, configuration: Configuration, *args, **kwargs):
        super().__init__(*args, is_episodic=False, **kwargs)

        self.return_estimator.update(ValueFetchMethod.ACTION)
        self._target_model: AveragedModel | None = None
        self.configuration = configuration

    def configure(self, model: Policy):
        super().configure(model)

        avg_fn = get_ema_avg_fn(self.configuration.decay)

        self._target_model = copy.deepcopy(model) #AveragedModel(model.clone(), avg_fn=avg_fn).to(self.device)

    def __train__(self, model: Policy):
        for _ in range(self.configuration.epochs):
            try:
                batch, info = self.storage.sample(device=self.device)
            except NotReadyException:
                return

            with torch.no_grad():
                q_values = self.estimate_q(model, batch)

            def compute_loss():
                actions = self.__get_action_values__(model, batch.state, batch.action)

                weight = torch.tensor(info['_weight']) if '_weight' in info.keys() else torch.ones_like(q_values)
                weight = weight.to(actions.device)

                loss_ = (self.loss(actions, q_values) * weight).mean()
                td_error_ = torch.square(actions - q_values)

                entropy = torch.distributions.Categorical(logits=actions).entropy().mean()

                return loss_, (td_error_, entropy)

            loss, result = self.step(compute_loss, self.optimizer)

            td_error, entropy = result
            td_error_mean = td_error.mean()

            self.record_loss(loss)
            self.record_loss(td_error_mean, key='td_error')
            self.record_loss(entropy, key='entropy')
            self.record_loss(q_values.mean(), key='q_values')

            print(f'loss: {loss}, td_error: {td_error_mean}, entropy: {entropy}')

        with torch.no_grad():
            if self.optimizer.step_count % self.configuration.update_steps == 0:
                self._target_model = copy.deepcopy(model)

            self.storage.update_priority(info['index'], td_error)

    def estimate_q(self, model: Policy, batch: Record | tensordict.TensorDictBase):
        target = self.__get_action_values__(self.target_model, batch.next_state, None)
        target = target.max(dim=-1).values

        q = batch.reward.squeeze() + self.return_estimator.discount_factor * target * (1 - batch.done.squeeze().int())

        return q

    @staticmethod
    def __get_action_values__(model: Policy, state, actions):
        output = model(state)
        _, action_values, _ = model.__fetch_values__(output)

        if actions is None:
            return action_values

        return action_values[range(actions.shape[0]), actions]

    @property
    def target_model(self):
        return self._target_model

    def state_dict(self):
        state_dict = super().state_dict()

        state_dict.update(dict(target_model=self.target_model.state_dict()))

        return state_dict

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)

        if self._target_model is not None:
            self.target_model.load_state_dict(state_dict['target_model'])

    @classmethod
    def from_cli(cls, parameters, **kwargs):
        train_schedule = TrainSchedule.from_cli(parameters)
        configuration = DeepQTrainer.Configuration.from_cli(parameters)

        return cls(configuration=configuration, train_schedule=train_schedule, **kwargs)
