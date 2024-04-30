from dataclasses import dataclass
from typing import Dict

from abc import ABCMeta

import torch

from ..rl import *



@dataclass
class PPOConfiguration:
    value_loss: Loss
    value_optimizer: Optimizer
    sample_count: float
    policy_step_ratio: float
    entropy_regularization: float
    entropy_decay: float
    rollback_ratio: float
    critic_weight: float
    epochs: int
    priority_reduction_ratio: float

    @staticmethod
    def base_parameters_from_cli(parameters: Dict):
        return dict(
            value_loss=Loss.from_cli(parameters['value_loss']),
            value_optimizer=Optimizer.from_cli(parameters['value_optimizer']) if 'value_optimizer' in parameters else None,
            sample_count=parameters.get('sample_count', 128),
            policy_step_ratio=parameters.get('policy_step_ratio', 1.0),
            entropy_regularization=parameters.get('entropy_regularization', 0.0),
            rollback_ratio=parameters.get('rollback_ratio', 0.0),
            critic_weight=parameters.get('critic_weight', 1.0),
            entropy_decay=parameters.get('entropy_decay', 0.0),
            epochs=parameters.get('epochs', 1),
            priority_reduction_ratio=parameters.get('priority_reduction_ratio', 1.05)
        )


class PPOMixin(RLTrainer, metaclass=ABCMeta):

    def __init__(self, configuration: PPOConfiguration, *args, **kwargs):
        self.configuration = configuration

        super().__init__(*args, is_episodic=True, **kwargs)

    def configure(self, model: Policy):
        super().configure(model)

        if self.configuration.value_optimizer is not None and not self.configuration.value_optimizer.is_connected:
            self.configuration.value_optimizer.connect(model.parameters())

    def __step__(self, batch: Record, model: Policy):
        entropy_reg = self.configuration.entropy_regularization * (self.configuration.entropy_decay ** self.optimizer.step_count)

        self.logger.info(f'Step {self.optimizer.step_count} with entropy regularization {entropy_reg}')

        def compute_loss():
            output = model(batch.state)
            value, logits, _ = model.__fetch_values__(output)

            actor_loss, entropy = self.actor_loss(batch, logits, self.configuration, self.device, entropy_reg)
            r = batch.info[Record.RETURN_KEY]
            r = (r - r.mean()) / (r.std() + 1e-8)

            print(f'Value: {value.view(-1)} Return: {r.view(-1)}')

            critic_loss = self.configuration.critic_weight * self.configuration.value_loss(value, r)

            return actor_loss, critic_loss, entropy

        actor_loss, critic_loss, entropy = compute_loss()

        if self.configuration.value_optimizer is not None:
            self.configuration.value_optimizer.zero_grad()
            self.optimizer.zero_grad()

            def compute_actor_grad():
                loss = -actor_loss

                loss.backward(retain_graph=True)

                return loss

            self.optimizer.step(compute_actor_grad)

            _, critic_loss, _ = compute_loss()

            def compute_critic_grad():
                critic_loss.backward()

                return critic_loss

            self.configuration.value_optimizer.step(compute_critic_grad)
        else:
            def compute_grad():
                loss = actor_loss - critic_loss
                loss = -loss

                loss.backward()

                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(compute_grad)

        self.record_loss(-actor_loss + critic_loss)
        self.record_loss(actor_loss, key='actor')
        self.record_loss(critic_loss, key='critic')
        self.record_loss(entropy, key='entropy')

    def __increase_memory_priority__(self, info):
        if '_weight' in info:
            self.storage.update_priority(info['index'], info['_weight'] / self.configuration.priority_reduction_ratio)

    def state_dict(self):
        result = super().state_dict()

        if self.configuration.value_optimizer is not None:
            result['value_optimizer'] = self.configuration.value_optimizer.state_dict()

        return result

    def load_state_dict(self, state_dict: dict):
        super().load_state_dict(state_dict)

        if self.configuration.value_optimizer is not None:
            self.configuration.value_optimizer.load_state_dict(state_dict['value_optimizer'])

    @staticmethod
    def actor_loss(batch, logits, configuration: PPOConfiguration, device, entropy_regularization=0.0):
        rollback_ratio = configuration.rollback_ratio
        policy_ratio = configuration.policy_step_ratio

        distribution = torch.distributions.Categorical(logits=logits)

        range = torch.arange(batch.shape[0], device=device)
        advantages = batch.info[Record.ADVANTAGE_KEY]

        # Normalize advantages
        # advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_probs = batch.info[Record.POLICY_KEY][range, batch.action.view(-1)]

        weights = distribution.log_prob(batch.action).view(-1) - torch.log(action_probs)
        weights = torch.exp(weights)

        rollback_value = - rollback_ratio * weights

        clipped_weights = torch.clamp(weights,
                                      rollback_value + (1 + rollback_ratio) * (1 - policy_ratio),
                                      rollback_value + (1 + rollback_ratio) * (1 + policy_ratio))

        advantages = torch.min(weights * advantages, clipped_weights * advantages)

        # print(advantages)

        entropy = distribution.entropy().mean()

        return torch.mean(advantages) + entropy_regularization * entropy, entropy
