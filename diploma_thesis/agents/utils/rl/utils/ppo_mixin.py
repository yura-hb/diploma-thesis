from dataclasses import dataclass
from typing import Dict

from abc import ABCMeta
from ..rl import *


@dataclass
class PPOConfiguration:
    value_loss: Loss
    sample_count: float
    policy_step_ratio: float
    entropy_regularization: float
    rollback_ratio: float
    critic_weight: float
    epochs: int
    priority_reduction_ratio: float

    @staticmethod
    def base_parameters_from_cli(parameters: Dict):
        return dict(
            value_loss=Loss.from_cli(parameters['value_loss']),
            sample_count=parameters.get('sample_count', 128),
            policy_step_ratio=parameters.get('policy_step_ratio', 1.0),
            entropy_regularization=parameters.get('entropy_regularization', 0.0),
            rollback_ratio=parameters.get('rollback_ratio', 0.1),
            critic_weight=parameters.get('critic_weight', 1.0),
            epochs=parameters.get('epochs', 1),
            priority_reduction_ratio=parameters.get('priority_reduction_ratio', 1.05)
        )


class PPOMixin(RLTrainer, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, is_episodic=True, **kwargs)

    def __step__(self, batch: Record, model: Policy, configuration: PPOConfiguration):
        def compute_loss():
            output = model(batch.state)
            value, logits, _ = model.__fetch_values__(output)

            actor_loss = self.actor_loss(batch, logits, configuration, self.device)
            critic_loss = configuration.critic_weight * configuration.value_loss(value, batch.info[Record.RETURN_KEY])

            # Want to maximize actor loss and minimize critic loss
            loss = actor_loss - critic_loss
            loss = -loss

            return loss, (actor_loss, critic_loss)

        loss, args = self.step(compute_loss, self.optimizer)
        actor_loss, critic_loss = args

        self.record_loss(loss)
        self.record_loss(actor_loss, key='actor')
        self.record_loss(critic_loss, key='critic')

    def __increase_memory_priority__(self, info, configuration: PPOConfiguration):
        if '_weight' in info:
            self.storage.update_priority(info['index'], info['_weight'] / configuration.priority_reduction_ratio)

    @staticmethod
    def actor_loss(batch, logits, configuration: PPOConfiguration, device):
        rollback_ratio = configuration.rollback_ratio
        policy_ratio = configuration.policy_step_ratio
        entropy_regularization = configuration.entropy_regularization

        distribution = torch.distributions.Categorical(logits=logits)

        range = torch.arange(batch.shape[0], device=device)
        advantages = batch.info[Record.ADVANTAGE_KEY]

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        action_probs = batch.info[Record.POLICY_KEY][range, batch.action.view(-1)]

        weights = distribution.log_prob(batch.action).view(-1) - torch.log(action_probs)
        weights = torch.exp(weights)

        rollback_value = - rollback_ratio * weights

        clipped_weights = torch.clamp(weights,
                                      rollback_value + (1 + rollback_ratio) * (1 - policy_ratio),
                                      rollback_value + (1 + rollback_ratio) * (1 + policy_ratio))

        advantages = torch.min(weights * advantages, clipped_weights * advantages)

        return torch.mean(advantages) + entropy_regularization * distribution.entropy().mean()
