from dataclasses import dataclass
from typing import Dict

from abc import ABCMeta
from ..rl import *


@dataclass
class PPOConfiguration:
    value_loss: Loss
    sample_ratio: float
    policy_step_ratio: float
    entropy_regularization: float
    update_advantages: bool
    rollback_ratio: float

    @staticmethod
    def base_parameters_from_cli(parameters: Dict):
        return dict(
            value_loss=Loss.from_cli(parameters['value_loss']),
            sample_ratio=parameters.get('sample_ratio', 0.5),
            policy_step_ratio=parameters.get('policy_step_ratio', 1.0),
            entropy_regularization=parameters.get('entropy_regularization', 0.0),
            update_advantages=parameters.get('update_advantages', True),
            rollback_ratio=parameters.get('rollback_ratio', 0.1),
        )


class PPOMixin(RLTrainer, metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, is_episodic=True, **kwargs)

    def __step__(self, batch: Record, model: Policy, configuration: PPOConfiguration):
        range = torch.arange(batch.shape[0], device=self.run_configuration.device)

        value, logits = model(batch.state)
        value = value[range, batch.action]

        loss = self.actor_loss(batch, logits, configuration, self.run_configuration.device)
        # Maximization of negative value is equivalent to minimization
        loss -= configuration.value_loss(value.view(-1), batch.info[Record.RETURN_KEY])

        # Want to maximize
        loss = -loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.record_loss(loss)

    @staticmethod
    def actor_loss(batch, logits, configuration: PPOConfiguration, device):
        rollback_ratio = configuration.rollback_ratio
        policy_ratio = configuration.policy_step_ratio
        entropy_regularization = configuration.entropy_regularization

        distribution = torch.distributions.Categorical(logits=logits)

        range = torch.arange(batch.shape[0], device=device)
        advantages = batch.info[Record.ADVANTAGE_KEY]

        action_probs = batch.info[Record.POLICY_KEY][range, batch.action.view(-1)]

        weights = distribution.log_prob(batch.action).view(-1) - torch.log(action_probs)
        weights = torch.exp(weights)

        rollback_value = - rollback_ratio * weights

        clipped_weights = torch.clamp(weights,
                                      rollback_value + (1 + rollback_ratio) * (1 - policy_ratio),
                                      rollback_value + (1 + rollback_ratio) * (1 + policy_ratio))

        advantages = torch.min(weights * advantages, clipped_weights * advantages)

        return torch.mean(advantages) + entropy_regularization * distribution.entropy().mean()
