
import torch

from typing import Dict
from dataclasses import dataclass
from .estimator import *


class NStep(Estimator):

    @dataclass
    class Configuration:
        discount_factor: float
        lambda_factor: float
        n: int
        off_policy: bool
        vtrace_clip: float | None

        @staticmethod
        def from_cli(parameters: Dict):
            n = parameters.get('n', 1)
            lambda_factor = parameters.get('lambda_factor', 0.95)

            assert isinstance(lambda_factor, float) or (isinstance(lambda_factor, list) and len(lambda_factor) == n),\
                   "Lambda factor must be a float or a list of floats of size n"

            return NStep.Configuration(
                discount_factor=parameters.get('discount_factor', 0.99),
                lambda_factor=lambda_factor,
                n=n,
                off_policy=parameters.get('off_policy', False),
                vtrace_clip=parameters.get('vtrace_clip', None)
            )

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.configuration = configuration


    @property
    def discount_factor(self) -> float:
        return self.configuration.discount_factor ** self.configuration.n

    def update_returns(self, records: List[Record]) -> List[Record]:
        td_errors = []
        off_policy_weights = []
        lambdas = []

        if isinstance(self.configuration.lambda_factor, list):
            lambdas = torch.tensor(self.configuration.lambda_factor)
        else:
            lambdas = torch.ones(self.configuration.n) * self.configuration.lambda_factor

        lambdas = torch.cumprod(lambdas, dim=0)

        for i in range(len(records)):
            action = records[i].action

            next_state_value = self.get_value(records[i + 1]) if i + 1 < len(records) else 0
            next_state_value *= self.configuration.discount_factor

            td_errors += [records[i].reward + next_state_value - self.get_value(records[i])]

            if self.configuration.off_policy:
                action_probs = torch.nn.functional.softmax(records[i].info[Record.ACTION_KEY], dim=0)
                weight = action_probs[action] / (records[i].info[Record.POLICY_KEY][action] + 1e-10)
                weight = torch.nan_to_num(weight, nan=1.0, posinf=1.0, neginf=1.0)

                if self.configuration.vtrace_clip:
                    weight = torch.clamp(weight, 1 - self.configuration.vtrace_clip, 1 + self.configuration.vtrace_clip)

                off_policy_weights += [weight]
            else:
                off_policy_weights += [1]

        for i in range(len(records)):
            g = records[i].info[Record.VALUE_KEY]
            n = min(self.configuration.n, len(records) - i)

            weights = off_policy_weights[i:i+n]
            weights = torch.cumprod(torch.tensor(weights), dim=0)

            for j in range(n):
                g += td_errors[i + j] * lambdas[j] * weights[j] * self.configuration.discount_factor ** j

            next_value_idx = i + n
            previous_value_idx = next_value_idx - 1

            # Preprocess trajectory for n-step learning
            records[i].reward = g - (self.get_value(records[next_value_idx]) if next_value_idx < len(records) else 0)
            records[i].next_state = records[previous_value_idx].next_state
            records[i].done = records[next_value_idx].done \
                if next_value_idx < len(records) \
                else records[previous_value_idx].done

        return records

    @staticmethod
    def from_cli(parameters: Dict):
        return NStep(NStep.Configuration.from_cli(parameters))
