
from typing import Dict
from dataclasses import dataclass
from .estimator import *


class NStep(Estimator):

    @dataclass
    class Configuration:
        discount_factor: float
        lambda_factor: float
        n: int
        trace_lambda: float | None

        @staticmethod
        def from_cli(parameters: Dict):
            return NStep.Configuration(
                discount_factor=parameters.get('discount_factor', 0.99),
                lambda_factor=parameters.get('lambda_factor', 0.95),
                n=parameters.get('n', 1),
                # vtrace_clip=parameters.get('vtrace_clip', None),
                trace_lambda=parameters.get('trace_lambda', None)
            )

    def __init__(self, configuration: Configuration):
        super().__init__()

        self.configuration = configuration

    def discount_factor(self) -> float:
        return self.configuration.discount_factor ** self.configuration.n

    def update_returns(self, records: List[Record]) -> List[Record]:
        pass

# def recursive(trajectory, V, done, args, compute_target_policy):
#     require_policy = args.off_policy
#     end = trajectory[-1]
#
#     for _ in (range(args.n) if done else [0]):
#         transition = trajectory[0]
#
#         policy = None
#
#         if require_policy:
#             policy = compute_target_policy(V)
#
#         G = 0
#
#         if not done:
#             G = V[end.next_state]
#
#         for index in reversed(range(args.n)):
#             tmp = trajectory[index]
#
#             if tmp.terminal:
#                 continue
#
#             weight = 1
#
#             if require_policy:
#                 weight = policy[tmp.state, tmp.action] / tmp.action_prob
#
#             if args.vtrace_clip is not None:
#                 weight = np.clip(weight, -args.vtrace_clip, args.vtrace_clip)
#
#             G = weight * (tmp.reward + args.gamma * G) + (1 - weight) * V[tmp.state]
#
#             if args.trace_lambda is not None:
#                 G_1 = tmp.reward + (1 - tmp.done) * args.gamma * V[tmp.next_state]
#
#                 G = (1 - args.trace_lambda) * G_1 + args.trace_lambda * G
#
#         if done:
#             trajectory.append(Transition(0.0, 0.0, 0.0, 0.0, 0.0, False, 0.0, True))
#
#         V[transition.state] += args.alpha * (G - V[transition.state])
