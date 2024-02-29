

from dataclasses import dataclass
from typing import List
from agents.utils.memory import Record


class Return:

    @dataclass
    class Configuration:
        discount_factor: float
        lambda_factor: float
        n: int
        vtrace_clip: float
        trace_lambda: float

    def __init__(self, configuration: Configuration):
        pass

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
