
import torch

from .transition import *
from typing import Dict
from itertools import product


class AddingArcTransition(ForwardTransition):
    """
    Add arc only if the arc is determined
    """

    def construct(self, job: Job) -> torch.Tensor:
        edges = []
        operations_count = 0

        for step_id, work_center_id in enumerate(job.step_idx):
            if step_id + 1 >= len(job.step_idx):
                break

            current_step_op_count = len(job.processing_times[step_id])
            next_step_op_count = len(job.processing_times[step_id + 1])

            result = []

            if current_step_op_count == 1 and next_step_op_count == 1:
                result = [[0, 0]]
            else:
                # If the machine for job was determined, then we don't need to define all edges
                # from current step to the next
                arrived_machine_idx = job.history.arrived_machine_idx

                did_arrive_at_machine = step_id <= job.current_step_idx and arrived_machine_idx[step_id] >= 0
                did_arrive_at_machine_and_arrived_on_next = did_arrive_at_machine and arrived_machine_idx[step_id + 1] >= 0

                if did_arrive_at_machine_and_arrived_on_next:
                    # Operation path is determined
                    result = [
                        [arrived_machine_idx[step_id], arrived_machine_idx[step_id + 1]]
                    ]

            result = torch.Tensor(list(result)).T.int().view(2, -1)
            result[0, :] += operations_count
            result[1, :] += operations_count + current_step_op_count

            operations_count += current_step_op_count

            edges += [result]

        return torch.cat(edges, dim=1)

    @staticmethod
    def from_cli(parameters: Dict):
        return AddingArcTransition()
