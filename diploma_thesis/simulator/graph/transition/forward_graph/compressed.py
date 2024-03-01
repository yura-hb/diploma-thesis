
from .transition import *
from typing import Dict
from itertools import product


class CompressedTransition(ForwardTransition):
    """
    Transition where operations between steps are connected to group node and group node is connected to the next step.
    It greatly reduces the number of edges in the job.

    Group nodes are encoded as -step_id - 1.
    """

    def construct(self, job: Job) -> torch.Tensor:
        edge_index = torch.LongTensor([]).view(2, 0)
        operations_count = 0

        for step_id, work_center_id in enumerate(job.step_idx):
            if step_id + 1 >= len(job.step_idx):
                break

            current_step_op_count = len(job.processing_times[step_id])
            next_step_op_count = len(job.processing_times[step_id + 1])
            group_id = -(step_id + 1)

            result = None

            # If the machine for job was determined, then we don't need to define all edges
            # from current step to the next
            arrived_machine_idx = job.history.arrived_machine_idx

            did_arrive_at_machine = step_id <= job.current_step_idx and arrived_machine_idx[step_id] >= 0
            did_arrive_at_machine_and_arrived_on_next = did_arrive_at_machine and arrived_machine_idx[step_id + 1] >= 0

            if did_arrive_at_machine:
                # Selected operation can proceed to any machine in next step
                result = [
                    [arrived_machine_idx[step_id], group_id],
                    *list(product([group_id], list(range(next_step_op_count))))
                ]
            elif did_arrive_at_machine_and_arrived_on_next:
                # Operation path is determined
                result = [
                    [arrived_machine_idx[step_id], group_id],
                    [group_id, arrived_machine_idx[step_id + 1]]
                ]
            else:
                # Any operation can proceed to any operation in next step
                result = [
                    *list(product(list(range(current_step_op_count)), [group_id])),
                    *list(product([group_id], list(range(next_step_op_count))))
                ]

            result = torch.Tensor(list(result)).T.int()
            result[0, result[0, :] >= 0] += operations_count
            result[1, result[1, :] >= 0] += operations_count + current_step_op_count

            operations_count += current_step_op_count

            edge_index = torch.cat([edge_index, result], dim=1)

        return edge_index

    @staticmethod
    def from_cli(parameters: Dict):
        return CompressedTransition()
