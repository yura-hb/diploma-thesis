
from abc import ABCMeta, abstractmethod

import torch

from agents.base import Graph
from environment import Job


class ScheduleTransition(metaclass=ABCMeta):

    @abstractmethod
    def schedule_implicit(self, job: Job, graph: Graph) -> Graph:
        pass

    @abstractmethod
    def schedule(self, job: Job, graph: Graph) -> Graph:
        pass

    @abstractmethod
    def process(self, job: Job, graph: Graph) -> Graph:
        pass

    @abstractmethod
    def remove(self, job: Job, graph: Graph) -> Graph:
        pass

    # Utils

    def __append_operations_to_scheduled_graph__(self, operations: torch.LongTensor, machine_index: int, graph: Graph):
        graph.data[Graph.MACHINE_KEY][machine_index][Graph.SCHEDULED_KEY] = torch.cat([
            graph.data[Graph.MACHINE_KEY][machine_index][Graph.SCHEDULED_KEY],
            operations
        ], dim=1)

    @classmethod
    def __get_current_machine_index__(cls, graph: Graph, job: Job) -> torch.LongTensor | None:
        if job.is_completed:
            return None

        work_center_id = job.step_idx[job.current_step_idx]
        machine_id = job.current_machine_idx

        return cls.__get_machine_index__(graph, work_center_id, machine_id)

    @classmethod
    def __get_machine_index__(
        cls, graph: Graph, work_center_id: torch.LongTensor, machine_id: torch.LongTensor
    ) -> torch.LongTensor | None:
        if not torch.is_tensor(work_center_id):
            work_center_id = torch.tensor(work_center_id)

        if not torch.is_tensor(machine_id):
            machine_id = torch.tensor(machine_id)

        index = torch.vstack([work_center_id, machine_id])
        index = (graph.data[Graph.MACHINE_INDEX_KEY] == index).all(dim=0)

        return torch.where(index)[0].item()

    def __current_operation_id__(cls, graph: Graph, job: Job) -> torch.Tensor | None:
        return cls.__operation_id__(graph, job, job.current_step_idx, job.current_machine_idx)

    @classmethod
    def __delete_by_first_row__(cls, value: torch.Tensor, tensor: torch.Tensor):
        mask = tensor[0, :] != value

        return tensor[:, mask]

    @staticmethod
    def __operation_id__(
            graph: Graph, job: Job, step_id: torch.LongTensor | int, machine_id: torch.LongTensor | int
    ) -> torch.LongTensor | None:
        if job.id not in graph.data[Graph.JOB_KEY]:
            return None

        prefix_op = torch.tensor(0)

        for i in range(0, step_id):
            prefix_op += len(job.processing_times[i])

        return torch.vstack([job.id, prefix_op + machine_id])
