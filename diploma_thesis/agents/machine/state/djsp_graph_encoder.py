import torch
from tensordict.prototype import tensorclass

from agents.base.state import GraphState
from .encoder import *


class DJSPGraphEncoder(GraphStateEncoder):

    @tensorclass
    class State(GraphState):
        pass

    def __encode__(self, parameters: StateEncoder.Input) -> State:
        if parameters.graph is None:
            raise ValueError("Graph is not provided")

        graph = parameters.graph

        job_ids = graph.data[Graph.JOB_INDEX_MAP][:, 0].unique()
        prev_ops = 0

        states = []

        for job_id in job_ids:
            job = parameters.machine.shop_floor.job(job_id)
            n_ops = job.processing_times.numel()

            start_times = self.__fill_job_matrix__(job, job.history.started_at)

            status = torch.ones_like(job.step_idx)
            status[job.current_step_idx:] = 2

            status = self.__fill_job_matrix__(job, status)

            result = torch.vstack([
                torch.full((n_ops,), fill_value=job_id.item()),
                torch.arange(n_ops) + prev_ops,
                torch.arange(n_ops),
                torch.full((n_ops,), fill_value=job.history.dispatched_at.item()),
                job.processing_times.view(-1),
                start_times.view(-1),
                status.view(-1),
                torch.full((n_ops,), fill_value=job.priority.item()),
            ])

            states += [result]

            prev_ops += n_ops

        states = torch.cat(states, dim=1).T
        graph.data[Graph.OPERATION_KEY].x = states

        return self.State(graph, batch_size=[])

    @classmethod
    def from_cli(cls, parameters: dict):
        return DJSPGraphEncoder(**cls.base_parameters_from_cli(parameters))
