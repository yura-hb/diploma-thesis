import torch

from .encoder import *


class DJSPGraphEncoder(GraphStateEncoder):

    def __encode__(self, parameters: StateEncoder.Input) -> State:
        if parameters.graph is None:
            raise ValueError("Graph is not provided")

        graph = parameters.graph

        job_ids = graph[Graph.JOB_INDEX_MAP][:, 0].unique()

        states = []

        for job_id in job_ids:
            job = parameters.machine.shop_floor.job(job_id)
            n_ops = job.processing_times.numel()

            start_times = self.__fill_job_matrix__(job, job.history.started_at)

            status = torch.ones_like(job.step_idx)
            status[job.current_step_idx - 1:] = 2

            status = self.__fill_job_matrix__(job, status)

            result = torch.vstack([
                # Job id
                torch.full((n_ops,), fill_value=job_id.item()),
                # Operation id
                torch.arange(n_ops),
                # Work center id
                job.step_idx.repeat_interleave(job.processing_times.shape[1]).view(-1),
                # Machine id
                torch.arange(job.processing_times.shape[1]).repeat(job.processing_times.shape[0]).view(-1),
                # Dispatched At
                torch.full((n_ops,), fill_value=job.history.dispatched_at.item()),
                # Operation processing time
                job.processing_times.view(-1),
                # Process start time
                start_times.view(-1),
                # Status of the job, 0 not scheduled, 1 scheduled, 2 is being scheduled right now
                status.view(-1),
                # Due time of the job as the alternative to priority
                torch.full((n_ops,), fill_value=job.due_at.item()),
                # Priority of the job
                torch.full((n_ops,), fill_value=job.priority.item()),
            ])

            states += [result]

        states = torch.cat(states, dim=1).T if len(states) > 0 else torch.tensor([]).view(0, 8)
        graph[Graph.OPERATION_KEY].x = states

        return State(graph=graph, batch_size=[])

    @classmethod
    def from_cli(cls, parameters: dict):
        return DJSPGraphEncoder(**cls.base_parameters_from_cli(parameters))
