from .encoder import *


class HierarchicalGraphEncoder(GraphStateEncoder):

    def __init__(self, include_due_dates: bool = False, **kwargs):
        self.include_due_dates = include_due_dates

        super().__init__(**kwargs)

    def __encode__(self, parameters: StateEncoder.Input) -> State:
        if parameters.graph is None:
            raise ValueError("Graph is not provided")

        graph = parameters.graph
        job_ids = graph[Graph.JOB_INDEX_MAP][:, 0].unique()
        states = []

        for job_id in job_ids:
            job = parameters.machine.shop_floor.job(job_id)
            completions_times, mean_completion_time, _ = self.__estimate_completion_times__(job)

            status = torch.zeros_like(job.step_idx)
            status[:job.current_step_idx] = 1
            status = self.__fill_job_matrix__(
                job,
                status,
                initial_matrix=-torch.ones_like(job.processing_times),
                until_current_step=False
            )

            values = [completions_times.view(-1), status.view(-1)]

            if self.include_due_dates:
                slack_times = job.due_at - mean_completion_time

                states += [torch.vstack(values + [slack_times.view(-1)])]
            else:
                states += [torch.vstack(values)]

        states = torch.cat(states, dim=1).T if len(states) > 0 else torch.tensor([]).view(0, self.include_due_dates + 2)
        graph[Graph.OPERATION_KEY].x = states

        return State(graph=graph, batch_size=[])

    @classmethod
    def from_cli(cls, parameters: dict):
        return HierarchicalGraphEncoder(
            include_due_dates=parameters.get('include_due_dates', False),
            **cls.base_parameters_from_cli(parameters)
        )
