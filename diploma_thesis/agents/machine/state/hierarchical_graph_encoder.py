from .encoder import *


class HierarchicalGraphEncoder(GraphStateEncoder):

    def __encode__(self, parameters: StateEncoder.Input) -> State:
        if parameters.graph is None:
            raise ValueError("Graph is not provided")

        graph = parameters.graph
        job_ids = graph[Graph.JOB_INDEX_MAP][:, 0].unique()
        states = []

        for job_id in job_ids:
            job = parameters.machine.shop_floor.job(job_id)
            completions_times = self.__fill_job_matrix__(job, job.history.finished_at)

            for j in range(job.current_step_idx, len(job.step_idx)):
                if j == 0:
                    moment = parameters.machine.shop_floor.now
                else:
                    moment = completions_times[j-1]

                completions_times[j] = moment + job.processing_times[j].min()

            status = torch.ones_like(job.step_idx)
            status = self.__fill_job_matrix__(job, status)

            states += [torch.vstack([completions_times.view(-1), status.view(-1)])]

        states = torch.cat(states, dim=1).T if len(states) > 0 else torch.tensor([]).view(0, 2)
        graph[Graph.OPERATION_KEY].x = states

        return State(graph=graph, batch_size=[])

    @classmethod
    def from_cli(cls, parameters: dict):
        return HierarchicalGraphEncoder(**cls.base_parameters_from_cli(parameters))
