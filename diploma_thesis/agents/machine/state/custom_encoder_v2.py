import torch

from .encoder import *


class CustomV2GraphEncoder(GraphStateEncoder):

    def __init__(self, norm_factor, include_due_dates: bool = False, **kwargs):
        self.norm_factor = norm_factor
        self.include_due_dates = include_due_dates

        super().__init__(**kwargs)

    def __encode__(self, parameters: StateEncoder.Input) -> State:
        if parameters.graph is None:
            raise ValueError("Graph is not provided")

        graph = parameters.graph
        job_ids = graph[Graph.JOB_INDEX_MAP][:, 0].unique()
        states = []
        is_target = []

        machine_util_rate = torch.nan_to_num(parameters.machine.utilization_rate, nan=0)
        arriving_jobs = len(parameters.machine.arriving_jobs)
        will_arrive_jobs = len(parameters.machine.will_arrive_jobs)
        time_till_available = parameters.machine.time_till_available / self.norm_factor
        expected_tardy_rate = parameters.machine.shop_floor.expected_tardy_rate(parameters.machine.shop_floor.now)

        winq = dict()
        avlm = dict()
        processing_power = dict()

        for job_id in job_ids:
            job = parameters.machine.shop_floor.job(job_id)

            completion_times = self.__estimate_completion_times__(job, parameters.now)
            l_completion_time, mean_completion_time, u_completion_time = completion_times

            completion_rate = self.__fill_job_matrix__(
                job, torch.arange(1, job.step_idx.shape[0] + 1) / job.step_idx.shape[0], until_current_step=False
            )

            slack_times = job.due_at - (mean_completion_time + parameters.now)
            wait_times = job.current_operation_waiting_time_on_machine(parameters.now)
            critical_ratios = 1 - slack_times / (torch.abs(slack_times) + self.norm_factor)

            idx = job.next_work_center_idx.item() if job.next_work_center_idx is not None else -1

            if idx not in winq.keys():
                winq[idx] = parameters.machine.shop_floor.work_in_next_queue(job) / self.norm_factor

            if idx not in avlm.keys():
                avlm[idx] = parameters.machine.shop_floor.average_waiting_in_next_queue(job) / self.norm_factor

            if idx not in processing_power.keys():
                processing_power[idx] = parameters.machine.shop_floor.processing_power_in_next_queue(job)

            placeholder = torch.zeros_like(job.processing_times).view(-1)

            values = [
                job.processing_times.view(-1) / self.norm_factor,

                job.history.arrived_at_machine >= 0,
                job.history.started_at >= 0,
                job.history.finished_at >= 0,

                # l_completion_time.view(-1) / self.norm_factor,
                mean_completion_time.view(-1) / self.norm_factor,
                # u_completion_time.view(-1) / self.norm_factor,
                slack_times.view(-1) / self.norm_factor,

                completion_rate.view(-1),
                critical_ratios.view(-1),

                placeholder + wait_times / self.norm_factor,
                placeholder + machine_util_rate,
                placeholder + arriving_jobs,
                placeholder + will_arrive_jobs,
                placeholder + time_till_available,
                placeholder + expected_tardy_rate,
                placeholder + winq[idx],
                placeholder + avlm[idx],
                placeholder + processing_power[idx]
            ]

            states += [torch.vstack(values)]

            is_in_machine_queue = torch.any(torch.cat([j.id == job_id.view(-1) for j in parameters.machine.queue]))
            is_target += [job.current_step_idx == job.step_idx if is_in_machine_queue else torch.zeros_like(job.step_idx).bool()]

        states = torch.cat(states, dim=1).T if len(states) > 0 else torch.tensor([]).view(0, self.include_due_dates + 2)

        # Preserve system info only in operations from the machine queue. Normalize the values so that their every convex
        # combination is equal can sum to 1.0.
        is_target = torch.cat(is_target) if len(is_target) > 0 else torch.tensor([])

        states[is_target, -9:] /= is_target.sum()
        states[~is_target, -9:] = 0

        graph[Graph.OPERATION_KEY].x = states

        return State(graph=graph, batch_size=[])

    @classmethod
    def from_cli(cls, parameters: dict):
        return CustomV2GraphEncoder(
            norm_factor=parameters.get('norm_factor', 128),
            **cls.base_parameters_from_cli(parameters)
        )
