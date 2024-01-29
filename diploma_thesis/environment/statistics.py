
from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import List, Callable, Tuple

import pandas as pd
import torch

from environment.job import Job
from environment.shop_floor import ShopFloor
from .utils.production_log_factory import ProductionLogFactory


class Statistics:
    """
    Support class to compute statistics from the history of shop floor.

    Notes:
     - The class represents the snapshot of the shop-floor, i.e. it wouldn't contain any
       changes in the shop-floor after its creation.
    """

    @dataclass
    class Predicate:
        @dataclass
        class WorkerPredicate:
            ...

        @dataclass
        class WorkCenterPredicate(WorkerPredicate):
            work_center_idx: int = None

        @dataclass
        class MachinePredicate(WorkerPredicate):
            work_center_idx: int = None
            machine_idx: int = None

        @dataclass
        class TimePredicate:
            class Kind(StrEnum):
                # Fetch all information before the given timestamp
                less_than = auto()
                # Fetch information for jobs started before timestamp and not completed yet
                cut = auto()
                # Fetch information for jobs started before timestamp and not completed yet or jobs started after
                # the timestamp
                greater_than = auto()

            at: float = -1
            kind: Kind = Kind.greater_than

        worker_predicate: WorkerPredicate = field(default_factory=WorkerPredicate)
        time_predicate: TimePredicate = field(default_factory=TimePredicate)

    def __init__(self, shop_floor: ShopFloor):
        self.shop_floor = shop_floor
        self.shop_floor_history = deepcopy(shop_floor.history)
        self.work_center_history = deepcopy([work_center.history for work_center in shop_floor.work_centers])
        self.machine_history = deepcopy([machine.history for machine in shop_floor.machines])

        self.production_logs = ProductionLogFactory().make(shop_floor)

    def jobs(self, predicate: Predicate = Predicate()) -> List[Job]:
        """
        Fetches all jobs, which are in the system given by the predicate

        Returns: A list of jobs
        """
        logs = self.__filter__(predicate)

        job_ids = logs['job_id'].unique()
        unfinished_job = [self.shop_floor_history.jobs[job_id] for job_id in job_ids]

        return unfinished_job

    def total_make_span(self, predicate: Predicate = Predicate()) -> torch.FloatTensor:
        """
        Computes make span for the shop-floor, i.e. last job completion time.
        """
        logs = self.__filter__(predicate)

        return logs[logs.event == ProductionLogFactory.Event.completed]['moment'].max()

    def utilization_rate(
        self,
        interval: [float, float] = None,
        predicate: Predicate.MachinePredicate = Predicate.MachinePredicate()
    ) -> torch.FloatTensor:
        """
        Computes utilization rate, i.e. the ratio of runtime to the total time. The metric is defined machine.
        In case higher-level predicate (work-center or shop-floor) is given, the value for each machine
        is returned.

        Args:
            interval: The interval to compute utilization rate for. If None, [0, makespan] is taken
            predicate: The predicate to select machines for which utilization rate should be computed.

        Return: Utilization rate for machine or machines depending on the predicate.
        """

        _interval = self.__estimate_interval__(interval)

        return self.run_time(interval=_interval, predicate=predicate) / (_interval[1] - _interval[0])

    def run_time(self,
                 interval: [float, float] = None,
                 predicate: Predicate.MachinePredicate = Predicate.MachinePredicate()) -> torch.FloatTensor:
        _interval = self.__estimate_interval__(interval)

        select_predicate = self.Predicate(
            worker_predicate=predicate,
            time_predicate=self.Predicate.TimePredicate(
                kind=self.Predicate.TimePredicate.Kind.less_than, at=_interval[1]
            )
        )

        logs = self.__filter__(select_predicate)

        production_records = logs.set_index(['job_id', 'operation_id'])

        df = production_records[production_records['event'] == ProductionLogFactory.Event.started_processing]
        df = df[['moment']].clip(lower=_interval[0], upper=None)

        ends = production_records[production_records['event'] == ProductionLogFactory.Event.finished_processing]
        ends = ends['moment'].clip(lower=None, upper=_interval[1])

        df['ends'] = ends
        df['ends'] = df['ends'].fillna(_interval[1])

        return (df['ends'] - df['moment']).sum()

    def total_flow_time(self, weighted_by_priority: bool, predicate: Predicate) -> torch.FloatTensor:
        """
        Computes total flow time for the shop-floor, i.e. the sum of durations from job dispatch to job completion.

        Notes:
            - The flow time is computed only for completed jobs.
            - Non-completed jobs have flow time equal to zero.
        """
        job_ids = self.__completed_job_ids__(predicate)

        return self.__reduce_jobs__(job_ids, weighted_by_priority, lambda job: job.flow_time)

    def total_tardiness(self, weighted_by_priority: bool, predicate: Predicate) -> torch.FloatTensor:
        """
        Computes tardiness for the shop-floor, i.e. the sum of durations from job completion to job due.

        Notes:
            - The tardiness is computed only for completed jobs.
            - Non-tardy jobs have tardiness equal to zero.
        """
        job_ids = self.__completed_job_ids__(predicate)

        return self.__reduce_jobs__(job_ids, weighted_by_priority, lambda job: job.tardiness)

    def total_earliness(self, weighted_by_priority: bool, predicate: Predicate) -> torch.FloatTensor:
        """
        Computes earliness for the shop-floor, i.e. the sum of durations from job due to job completion (the opposite of
        tardiness).

        Notes:
            - The earliness is computed only for completed jobs.
            - Non-tardy jobs have tardiness equal to zero.

        """
        job_ids = self.__completed_job_ids__(predicate)

        return self.__reduce_jobs__(job_ids, weighted_by_priority, lambda job: job.earliness)

    def total_number_of_tardy_jobs(
        self,
        weighted_by_priority: bool,
        predicate: Predicate = Predicate()
    ) -> torch.FloatTensor:
        """
        Computes earliness for the shop-floor, i.e. the sum of durations from job due to job completion (the opposite of
        tardiness).

        Notes:
            - The earliness is computed only for completed jobs.
            - Non-tardy jobs have tardiness equal to zero.
        """
        job_ids = self.__completed_job_ids__(predicate)

        return self.__reduce_jobs__(job_ids, weighted_by_priority, lambda job: 1 if job.is_tardy else 0)

    def __filter__(self, predicate: Predicate) -> pd.DataFrame:
        logs = None

        match predicate.time_predicate.kind:
            case predicate.TimePredicate.Kind.less_than:
                logs = self.production_logs[self.production_logs.moment < predicate.time_predicate.at]
            case predicate.TimePredicate.Kind.cut:
                job_ids = self.__started_and_not_completed_job_ids__(predicate.time_predicate.at)

                logs = self.production_logs[self.production_logs.job_id.isin(job_ids)]
            case predicate.TimePredicate.Kind.greater_than:
                job_ids = self.__started_and_not_completed_job_ids__(predicate.time_predicate.at)

                logs = self.production_logs[
                    (self.production_logs.moment > predicate.time_predicate.at) |
                    (self.production_logs.job_id.isin(job_ids))
                ]
            case _:
                raise ValueError(f"Unsupported predicate {predicate}")

        match predicate.worker_predicate:
            case self.Predicate.WorkCenterPredicate(work_center_idx):
                return logs[logs.work_center_idx == work_center_idx]
            case self.Predicate.MachinePredicate(work_center_idx, machine_idx):
                return logs[(logs.work_center_idx == work_center_idx) & (logs.machine_idx == machine_idx)]
            case self.Predicate.WorkerPredicate():
                return logs
            case _:
                raise ValueError(f"Unsupported predicate {predicate}")

    def __started_and_not_completed_job_ids__(self, at: float):
        completed_job_ids_before_moment = self.production_logs[
            (self.production_logs.event == ProductionLogFactory.Event.completed) &
            (self.production_logs.moment <= at)
            ]['job_id'].unique()

        created_job_ids_before_moment = self.production_logs[
            (self.production_logs.event == ProductionLogFactory.Event.created) &
            (self.production_logs.moment <= at)
            ]['job_id'].unique()

        job_ids = set(created_job_ids_before_moment) - set(completed_job_ids_before_moment)

        return job_ids

    def all_criteria(self, predicate) -> pd.DataFrame:
        ...

    def __estimate_interval__(self, interval: Tuple[float, float] = None):
        assert interval is None or (len(interval) == 2 and interval[0] < interval[1]), \
            "Interval must be None or a list of two elements"

        _interval = interval

        if _interval is None:
            _interval = [0, self.total_make_span()]

        return _interval

    def __completed_job_ids__(self, predicate: Predicate):
        logs = self.__filter__(predicate)

        completed = logs[logs.event == ProductionLogFactory.Event.completed]['job_id']

        return completed.index.unique()

    def __reduce_jobs__(self, job_ids, weighted_by_priority: bool, get_value: Callable[[Job], float]) -> torch.FloatTensor:
        value = 0
        total_weights = 0

        for job_id in job_ids:
            job = self.shop_floor_history.jobs[job_id]
            weight = job.priority if weighted_by_priority else 1

            value += get_value(job) * weight
            total_weights += weight

        value = torch.FloatTensor(value)

        if weighted_by_priority:
            return value / total_weights

        return value
