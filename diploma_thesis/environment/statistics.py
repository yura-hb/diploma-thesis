from copy import deepcopy
from dataclasses import dataclass, field
from enum import StrEnum, auto
from typing import List, Callable, Tuple, Iterable

import pandas as pd
import torch

import environment
import environment.utils as st
from utils.persistence import save, load

INFO_KEY = 'info'
PRODUCTION_LOG_KEY = 'production_logs.feather'


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

    def __init__(self,
                 shop_floor_history: 'environment.ShopFloorHistory',
                 map: 'environment.ShopFloorMap',
                 work_center_history: List['environment.WorkCenterHistory'],
                 machine_history: List['environment.MachineHistory'],
                 production_logs: pd.DataFrame):
        self.shop_floor_history = shop_floor_history
        self.shop_floor_map = map
        self.work_center_history = work_center_history
        self.machine_history = machine_history
        self.production_logs = production_logs

    def jobs(self, predicate: Predicate = Predicate()) -> List[environment.Job]:
        """
        Fetches all jobs, which are in the system given by the predicate

        Returns: A list of jobs
        """
        logs = self.__filter__(predicate)

        job_ids = logs['job_id'].unique()
        job_ids = job_ids[job_ids >= 0]
        unfinished_job = [self.shop_floor_history.job(job_id) for job_id in job_ids]

        return unfinished_job

    def total_make_span(self, predicate: Predicate = Predicate()) -> float:
        """
        Computes make span for the shop-floor, i.e. last job completion time.
        """
        logs = self.__filter__(predicate)

        return logs[logs.event == st.LogEvent.completed]['moment'].max()

    def utilization_rate(
        self,
        time_predicate: Predicate.TimePredicate = Predicate.TimePredicate(),
        predicate: Predicate.MachinePredicate = Predicate.MachinePredicate()
    ) -> float:
        """
        Computes utilization rate, i.e. the ratio of runtime to the total time. The metric is defined for machine.

        Args:
            interval: The interval to compute utilization rate for. If None, [0, makespan] is taken
            time_predicate: The predicate to select time interval for which utilization rate should be computed.
            predicate: The predicate to select machine for which utilization rate should be computed.

        Return: Utilization rate for machine or machines depending on the predicate.
        """

        _interval = self.__estimate_interval__(time_predicate)

        return self.run_time(time_predicate=time_predicate, predicate=predicate) / (_interval[1] - _interval[0])

    def run_time(self,
                 time_predicate: Predicate.TimePredicate = Predicate.TimePredicate(),
                 predicate: Predicate.MachinePredicate = Predicate.MachinePredicate()) -> float:
        """
        Computes total run time, i.e. the time the job was processing on the machine
        """
        _interval = self.__estimate_interval__(time_predicate)

        select_predicate = self.Predicate(worker_predicate=predicate, time_predicate=time_predicate)

        logs = self.__filter__(select_predicate)

        production_records = logs.set_index(['job_id', 'operation_id'])

        df = production_records[production_records['event'] == st.LogEvent.started_processing]
        df = df[['moment']].clip(lower=_interval[0], upper=None)

        ends = production_records[production_records['event'] == st.LogEvent.finished_processing]
        ends = ends['moment'].clip(lower=None, upper=_interval[1])

        df['ends'] = ends
        df['ends'] = df['ends'].fillna(_interval[1])

        return (df['ends'] - df['moment']).sum()

    def total_number_of_processed_operations(
        self,
        time_predicate: Predicate.TimePredicate = Predicate.TimePredicate(),
        predicate: Predicate.MachinePredicate = Predicate.MachinePredicate()
    ) -> float:
        """
        Computes total number of completed jobs on the machine
        """
        _predicate = self.Predicate(time_predicate=time_predicate, worker_predicate=predicate)
        job_ids = self.__job_ids__(st.LogEvent.finished_processing, _predicate)

        return len(job_ids)

    def total_flow_time(self, weighted_by_priority: bool = False, predicate: Predicate = Predicate()) -> float:
        """
        Computes total flow time for the shop-floor, i.e. the sum of durations from job dispatch to job completion.

        Notes:
            - The flow time is computed only for completed jobs.
            - Non-completed jobs have flow time equal to zero.
        """
        job_ids = self.__completed_job_ids__(predicate)

        return self.__reduce_jobs__(job_ids, weighted_by_priority, lambda job: job.flow_time_upon_completion)

    def total_tardiness(self, weighted_by_priority: bool = False, predicate: Predicate = Predicate()) -> float:
        """
        Computes tardiness for the shop-floor, i.e. the sum of durations from job completion to job due.

        Notes:
            - The tardiness is computed only for completed jobs.
            - Non-tardy jobs have tardiness equal to zero.
        """
        job_ids = self.__completed_job_ids__(predicate)

        return self.__reduce_jobs__(job_ids, weighted_by_priority, lambda job: job.tardiness_upon_completion)

    def total_earliness(self, weighted_by_priority: bool = False, predicate: Predicate = Predicate()) -> float:
        """
        Computes earliness for the shop-floor, i.e. the sum of durations from job due to job completion (the opposite of
        tardiness).

        Notes:
            - The earliness is computed only for completed jobs.
            - Non-tardy jobs have tardiness equal to zero.

        """
        job_ids = self.__completed_job_ids__(predicate)

        return self.__reduce_jobs__(job_ids, weighted_by_priority, lambda job: job.earliness_upon_completion)

    def total_number_of_tardy_jobs(
        self,
        weighted_by_priority: bool = False,
        predicate: Predicate = Predicate()
    ) -> float:
        """
        Computes earliness for the shop-floor, i.e. the sum of durations from job due to job completion (the opposite of
        tardiness).

        Notes:
            - The earliness is computed only for completed jobs.
            - Non-tardy jobs have tardiness equal to zero.
        """
        job_ids = self.__completed_job_ids__(predicate)

        return self.__reduce_jobs__(job_ids,
                                    weighted_by_priority,
                                    lambda job: torch.LongTensor([1 if job.is_tardy_upon_completion else 0]))

    def report(self, time_predicate: Predicate.TimePredicate = Predicate.TimePredicate()) -> st.Report:
        report_factory = st.ReportFactory(self, self.shop_floor_map, time_predicate)

        return report_factory.make()

    # Utility methods

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

    def __estimate_interval__(
        self, time_predicate: Predicate.TimePredicate = Predicate.TimePredicate()
    ) -> Tuple[float, float]:
        match time_predicate.kind:
            case self.Predicate.TimePredicate.Kind.less_than:
                return [0, time_predicate.at]
            case self.Predicate.TimePredicate.Kind.cut:
                # Estimate interval from logs
                predicate = self.Predicate(
                    time_predicate=time_predicate,
                    worker_predicate=self.Predicate.WorkerPredicate()
                )
                logs = self.__filter__(predicate)

                lower = logs[logs.event == st.LogEvent.started_processing]['moment'].min()
                upper = logs[logs.event == st.LogEvent.finished_processing]['moment'].max()

                return [lower, upper]
            case self.Predicate.TimePredicate.Kind.greater_than:
                return [time_predicate.at, self.total_make_span()]

    def __started_and_not_completed_job_ids__(self, at: float):
        completed_job_ids_before_moment = self.production_logs[
            (self.production_logs.event == st.LogEvent.completed) &
            (self.production_logs.moment <= at)
            ]['job_id'].unique()

        created_job_ids_before_moment = self.production_logs[
            (self.production_logs.event == st.LogEvent.created) &
            (self.production_logs.moment <= at)
            ]['job_id'].unique()

        job_ids = set(created_job_ids_before_moment) - set(completed_job_ids_before_moment)

        return job_ids

    def __completed_job_ids__(self, predicate: Predicate) -> Iterable[str]:
        logs = self.__filter__(predicate)

        completed = logs[logs.event == st.LogEvent.completed]['job_id']

        return completed.unique()

    def __job_ids__(self, event: st.LogEvent, predicate: Predicate) -> Iterable[str]:
        logs = self.__filter__(predicate)

        completed = logs[logs.event == event]['job_id']

        return completed.unique()

    def __reduce_jobs__(self,
                        job_ids: Iterable[str],
                        weighted_by_priority: bool,
                        get_value: Callable[[environment.Job], float]) -> float:
        value = torch.FloatTensor([0])
        total_weights = 0

        for job_id in job_ids:
            job = self.shop_floor_history.jobs[job_id]
            weight = job.priority if weighted_by_priority else 1
            metric = get_value(job)

            if not torch.is_tensor(metric):
                metric = torch.FloatTensor([metric])

            value += torch.atleast_1d(metric) * weight
            total_weights += weight

        value = torch.FloatTensor(value)

        if weighted_by_priority:
            value /= total_weights

        return value.item()

    # I/O

    def save(self, path: str):
        save(path, self)

    @staticmethod
    def load(path: str):
        return load(path)

    @staticmethod
    def from_shop_floor(shop_floor: 'environment.shop_floor.ShopFloor'):
        shop_floor_history = shop_floor.history
        map = shop_floor.map
        work_center_history = [work_center.history for work_center in shop_floor.work_centers]
        machine_history = [machine.history for machine in shop_floor.machines]
        try:
            production_logs = st.ProductionLogFactory().make(shop_floor)
        except:
            a = 10

        return Statistics(shop_floor_history=deepcopy(shop_floor_history),
                          map=map,
                          work_center_history=deepcopy(work_center_history),
                          machine_history=deepcopy(machine_history),
                          production_logs=production_logs)

