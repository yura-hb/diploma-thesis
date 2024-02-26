from dataclasses import dataclass
from enum import StrEnum, auto
from functools import reduce
from typing import List

import pandas as pd
import torch

import environment


class LogEvent(StrEnum):
    created = auto()
    dispatched = auto()
    arrived_at_work_center = auto()
    arrived_at_machine = auto()
    started_processing = auto()
    finished_processing = auto()
    machine_decision = auto()
    work_center_decision = auto()
    machine_breakdown = auto()
    machine_repair = auto()
    completed = auto()


class ProductionLogFactory:

    @dataclass
    class ProductionLog:
        job_id: int
        operation_id: int
        work_center_idx: int
        machine_idx: int
        event: str
        moment: float

    def make(self, shop_floor: 'environment.ShopFloor') -> pd.DataFrame:
        """
        Completes production logs for the shop-floor.

        Args:
            shop_floor: The shop-floor at moment $t$ for which the logs should be made

        Returns: pandas.DataFrame representing production logs
        """
        jobs = shop_floor.history.jobs

        result = [self.__make_production_logs_from_job__(job) for _, job in jobs.items()]
        result += self.__make_shop_floor_events_logs__(shop_floor)
        result = reduce(lambda x, y: x + y, result, [])
        result = [log.__dict__ for log in result]

        df = pd.DataFrame(result)
        df = df.astype({
            'job_id': 'int32',
            'operation_id': 'int32',
            'work_center_idx': 'int32',
            'machine_idx': 'int32',
            'event': pd.CategoricalDtype(categories=LogEvent.__members__.values(), ordered=True),
            'moment': 'float32'
        })

        started_at = shop_floor.history.started_at

        if torch.is_tensor(started_at):
            started_at = started_at.item()

        df['moment'] -= started_at

        return df

    def __make_production_logs_from_job__(self, job: environment.Job) -> List[ProductionLog]:
        result = []

        result += [self.ProductionLog(job.id, 0, -1, -1, LogEvent.created, job.history.created_at)]
        result += [self.ProductionLog(job.id, 0, -1, -1, LogEvent.dispatched, job.history.dispatched_at)]

        if job.current_step_idx >= 0:
            for i in range(job.current_step_idx + 1):
                arrived_at = job.history.arrived_at_work_center[i]

                if arrived_at >= 0:
                    result += [self.ProductionLog(
                        job.id, i, job.step_idx[i], -1, LogEvent.arrived_at_work_center, arrived_at
                    )]

                arrived_at = job.history.arrived_at_machine[i]

                if arrived_at >= job.history.arrived_at_machine[i]:
                    result += [self.ProductionLog(
                        job.id, i, job.step_idx[i], job.history.arrived_machine_idx[i],
                        LogEvent.arrived_at_machine, arrived_at
                    )]

                started_at = job.history.started_at[i]

                if started_at >= 0:
                    result += [self.ProductionLog(
                        job.id, i, job.step_idx[i], job.history.arrived_machine_idx[i],
                        LogEvent.started_processing, started_at
                    )]

                if finished_at := job.history.finished_at[i]:
                    result += [self.ProductionLog(
                        job.id, i, job.step_idx[i], job.history.arrived_machine_idx[i],
                        LogEvent.finished_processing, finished_at
                    )]

        if job.is_completed:
            result += [self.ProductionLog(job.id, 0, -1, -1, LogEvent.completed, job.history.completed_at)]

        return result

    def __make_shop_floor_events_logs__(self, shop_floor: 'environment.ShopFloor') -> List[ProductionLog]:
        result = []

        result += [self.__make_breakdown_logs_from_machine__(machine) for machine in shop_floor.machines]
        result += [self.__make_decision_time_logs_from_machine__(machine) for machine in shop_floor.machines]
        result += [
            self.__make_decision_time_logs_from_work_center__(work_center) for work_center in shop_floor.work_centers
        ]

        return result

    def __make_breakdown_logs_from_machine__(self, machine: environment.Machine) -> List[ProductionLog]:
        result = []

        for i in range(machine.history.breakdown_start_at.shape[0]):
            result += [self.ProductionLog(
                -1, -1,
                work_center_idx=machine.work_center_idx,
                machine_idx=machine.machine_idx,
                event=LogEvent.machine_breakdown,
                moment=machine.history.breakdown_start_at[i])
            ]

        for i in range(machine.history.breakdown_end_at.shape[0]):
            result += [self.ProductionLog(
                -1, -1,
                work_center_idx=machine.work_center_idx,
                machine_idx=machine.machine_idx,
                event=LogEvent.machine_repair,
                moment=machine.history.breakdown_end_at[i])
            ]

        return result

    def __make_decision_time_logs_from_machine__(self, machine: environment.Machine) -> List[ProductionLog]:
        result = []

        for i in range(machine.history.decision_times.shape[0]):
            result += [
                self.ProductionLog(
                    -1, -1,
                    work_center_idx=machine.work_center_idx,
                    machine_idx=machine.machine_idx,
                    event=LogEvent.machine_decision,
                    moment=machine.history.decision_times[i]
                )
            ]

        return result

    def __make_decision_time_logs_from_work_center__(self, work_center: environment.WorkCenter) -> List[ProductionLog]:
        result = []

        for i in range(work_center.history.decision_times.shape[0]):
            result += [self.ProductionLog(
                -1, -1,
                work_center_idx=work_center.state.idx,
                machine_idx=-1,
                event=LogEvent.work_center_decision,
                moment=work_center.history.decision_times[i]
            )]

        return result
