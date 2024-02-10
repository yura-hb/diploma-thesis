
import multiprocessing as mp
from dataclasses import dataclass, asdict
from enum import StrEnum, auto
from functools import reduce
from typing import List

import pandas as pd
import torch
from joblib import Parallel, delayed

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
        cpu_count = mp.cpu_count()
        jobs = shop_floor.history.jobs

        result = Parallel(n_jobs=cpu_count, batch_size=len(jobs) // cpu_count)(
            delayed(self.__make_production_logs_from_job__)(job) for _, job in jobs.items()
        )

        result = reduce(lambda x, y: x + y, result, [])

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
                def new_log(machine_idx: int, event: str, moment: float):
                    return self.ProductionLog(
                        job.id, i, work_center_idx=job.step_idx[i], machine_idx=machine_idx, event=event, moment=moment
                    )

                if arrived_at := job.history.arrived_at_work_center[i]:
                    result += [new_log(-1, LogEvent.arrived_at_work_center, arrived_at)]

                if arrived_at := job.history.arrived_at_machine[i]:
                    result += [new_log(job.history.arrived_machine_idx[i], LogEvent.arrived_at_work_center, arrived_at)]

                if started_at := job.history.started_at[i]:
                    result += [new_log(job.history.arrived_machine_idx[i], LogEvent.started_processing, started_at)]

                if finished_at := job.history.finished_at[i]:
                    result += [new_log(job.history.arrived_machine_idx[i], LogEvent.finished_processing, finished_at)]

        if job.is_completed:
            result += [self.ProductionLog(job.id, 0, -1, -1, LogEvent.completed, job.history.completed_at)]

        # Convert to dict to speed up further computations
        return [asdict(log) for log in result]
