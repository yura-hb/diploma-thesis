from typing import List

from agents import MachineInput, WorkCenterInput
from environment import Job, WorkCenter, Machine, WaitInfo
from .simulator import Simulator


class TDSimulator(Simulator):
    """
    A simulator, which estimates returns in Temporal Difference manner and send information for training as soon as
    possible
    """

    def schedule(self, shop_floor_id: int, machine: Machine, now: int) -> Job | WaitInfo:
        parameters = MachineInput(machine, now)

        return self.machine.schedule(parameters).result

    def route(self, shop_floor_id: int, job: Job, work_center_idx: int, machines: List[Machine]) -> 'Machine | None':
        parameters = WorkCenterInput(job, work_center_idx, machines)

        return self.work_center.schedule(parameters).result

    def did_start_simulation(self, shop_floor_id: int):
        pass

    def will_produce(self, shop_floor_id: int, job: Job, machine: Machine):
        pass

    def did_produce(self, shop_floor_id: int, job: Job, machine: Machine):
        pass

    def will_dispatch(self, shop_floor_id: int, job: Job, work_center: WorkCenter):
        pass

    def did_dispatch(self, shop_floor_id: int, job: Job, work_center: WorkCenter, machine: Machine):
        pass

    def did_finish_dispatch(self, shop_floor_id: int, work_center: WorkCenter):
        pass

    def did_complete(self, shop_floor_id: int, job: Job):
        pass

