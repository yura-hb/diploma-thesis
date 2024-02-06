from typing import List

from environment import Job, WorkCenter, Machine, WaitInfo
from .simulator import Simulator


class EpisodicSimulator(Simulator):
    """
    A simulator, which launches several shop=floors in parallel and simulates them until terminating conditions are met.
    During the process of the simulation the whole episode of environment is recorded.

    After the simulation is finished returns are estimated and passed to the agent for training.
    """

    def schedule(self, shop_floor_id: int, machine: Machine, now: int) -> Job | WaitInfo:
        state = self.machine.encode_state(machine)

        return self.machine.schedule(state)

    def did_start_simulation(self, shop_floor_id: int):
        pass

    def route(self, shop_floor_id: int, job: Job, work_center_idx: int, machines: List[Machine]) -> 'Machine | None':
        state = self.work_center.encode_state(job, work_center_idx, machines)

        return self.work_center.schedule(state)

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

