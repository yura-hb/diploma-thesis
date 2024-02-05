
import simpy

import environment

from environment import Job, WorkCenter, Machine


class Agent(environment.Agent):
    """

    """

    def __init__(self):
        super().__init__()

    def learn(self):
        pass

    def evaluate(self):
        pass

    # Response from shopfloor

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
