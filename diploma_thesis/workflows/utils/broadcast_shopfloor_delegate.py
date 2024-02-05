
import environment

from typing import List


class BroadCastShopFloorDelegate(environment.ShopFloor.Delegate):

    def __init__(self, sub_delegates: List[environment.ShopFloor.Delegate]):
        self.sub_delegates = sub_delegates

    def will_produce(self, job: environment.Job, machine: environment.Machine):
        for delegate in self.sub_delegates:
            delegate.will_produce(job, machine)

    def did_produce(self, job: environment.Job, machine: environment.Machine):
        for delegate in self.sub_delegates:
            delegate.did_produce(job, machine)

    def will_dispatch(self, job: environment.Job, work_center: environment.WorkCenter):
        for delegate in self.sub_delegates:
            delegate.will_dispatch(job, work_center)

    def did_complete(self, job: environment.Job):
        for delegate in self.sub_delegates:
            delegate.did_complete(job)
