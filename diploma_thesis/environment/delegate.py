import environment

from abc import abstractmethod, ABCMeta
from environment import Job, Machine, WorkCenter
from dataclasses import dataclass


@dataclass
class DelegateContext:
    shop_floor: 'environment.ShopFloor'
    moment: float


class Delegate(metaclass=ABCMeta):

    def did_start_simulation(self, context: DelegateContext):
        """
        Will be triggered after the start of simulation
        """
        pass

    def will_produce(self, context: DelegateContext, job: Job, machine: Machine):
        """
        Will be triggered before the production of job on machine
        """
        pass

    def did_produce(self, context: DelegateContext, job: Job, machine: Machine):
        """
        Will be triggered after the production of job on machine
        """
        pass

    def will_dispatch(self, context: DelegateContext, job: Job, work_center: WorkCenter):
        """
        Will be triggered before dispatch of job on the work-center
        """
        pass

    def did_dispatch(self, context: DelegateContext, job: Job, work_center: WorkCenter, machine: Machine):
        """
        Will be triggered after the dispatch of job to the machine
        """
        pass

    def did_finish_dispatch(self, context: DelegateContext, work_center: WorkCenter):
        """
        Will be triggered after the dispatch of job on the work-center
        """
        pass

    def did_complete(self, context: DelegateContext, job: Job):
        """
        Will be triggered after the completion of job
        """
        pass

    def did_breakdown(self, context: DelegateContext, machine: Machine, repair_time: float):
        """
        Will be triggered after the breakdown of machine
        """
        pass

    def did_repair(self, context: DelegateContext, machine: Machine):
        """
        Will be triggered after the repair of machine
        """
        pass

    def did_finish_simulation(self, context: DelegateContext):
        """
        Will be triggered after all jobs have been completed
        """
        pass
