from abc import abstractmethod, ABCMeta
from environment import Job, Machine, WorkCenter


class Delegate(metaclass=ABCMeta):

    def did_start_simulation(self, shop_floor_id: str):
        """
        Will be triggered after the start of simulation
        """
        pass

    def will_produce(self, shop_floor_id: str, job: Job, machine: Machine):
        """
        Will be triggered before the production of job on machine
        """
        pass

    def did_produce(self, shop_floor_id: str, job: Job, machine: Machine):
        """
        Will be triggered after the production of job on machine
        """
        pass

    def will_dispatch(self, shop_floor_id: str, job: Job, work_center: WorkCenter):
        """
        Will be triggered before dispatch of job on the work-center
        """
        pass

    def did_dispatch(self, shop_floor_id: str, job: Job, work_center: WorkCenter, machine: Machine):
        """
        Will be triggered after the dispatch of job to the machine
        """
        pass

    def did_finish_dispatch(self, shop_floor_id: str, work_center: WorkCenter):
        """
        Will be triggered after the dispatch of job on the work-center
        """
        pass

    def did_complete(self, shop_floor_id: str, job: Job):
        """
        Will be triggered after the completion of job
        """
        pass

    def did_breakdown(self, shop_floor_id: str, machine: Machine, repair_time: float):
        """
        Will be triggered after the breakdown of machine
        """
        pass

    def did_repair(self, shop_floor_id: str, machine: Machine):
        """
        Will be triggered after the repair of machine
        """
        pass

    def did_finish_simulation(self, shop_floor_id: str):
        """
        Will be triggered after all jobs have been completed
        """
        pass
