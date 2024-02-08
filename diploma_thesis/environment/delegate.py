from abc import abstractmethod, ABCMeta
from environment import Job, Machine, WorkCenter


class Delegate(metaclass=ABCMeta):

    @abstractmethod
    def did_start_simulation(self, shop_floor_id: str):
        """
        Will be triggered after the start of simulation
        """
        ...

    @abstractmethod
    def will_produce(self, shop_floor_id: str, job: Job, machine: Machine):
        """
        Will be triggered before the production of job on machine
        """
        ...

    @abstractmethod
    def did_produce(self, shop_floor_id: str, job: Job, machine: Machine):
        """
        Will be triggered after the production of job on machine
        """
        ...

    @abstractmethod
    def will_dispatch(self, shop_floor_id: str, job: Job, work_center: WorkCenter):
        """
        Will be triggered before dispatch of job on the work-center
        """
        ...

    @abstractmethod
    def did_dispatch(self, shop_floor_id: str, job: Job, work_center: WorkCenter, machine: Machine):
        """
        Will be triggered after the dispatch of job to the machine
        """
        ...

    @abstractmethod
    def did_finish_dispatch(self, shop_floor_id: str, work_center: WorkCenter):
        """
        Will be triggered after the dispatch of job on the work-center
        """
        ...

    @abstractmethod
    def did_complete(self, shop_floor_id: str, job: Job):
        """
        Will be triggered after the completion of job
        """
        ...

    @abstractmethod
    def did_finish_simulation(self, shop_floor_id: str):
        """
        Will be triggered after all jobs have been completed
        """
        ...
